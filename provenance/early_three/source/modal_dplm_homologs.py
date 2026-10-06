# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.0"]
# ///
"""Generate query-centred DPLM variants on Modal; no training or residue overwrites.

Example (positions are 1-based, inclusive):
  .venv/bin/python -m modal run scripts/modal_dplm_homologs.py \
    --fasta query.fasta --num-variants 50 --target-identity 0.85 \
    --identity-tolerance 0 --fixed-positions 27 --out-dir out/dplm

Identity is ungapped identity to the query, not pairwise pool identity. The target
is rounded to the nearest integer mutation count; tolerance is a fraction of
query length. Candidates are accepted by identity, uniqueness and alphabet only.
The output A3M is already aligned because this sampler makes substitutions only.
Synthetic variants are not evidence of evolutionary homology or functionality.

Gumbel-argmax uses upstream's fixed noise scale; temperature only affects vanilla
categorical sampling. The source CLI declares conditioning flags but does not wire
them through; this wrapper calls generate(partial_masks=...) directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

DPLM_COMMIT = "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
app = modal.App("dplm-homologs")
cache = modal.Volume.from_name("dplm-homologs-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("dplm-homologs-results", create_if_missing=True)


def load_inference_class():
    """Load unchanged upstream inference modules, skipping training-only imports.

Upstream package initializers eagerly import all training/structure dependencies.
Namespace shells provide only the registry and YAML utility used by inference.
The model, categorical samplers, and diffusion loop remain upstream code.
Python 3.10 accommodates upstream dataclass defaults.
"""
    import importlib
    import sys
    import types
    from omegaconf import OmegaConf

    root = Path("/opt/dplm/src/byprot")
    for name, relative in [("byprot", ""), ("byprot.models", "models"),
                           ("byprot.models.dplm", "models/dplm"),
                           ("byprot.models.dplm.modules", "models/dplm/modules"),
                           ("byprot.utils", "utils")]:
        module = types.ModuleType(name)
        module.__path__ = [str(root / relative)]
        sys.modules[name] = module
    registry = sys.modules["byprot.models"]
    registry.MODEL_REGISTRY = {}

    def register_model(name):
        def register(cls):
            registry.MODEL_REGISTRY[name] = cls
            return cls
        return register

    registry.register_model = register_model
    sys.modules["byprot.utils"].load_yaml_config = OmegaConf.load
    importlib.import_module("byprot.models.dplm.modules.dplm_modeling_esm")
    return importlib.import_module("byprot.models.dplm.dplm").DiffusionProteinLanguageModel


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==2.2.2", "numpy<2", "transformers==4.39.2",
                 "omegaconf==2.3.0", "huggingface_hub<1", "tqdm", "pyyaml")
    .run_commands("git clone https://github.com/bytedance/dplm /opt/dplm",
                  f"cd /opt/dplm && git checkout {DPLM_COMMIT}")
    .run_function(load_inference_class)
    .env({"HF_HOME": "/model_cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
)


def parse_positions(spec: str, length: int) -> set[int]:
    """Parse comma-separated 1-based positions/ranges into zero-based indices."""
    positions = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        ends = part.split("-")
        if len(ends) not in (1, 2):
            raise ValueError(f"Invalid position range: {part}")
        lo, hi = int(ends[0]), int(ends[-1])
        if not 1 <= lo <= hi <= length:
            raise ValueError(f"Position range {part} is outside 1..{length}")
        positions.update(range(lo - 1, hi))
    return positions


def read_query(fasta: str, sequence: str) -> str:
    if bool(fasta) == bool(sequence):
        raise ValueError("Provide exactly one of --fasta or --sequence")
    if fasta:
        lines = Path(fasta).read_text().splitlines()
        if sum(line.startswith(">") for line in lines) != 1:
            raise ValueError("FASTA must contain exactly one record")
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    sequence = "".join(sequence.upper().split())
    if not sequence or set(sequence) - AA:
        raise ValueError("Query must contain only the 20 canonical amino acids")
    return sequence


def validate_settings(sequence: str, settings: dict) -> list[int]:
    if len(sequence) > 1022:
        raise ValueError("This wrapper limits queries to 1022 residues")
    if not 0 < settings["target_identity"] < 1:
        raise ValueError("target_identity must be between 0 and 1, exclusive")
    if not 0 <= settings["identity_tolerance"] < 1:
        raise ValueError("identity_tolerance must be in [0, 1)")
    for key in ("num_variants", "batch_size", "max_candidates", "max_iter"):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if settings["max_candidates"] < settings["num_variants"]:
        raise ValueError("max_candidates must be at least num_variants")
    if not 0 < settings["mask_fraction"] <= 1:
        raise ValueError("mask_fraction must be in (0, 1]")
    if settings["sampling_strategy"] not in ("gumbel_argmax", "vanilla", "argmax"):
        raise ValueError("sampling_strategy must be gumbel_argmax, vanilla, or argmax")
    if settings["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if settings["sampling_strategy"] != "vanilla" and settings["temperature"] != 1.0:
        raise ValueError("temperature is only supported by vanilla sampling")
    fixed = parse_positions(settings["fixed_positions"], len(sequence))
    mutable = (parse_positions(settings["mutable_positions"], len(sequence))
               if settings["mutable_positions"] else set(range(len(sequence))))
    mutable -= fixed
    target = round((1 - settings["target_identity"]) * len(sequence))
    if target < 1 or target > len(mutable):
        raise ValueError("Rounded target mutation count is infeasible for mutable positions")
    if not settings["adaptive_mask"] and round(len(mutable) * settings["mask_fraction"]) < target:
        raise ValueError("Mask too small to reach target; increase mask_fraction")
    return sorted(mutable)


def candidate_status(sequence: str, query: str, target: int, tolerance: float,
                     fixed: set[int], seen: set[str]) -> tuple[str, int | None]:
    if len(sequence) != len(query) or set(sequence) - AA:
        return "invalid_sequence", None
    if any(sequence[i] != query[i] for i in fixed):
        raise RuntimeError("DPLM changed a protected position")
    mutations = sum(a != b for a, b in zip(sequence, query))
    if abs(mutations - target) > tolerance * len(query) + 1e-9:
        return "identity_rejected", mutations
    if sequence in seen or sequence == query:
        return "duplicate", mutations
    return "accepted", mutations


@app.function(image=image, gpu=os.environ.get("GPU", "A100-80GB"),
              cpu=4, memory=32768, timeout=7200,
              volumes={"/model_cache": cache, "/results": artifacts})
def generate_homologs(sequence: str, settings: dict, run_id: str) -> dict:
    import datetime
    import random
    import time
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    mutable = validate_settings(sequence, settings)
    protected = set(range(len(sequence))) - set(mutable)
    target = round((1 - settings["target_identity"]) * len(sequence))
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    start = time.monotonic()
    checkpoint = snapshot_download(settings["model_name"], revision=settings["model_revision"],
                                   ignore_patterns=["*.bin", "*.h5", "*.msgpack"])
    # Some upstream releases have only pytorch_model.bin.
    if not list(Path(checkpoint).glob("*.safetensors")):
        checkpoint = snapshot_download(settings["model_name"], revision=settings["model_revision"])
    cache.commit()
    cls = load_inference_class()
    model = cls.from_pretrained(checkpoint).eval().cuda()
    tokenizer = model.tokenizer
    query_tokens = tokenizer(sequence, return_tensors="pt")["input_ids"].cuda()
    if query_tokens.shape[1] != len(sequence) + 2:
        raise RuntimeError("Expected one token per residue plus BOS/EOS")
    decoded_query = "".join(tokenizer.decode(query_tokens[0], skip_special_tokens=True).split())
    assert decoded_query == sequence
    root = Path("/results") / run_id
    if (root / "progress.json").exists():
        raise FileExistsError("Remote run_id already exists; choose a new run_id or recover it")
    root.mkdir(parents=True, exist_ok=True)
    rows, accepted, seen = [], [], set()
    nmask = max(target, min(len(mutable), round(len(mutable) * settings["mask_fraction"])))
    batch_no = 0
    metadata = dict(settings=settings, query=sequence, query_length=len(sequence),
                    source_commit=DPLM_COMMIT, model_revision=Path(checkpoint).name,
                    target_mutations=target, effective_target_identity=1-target/len(sequence),
                    gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    run_id=run_id)
    while len(accepted) < settings["num_variants"] and len(rows) < settings["max_candidates"]:
        bsize = min(settings["batch_size"], settings["max_candidates"] - len(rows))
        inputs = query_tokens.repeat(bsize, 1)
        partial = torch.ones_like(inputs, dtype=torch.bool)
        masks = []
        for j in range(bsize):
            positions = sorted(random.sample(mutable, nmask))
            masks.append(positions)
            partial[j, torch.tensor(positions, device="cuda") + 1] = False
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(input_tokens=inputs, tokenizer=tokenizer,
                                    max_iter=settings["max_iter"],
                                    temperature=settings["temperature"],
                                    sampling_strategy=settings["sampling_strategy"],
                                    partial_masks=partial, disable_resample=True)
        if not torch.equal(output[partial], inputs[partial]):
            raise RuntimeError("Sampler violated fixed query conditioning")
        seqs = ["".join(s.split()) for s in tokenizer.batch_decode(output, skip_special_tokens=True)]
        counts = []
        for seq, positions in zip(seqs, masks):
            status, mutations = candidate_status(seq, sequence, target,
                settings["identity_tolerance"], protected, seen)
            row = dict(candidate=len(rows)+1, batch=batch_no, sequence=seq,
                       masked_positions_1based=[i+1 for i in positions],
                       mutations=mutations, identity=None if mutations is None else 1-mutations/len(sequence),
                       status=status)
            if mutations is not None:
                counts.append(mutations)
            if status == "accepted":
                seen.add(seq)
                if len(accepted) < settings["num_variants"]:
                    accepted.append(row)
                else:
                    row["status"] = "accepted_surplus"
            rows.append(row)
        metadata.update(accepted=len(accepted), candidates=len(rows),
                        elapsed_seconds=time.monotonic()-start,
                        complete=len(accepted) == settings["num_variants"])
        (root/"progress.json").write_text(json.dumps(metadata, indent=2))
        (root/"candidates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (root/"accepted.json").write_text(json.dumps(accepted, indent=2))
        artifacts.commit()
        print(f"Batch {batch_no}: mask={nmask}, mutations={counts}, accepted={len(accepted)}/{settings['num_variants']}", flush=True)
        if settings["adaptive_mask"] and counts:
            # Adjust next batch's mask size, not any decoded sequence.
            proposed = round(nmask * target / max(float(np.mean(counts)), 1.0))
            nmask = max(target, min(len(mutable), round((nmask + proposed) / 2)))
        batch_no += 1
    metadata["elapsed_seconds"] = time.monotonic()-start
    (root/"progress.json").write_text(json.dumps(metadata, indent=2))
    artifacts.commit()
    # Keep local consumers independent of the GPU's torch/numpy installation.
    return json.loads(json.dumps(dict(metadata=metadata, accepted=accepted, candidates=rows)))


@app.local_entrypoint()
def main(fasta: str = "", sequence: str = "", out_dir: str = "out/dplm_homologs",
         num_variants: int = 50, target_identity: float = 0.85,
         identity_tolerance: float = 0.0, fixed_positions: str = "",
         mutable_positions: str = "", mask_fraction: float = 0.4,
         adaptive_mask: bool = True, model_name: str = "airkingbd/dplm_650m",
         model_revision: str = "main", max_iter: int = 100,
         sampling_strategy: str = "gumbel_argmax", temperature: float = 1.0,
         seed: int = 42, batch_size: int = 16, max_candidates: int = 2048,
         run_id: str = "", recover_run_id: str = ""):
    """Generate distinct variants; save partial results and fail if quota is unmet."""
    import datetime
    if recover_run_id:
        if Path(recover_run_id).name != recover_run_id or recover_run_id in {".", ".."}:
            raise ValueError("recover_run_id must be a single directory name")
        def read_remote(name):
            return b"".join(artifacts.read_file(f"{recover_run_id}/{name}")).decode()
        result = dict(metadata=json.loads(read_remote("progress.json")),
                      accepted=json.loads(read_remote("accepted.json")),
                      candidates=[json.loads(line) for line in read_remote("candidates.jsonl").splitlines()])
        write_outputs(result, Path(out_dir))
        return
    settings = {k: v for k, v in locals().copy().items()
                if k not in {"fasta", "sequence", "out_dir", "run_id", "recover_run_id", "datetime"}}
    sequence = read_query(fasta, sequence)
    validate_settings(sequence, settings)
    if not run_id:
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("dplm_%Y%m%dT%H%M%S")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"metadata.json").exists():
        raise FileExistsError("Output metadata already exists; choose a new out_dir")
    (root/"request.json").write_text(json.dumps(dict(query=sequence, settings=settings, run_id=run_id), indent=2))
    call = generate_homologs.spawn(sequence, settings, run_id)
    (root/"modal_call_id.txt").write_text(call.object_id+"\n")
    result = call.get()
    write_outputs(result, root)


def write_outputs(result: dict, root: Path):
    """Export a remote result, including incomplete pools for inspection."""
    root.mkdir(parents=True, exist_ok=True)
    if (root/"metadata.json").exists():
        raise FileExistsError("Output metadata already exists; choose a new out_dir")
    sequence = result["metadata"]["query"]
    variants = "".join(f">variant_{i:03d} identity={r['identity']:.6f} mutations={r['mutations']} candidate={r['candidate']}\n{r['sequence']}\n"
                       for i, r in enumerate(result["accepted"], 1))
    query = f">query\n{sequence}\n"
    (root/"query.fasta").write_text(query)
    (root/"variants.fasta").write_text(variants)
    (root/"synthetic.a3m").write_text(query+variants)
    (root/"metadata.json").write_text(json.dumps(result["metadata"], indent=2))
    (root/"candidates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in result["candidates"]))
    print(json.dumps(result["metadata"], indent=2))
    if not result["metadata"]["complete"]:
        raise RuntimeError("Candidate budget exhausted; partial outputs saved. Increase max_candidates or adjust mask/sampling controls.")
