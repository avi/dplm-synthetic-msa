# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Modal wrapper for OpenFold3 with optional precomputed MSA support.

Docs: https://openfold-3.readthedocs.io/en/latest/
Repo: https://github.com/aqlaboratory/openfold-3
Based on: https://github.com/hgbrian/foldism/blob/main/backends/openfold3.py

Usage examples
--------------
Predict from a single query JSON with the ColabFold MSA server:

    modal run modal_openfold3.py --query-json examples/query.json --msa-mode use-msa-server

Predict from a query JSON that already references local MSA/template files:

    modal run modal_openfold3.py --query-json examples/query.json --msa-mode disabled

Precompute missing chain MSAs locally, inject them into the query JSON, and run:

    modal run modal_openfold3.py --query-json examples/query.json --msa-mode precomputed

Batch multiple query JSON / FASTA inputs into one OpenFold3 run:

    modal run modal_openfold3.py --input-files a.json,b.json,c.fasta --msa-mode precomputed
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))  # minutes
OPENFOLD3_CHECKPOINT = "of3-p2-155k"
REPO_ROOT = Path(__file__).resolve().parents[1]
OF3_MAIN_MSA_FILENAME = "custom_database_hits.a3m"

JSON_SUFFIXES = {".json"}
FASTA_SUFFIXES = {".faa", ".fasta", ".fa"}
MSA_MODE_CHOICES = {"disabled", "use-msa-server", "precomputed"}
PROTEIN_MOLECULE_TYPES = {"PROTEIN", "protein"}
PATH_LIST_FIELDS = {"main_msa_file_paths", "paired_msa_file_paths"}
PATH_SCALAR_FIELDS = {"template_alignment_file_path", "sdf_file_path"}
DEFAULT_OUTPUT_GLOBS = "**/*"

app = App("openfold3")
model_volume = modal.Volume.from_name("openfold3-models", create_if_missing=True)


def _download_openfold3_models():
    """Download OpenFold3 checkpoint and CCD to volume."""
    cache_dir = Path("/models/openfold3")
    cache_dir.mkdir(parents=True, exist_ok=True)

    ckpt = cache_dir / f"{OPENFOLD3_CHECKPOINT}.pt"
    if not ckpt.exists():
        print(f"[openfold3] Downloading checkpoint {OPENFOLD3_CHECKPOINT}.pt...")
        subprocess.run([
            "aws",
            "s3",
            "cp",
            "--no-sign-request",
            f"s3://openfold3-data/openfold3-parameters/{OPENFOLD3_CHECKPOINT}.pt",
            str(ckpt),
        ], check=True)
        print(f"[openfold3] Checkpoint downloaded ({ckpt.stat().st_size:,} bytes)")

    (cache_dir / "ckpt_root").write_text(str(cache_dir))

    ccd = cache_dir / "components.bcif"
    if not ccd.exists():
        print("[openfold3] Downloading CCD components.bcif...")
        subprocess.run([
            "aws",
            "s3",
            "cp",
            "--no-sign-request",
            "s3://openfold3-data/components.bcif",
            str(ccd),
        ], check=True)
        print(f"[openfold3] CCD downloaded ({ccd.stat().st_size:,} bytes)")


image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "wget", "libxrender1", "libxext6")
    .env({
        "OPENFOLD_CACHE": "/models/openfold3",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0",
        "CUDA_HOME": "/usr/local/cuda",
    })
    # Biotite 1.7.1 breaks OF3 0.4.0's create_atom_names(list) call.
    .pip_install("openfold3==0.4.0", "biotite==1.6.0")
    .run_function(_download_openfold3_models, volumes={"/models": model_volume})
)


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.strip().upper().split())


def sequence_cache_key(sequence: str) -> str:
    normalized = normalize_sequence(sequence)
    digest = hashlib.sha1(normalized.encode()).hexdigest()[:16]
    return f"protein_len{len(normalized)}_{digest}"


def local_boltz_python() -> str:
    explicit = os.environ.get("LOCAL_BOLTZ_PYTHON") or os.environ.get("BOLTZ_PYTHON")
    if explicit:
        return explicit

    boltz_bin = os.environ.get("BOLTZ_BIN")
    resolved = shutil.which(boltz_bin) if boltz_bin else shutil.which("boltz")
    if resolved:
        sibling_python = Path(resolved).resolve().parent / "python"
        if sibling_python.exists():
            return str(sibling_python)
    return sys.executable


def _normalize_chain(chain: dict) -> dict:
    normalized = dict(chain)
    chain_ids = normalized.get("chain_ids")
    if isinstance(chain_ids, str):
        normalized["chain_ids"] = [chain_ids]
    elif isinstance(chain_ids, list):
        normalized["chain_ids"] = [str(item) for item in chain_ids]
    else:
        normalized["chain_ids"] = None

    for field in PATH_LIST_FIELDS:
        value = normalized.get(field)
        if value is None:
            normalized[field] = None
        elif isinstance(value, list):
            normalized[field] = [str(item) for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            normalized[field] = [value]
        else:
            normalized[field] = None

    for field in PATH_SCALAR_FIELDS:
        value = normalized.get(field)
        normalized[field] = str(value) if isinstance(value, str) and value.strip() else None

    normalized.setdefault("description", None)
    normalized.setdefault("non_canonical_residues", None)
    normalized.setdefault("smiles", None)
    normalized.setdefault("ccd_codes", None)
    normalized.setdefault("template_entry_chain_ids", None)
    return normalized


def _normalize_query(query: dict, query_name: str) -> dict:
    normalized = dict(query)
    chains = normalized.get("chains") or []
    normalized["query_name"] = normalized.get("query_name") or query_name
    normalized["chains"] = [
        _normalize_chain(chain)
        for chain in chains
        if isinstance(chain, dict)
    ]
    normalized.setdefault("use_msas", False)
    normalized.setdefault("use_paired_msas", True)
    normalized.setdefault("use_main_msas", True)
    normalized.setdefault("covalent_bonds", None)
    return normalized


def _normalize_query_payload(payload: dict, default_name: str = "query") -> dict:
    if "queries" not in payload:
        payload = {"queries": {default_name: payload}}

    normalized = {
        "seeds": payload.get("seeds") or [42],
        "queries": {},
    }
    for query_name, query in (payload.get("queries") or {}).items():
        if not isinstance(query, dict):
            continue
        normalized["queries"][query_name] = _normalize_query(query, str(query_name))
    return normalized


def _resolve_msa_mode(use_msa_server: bool, msa_mode: str | None) -> str:
    if msa_mode is None:
        return "use-msa-server" if use_msa_server else "disabled"
    resolved = msa_mode.strip().lower()
    if resolved not in MSA_MODE_CHOICES:
        raise ValueError(f"msa_mode must be one of {sorted(MSA_MODE_CHOICES)}")
    if use_msa_server and resolved != "use-msa-server":
        raise ValueError("use_msa_server=true conflicts with explicit msa_mode")
    return resolved


def _resolve_local_asset_path(raw_ref: str, source_dir: Path | None) -> str:
    candidate = Path(raw_ref).expanduser()
    if not candidate.is_absolute():
        if source_dir is None:
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = (source_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def _absolutize_query_asset_paths(payload: dict, source_dir: Path | None) -> dict:
    normalized = copy.deepcopy(payload)
    for query in normalized.get("queries", {}).values():
        if not isinstance(query, dict):
            continue
        for chain in query.get("chains") or []:
            if not isinstance(chain, dict):
                continue
            for field in PATH_LIST_FIELDS:
                refs = chain.get(field)
                if not isinstance(refs, list):
                    continue
                chain[field] = [
                    _resolve_local_asset_path(ref, source_dir)
                    for ref in refs
                    if isinstance(ref, str) and ref.strip()
                ] or None
            for field in PATH_SCALAR_FIELDS:
                ref = chain.get(field)
                if isinstance(ref, str) and ref.strip():
                    chain[field] = _resolve_local_asset_path(ref, source_dir)
    return normalized


def _merge_query_payloads(payloads: Iterable[dict]) -> dict:
    merged_queries: dict[str, dict] = {}
    merged_seeds: list[int] = []

    for payload in payloads:
        for seed in payload.get("seeds") or []:
            if seed not in merged_seeds:
                merged_seeds.append(seed)

        for query_name, query in payload.get("queries", {}).items():
            candidate = str(query_name)
            counter = 2
            while candidate in merged_queries:
                candidate = f"{query_name}_{counter}"
                counter += 1
            query_copy = copy.deepcopy(query)
            query_copy["query_name"] = candidate
            merged_queries[candidate] = _normalize_query(query_copy, candidate)

    return {
        "seeds": merged_seeds or [42],
        "queries": merged_queries,
    }


def _iter_query_chains(payload: dict):
    for query_name, query in payload.get("queries", {}).items():
        if not isinstance(query, dict):
            continue
        chains = query.get("chains") or []
        yield query_name, query, chains


def _collect_query_asset_specs(payload: dict) -> list[tuple[str, str, bytes]]:
    asset_specs: list[tuple[str, str, bytes]] = []
    seen: set[tuple[str, str]] = set()

    for _query_name, _query, chains in _iter_query_chains(payload):
        for chain in chains:
            if not isinstance(chain, dict):
                continue
            for field in PATH_LIST_FIELDS:
                refs = chain.get(field)
                if not isinstance(refs, list):
                    continue
                for ref in refs:
                    if not isinstance(ref, str) or not ref.strip():
                        continue
                    item = (field, ref)
                    if item in seen:
                        continue
                    path = Path(ref).expanduser()
                    if not path.exists():
                        raise FileNotFoundError(f"Referenced query asset does not exist: {path}")
                    asset_specs.append((field, ref, path.read_bytes()))
                    seen.add(item)
            for field in PATH_SCALAR_FIELDS:
                ref = chain.get(field)
                if not isinstance(ref, str) or not ref.strip():
                    continue
                item = (field, ref)
                if item in seen:
                    continue
                path = Path(ref).expanduser()
                if not path.exists():
                    raise FileNotFoundError(f"Referenced query asset does not exist: {path}")
                asset_specs.append((field, ref, path.read_bytes()))
                seen.add(item)

    return asset_specs


def _has_query_asset_refs(payload: dict) -> bool:
    for _query_name, _query, chains in _iter_query_chains(payload):
        for chain in chains:
            if not isinstance(chain, dict):
                continue
            for field in PATH_LIST_FIELDS:
                refs = chain.get(field)
                if isinstance(refs, list) and any(isinstance(ref, str) and ref.strip() for ref in refs):
                    return True
            for field in PATH_SCALAR_FIELDS:
                ref = chain.get(field)
                if isinstance(ref, str) and ref.strip():
                    return True
    return False


def _rewrite_query_asset_paths(
    query_json_str: str,
    *,
    asset_specs: list[tuple[str, str, bytes]],
    asset_root: Path,
) -> str:
    payload = json.loads(query_json_str)
    target_map: dict[tuple[str, str], str] = {}
    used_targets: set[Path] = set()

    def reserve_target(field: str, original_ref: str) -> Path:
        bucket = {
            "main_msa_file_paths": "main_msa",
            "paired_msa_file_paths": "paired_msa",
            "template_alignment_file_path": "templates",
            "sdf_file_path": "ligands",
        }.get(field, "assets")
        source_path = Path(original_ref)
        if field in PATH_LIST_FIELDS:
            chain_dir = source_path.parent.name or "chain"
            target = asset_root / bucket / chain_dir / source_path.name
            counter = 1
            while target in used_targets:
                target = asset_root / bucket / f"{chain_dir}_{counter}" / source_path.name
                counter += 1
        else:
            target = asset_root / bucket / source_path.name
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while target in used_targets:
                target = target.with_name(f"{stem}_{counter}{suffix}")
                counter += 1
        used_targets.add(target)
        return target

    for field, original_ref, content in asset_specs:
        target = reserve_target(field, original_ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target_map[(field, original_ref)] = str(target.resolve())

    for _query_name, _query, chains in _iter_query_chains(payload):
        for chain in chains:
            if not isinstance(chain, dict):
                continue
            for field in PATH_LIST_FIELDS:
                refs = chain.get(field)
                if not isinstance(refs, list):
                    continue
                chain[field] = [
                    target_map.get((field, ref), ref)
                    for ref in refs
                    if isinstance(ref, str) and ref.strip()
                ] or None
            for field in PATH_SCALAR_FIELDS:
                ref = chain.get(field)
                if isinstance(ref, str) and ref.strip():
                    chain[field] = target_map.get((field, ref), ref)

    return json.dumps(payload, indent=2)


def _chain_requires_precomputed_main_msa(chain: dict) -> bool:
    if chain.get("molecule_type") not in PROTEIN_MOLECULE_TYPES:
        return False
    refs = chain.get("main_msa_file_paths")
    if not isinstance(refs, list) or not refs:
        return True
    for ref in refs:
        if isinstance(ref, str) and ref.strip() and Path(ref).expanduser().exists():
            return False
    return True


def _of3_main_msa_cache_path(msa_cache_dir: Path, sequence: str) -> Path:
    return msa_cache_dir / sequence_cache_key(sequence) / OF3_MAIN_MSA_FILENAME


def _migrate_legacy_of3_main_msa(msa_cache_dir: Path, sequence: str) -> Path:
    new_path = _of3_main_msa_cache_path(msa_cache_dir, sequence)
    legacy_path = msa_cache_dir / f"{sequence_cache_key(sequence)}.a3m"
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(legacy_path.read_text())
        return new_path
    return new_path


def _precompute_missing_main_msas(payload: dict, msa_cache_dir: Path) -> dict:
    msa_cache_dir.mkdir(parents=True, exist_ok=True)
    requested: dict[str, str] = {}
    affected_chains: list[tuple[dict, Path]] = []

    for _query_name, query, chains in _iter_query_chains(payload):
        for chain in chains:
            if not isinstance(chain, dict):
                continue
            if not _chain_requires_precomputed_main_msa(chain):
                continue
            sequence = normalize_sequence(str(chain.get("sequence") or ""))
            if not sequence:
                continue
            cache_path = _migrate_legacy_of3_main_msa(msa_cache_dir, sequence)
            affected_chains.append((chain, cache_path))
            requested.setdefault(cache_path.stem, sequence)
            query["use_main_msas"] = True
            query.setdefault("use_msas", False)
            query.setdefault("use_paired_msas", True)

    missing = [sequence for sequence in requested.values() if not _of3_main_msa_cache_path(msa_cache_dir, sequence).exists()]
    stdout = ""
    if missing:
        cmd = [
            local_boltz_python(),
            "scripts/precompute_openfold3_msa.py",
            "--out-dir",
            str(msa_cache_dir),
        ]
        for sequence in sorted(missing):
            cmd.extend(["--sequence", sequence])
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = "\n".join(
                part.strip()
                for part in [exc.stdout or "", exc.stderr or ""]
                if part and part.strip()
            )
            raise RuntimeError(
                "OpenFold3 MSA precompute failed. Set LOCAL_BOLTZ_PYTHON/BOLTZ_PYTHON "
                "to a Boltz env and check network access to the ColabFold MSA server."
                + (f"\n{details[-4000:]}" if details else "")
            ) from exc
        stdout = completed.stdout.strip()
        if stdout:
            print(stdout, flush=True)

    for chain, cache_path in affected_chains:
        if not cache_path.exists():
            raise FileNotFoundError(f"Expected precomputed OF3 MSA not found: {cache_path}")
        chain["main_msa_file_paths"] = [str(cache_path)]

    summary = {
        "msa_mode": "precomputed",
        "requested_sequences": len(requested),
        "submitted_sequences": len(missing),
        "msa_cache_dir": str(msa_cache_dir),
        "runner_script": "scripts/precompute_openfold3_msa.py",
    }
    if stdout:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                summary["precompute_result"] = parsed
        except json.JSONDecodeError:
            summary["precompute_stdout"] = stdout
    return summary


def _load_query_payload_from_path(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in JSON_SUFFIXES:
        payload = json.loads(path.read_text())
        payload = _normalize_query_payload(payload, default_name=path.stem)
        return _absolutize_query_asset_paths(payload, path.resolve().parent)
    if suffix in FASTA_SUFFIXES:
        fasta_payload = _fasta_to_query_json(path.read_text(), name=path.stem)
        payload = _normalize_query_payload(fasta_payload, default_name=path.stem)
        return _absolutize_query_asset_paths(payload, None)
    raise ValueError(f"Unsupported input file type: {path}")


def _load_input_payloads(
    *,
    query_json: str | None,
    input_faa: str | None,
    input_files: str | None,
) -> list[dict]:
    payloads: list[dict] = []

    if input_files:
        for raw_item in input_files.split(","):
            item = raw_item.strip()
            if not item:
                continue
            payloads.append(_load_query_payload_from_path(Path(item).expanduser().resolve()))

    if query_json:
        payloads.append(_load_query_payload_from_path(Path(query_json).expanduser().resolve()))

    if input_faa:
        path = Path(input_faa).expanduser().resolve()
        payloads.append(_load_query_payload_from_path(path))

    if not payloads:
        raise ValueError("Provide query_json, input_faa, or input_files")

    return payloads


def _parse_output_globs(output_globs: str | None) -> list[str]:
    patterns = [item.strip() for item in (output_globs or DEFAULT_OUTPUT_GLOBS).split(",")]
    return [pattern for pattern in patterns if pattern]


def _matches_output_globs(relative_path: Path, output_globs: str | None) -> bool:
    rel = relative_path.as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in _parse_output_globs(output_globs))


def _fasta_to_query_json(fasta_str: str, name: str = "query") -> dict:
    """Convert a Chai-style FASTA string to an OpenFold3 query JSON."""
    chains = []
    chain_id_counter = ord("A")

    ion_ccd = {
        "MG": "MG",
        "ZN": "ZN",
        "FE": "FE",
        "CA": "CA",
        "MN": "MN",
        "CO": "CO",
        "NI": "NI",
        "CU": "CU",
        "NA": "NA",
        "K": "K",
        "CL": "CL",
    }
    ligand_ccd = {
        "ATP": "ATP",
        "ADP": "ADP",
        "GTP": "GTP",
        "GDP": "GDP",
        "NAD": "NAD",
        "FAD": "FAD",
        "HEM": "HEM",
    }

    entries = []
    current_header = None
    current_seq: list[str] = []

    for line in fasta_str.strip().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_header is not None:
                entries.append((current_header, "".join(current_seq)))
            current_header = line[1:]
            current_seq = []
        elif line:
            current_seq.append(line)
    if current_header is not None:
        entries.append((current_header, "".join(current_seq)))

    for header, seq in entries:
        parts = header.split("|")
        entity_type = parts[0].strip().lower()
        entity_name = ""
        for part in parts[1:]:
            if part.strip().startswith("name="):
                entity_name = part.strip()[5:]

        chain_id = chr(chain_id_counter)
        chain_id_counter += 1

        if entity_type == "protein":
            chains.append({
                "molecule_type": "PROTEIN",
                "chain_ids": [chain_id],
                "description": None,
                "sequence": seq,
                "non_canonical_residues": None,
                "smiles": None,
                "ccd_codes": None,
                "paired_msa_file_paths": None,
                "main_msa_file_paths": None,
                "template_alignment_file_path": None,
                "template_entry_chain_ids": None,
                "sdf_file_path": None,
            })
        elif entity_type == "ligand":
            upper_name = entity_name.upper()
            upper_seq = seq.strip().upper()
            if upper_name in ligand_ccd:
                chains.append({
                    "molecule_type": "LIGAND",
                    "chain_ids": [chain_id],
                    "ccd_codes": ligand_ccd[upper_name],
                })
            elif upper_name in ion_ccd:
                chains.append({
                    "molecule_type": "LIGAND",
                    "chain_ids": [chain_id],
                    "ccd_codes": ion_ccd[upper_name],
                })
            elif upper_seq in ion_ccd:
                chains.append({
                    "molecule_type": "LIGAND",
                    "chain_ids": [chain_id],
                    "ccd_codes": ion_ccd[upper_seq],
                })
            else:
                chains.append({
                    "molecule_type": "LIGAND",
                    "chain_ids": [chain_id],
                    "smiles": seq,
                })
        elif entity_type == "ion":
            code = entity_name.upper() or seq.strip().upper()
            chains.append({
                "molecule_type": "LIGAND",
                "chain_ids": [chain_id],
                "ccd_codes": ion_ccd.get(code, code),
            })

    return {
        "seeds": [42],
        "queries": {
            name: {
                "query_name": name,
                "chains": chains,
                "use_msas": False,
                "use_paired_msas": True,
                "use_main_msas": True,
                "covalent_bonds": None,
            }
        },
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={"/models": model_volume},
)
def run_openfold3(
    query_json_str: str,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = False,
    asset_specs: list[tuple[str, str, bytes]] | None = None,
    output_globs: str | None = None,
) -> list[tuple[str, bytes]]:
    """Run OpenFold3 prediction."""
    import yaml

    model_volume.reload()
    ckpt_path = Path(f"/models/openfold3/{OPENFOLD3_CHECKPOINT}.pt")
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    with TemporaryDirectory() as td:
        workdir = Path(td)
        outdir = workdir / "output"
        outdir.mkdir()

        if asset_specs:
            query_json_str = _rewrite_query_asset_paths(
                query_json_str,
                asset_specs=asset_specs,
                asset_root=workdir / "input_assets",
            )

        query_path = workdir / "query.json"
        query_path.write_text(query_json_str)

        runner_config = {
            "model_update": {
                "presets": ["predict", "pae_enabled"],
                "custom": {
                    "settings": {
                        "memory": {
                            "eval": {
                                "use_deepspeed_evo_attention": False,
                            }
                        }
                    }
                },
            }
            ,
            "dataset_config_kwargs": {
                "msa": {
                    "max_seq_counts": {
                        "custom_database_hits": 50000,
                    },
                    "msas_to_pair": [],
                    "aln_order": [
                        "custom_database_hits",
                    ],
                }
            },
        }
        runner_yaml = workdir / "runner.yaml"
        runner_yaml.write_text(yaml.dump(runner_config))

        cmd = (
            f"stdbuf -oL run_openfold predict"
            f" --query-json {query_path}"
            f" --output-dir {outdir}"
            f" --inference-ckpt-path {ckpt_path}"
            f" --num-diffusion-samples {num_diffusion_samples}"
            f" --num-model-seeds {num_model_seeds}"
            f" --use-msa-server {str(use_msa_server).lower()}"
            f" --runner-yaml {runner_yaml}"
        )

        print(f"Running: {cmd}")
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.rstrip(), flush=True)

        process.wait()
        if process.returncode != 0:
            for log_file in Path(outdir).rglob("*.log"):
                print(f"\n=== {log_file} ===")
                print(log_file.read_text()[-2000:])
            raise subprocess.CalledProcessError(process.returncode, cmd)

        results = []
        for path in outdir.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(outdir)
            if not _matches_output_globs(relative_path, output_globs):
                continue
            results.append((str(relative_path), path.read_bytes()))

        return results


@app.local_entrypoint()
def main(
    query_json: str | None = None,
    input_faa: str | None = None,
    input_files: str | None = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = False,
    msa_mode: str | None = None,
    msa_cache_dir: str | None = None,
    output_globs: str | None = None,
    out_dir: str = "./out/openfold3",
    run_name: str | None = None,
):
    """Run OpenFold3 predictions via Modal."""
    resolved_msa_mode = _resolve_msa_mode(use_msa_server, msa_mode)
    payloads = _load_input_payloads(
        query_json=query_json,
        input_faa=input_faa,
        input_files=input_files,
    )
    merged_payload = _merge_query_payloads(payloads)

    outdir = Path(out_dir) / (run_name or datetime.now().strftime("%Y%m%d%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)

    precompute_summary = None
    if resolved_msa_mode == "precomputed":
        cache_dir = Path(msa_cache_dir or (Path(out_dir) / "precomputed_msa")).expanduser().resolve()
        precompute_summary = _precompute_missing_main_msas(merged_payload, cache_dir)
        (outdir / "msa_precompute.json").write_text(json.dumps(precompute_summary, indent=2))

    if resolved_msa_mode == "use-msa-server":
        asset_specs: list[tuple[str, str, bytes]] = []
        if _has_query_asset_refs(merged_payload):
            print(
                "[openfold3] MSA server mode enabled; explicit local MSA/template paths will be ignored by OF3.",
                flush=True,
            )
    else:
        asset_specs = _collect_query_asset_specs(merged_payload)

    query_json_str = json.dumps(merged_payload, indent=2)
    (outdir / "input_query.json").write_text(query_json_str)

    outputs = run_openfold3.remote(
        query_json_str=query_json_str,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=(resolved_msa_mode == "use-msa-server"),
        asset_specs=asset_specs,
        output_globs=output_globs,
    )

    for fname, data in outputs:
        dest = outdir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    print(f"Saved {len(outputs)} files to {outdir}")
