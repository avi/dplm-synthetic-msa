#!/usr/bin/env python3
"""Precompute OpenFold3-compatible A3M MSAs for one or more protein sequences.

This uses the same MMSeqs2/ColabFold client Boltz uses, but writes the raw A3M
files expected by OpenFold3's `main_msa_file_paths`.

Examples:

    python scripts/precompute_openfold3_msa.py \
      --out-dir proteins/conformational/out/of3_precomputed_msa \
      --sequence MSEQUENCEONE \
      --sequence MSEQUENCETWO

    python scripts/precompute_openfold3_msa.py \
      --out-dir proteins/conformational/out/of3_precomputed_msa \
      --sequences-file seqs.fasta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

OF3_MAIN_MSA_FILENAME = "custom_database_hits.a3m"


def sequence_cache_key(sequence: str) -> str:
    normalized = normalize_sequence(sequence)
    digest = hashlib.sha1(normalized.encode()).hexdigest()[:16]
    return f"protein_len{len(normalized)}_{digest}"


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.strip().upper().split())


def load_sequences_from_file(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON sequence input must be a list of strings.")
        return [normalize_sequence(str(item)) for item in data if str(item).strip()]

    sequences: list[str] = []
    current: list[str] = []
    saw_fasta_header = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            saw_fasta_header = True
            if current:
                sequences.append(normalize_sequence("".join(current)))
                current = []
            continue
        if saw_fasta_header:
            current.append(line)
        else:
            sequences.append(normalize_sequence(line))
    if current:
        sequences.append(normalize_sequence("".join(current)))
    return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute OpenFold3 A3M cache entries for one or more protein sequences.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where sequence-keyed .a3m cache files will be written.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Protein sequence to precompute. Repeat for multiple sequences.",
    )
    parser.add_argument(
        "--sequences-file",
        action="append",
        default=[],
        help="Optional text/FASTA/JSON file containing sequences to precompute.",
    )
    parser.add_argument(
        "--msa-server-url",
        default="https://api.colabfold.com",
        help="MMSeqs2/ColabFold server URL.",
    )
    parser.add_argument(
        "--msa-server-username",
        default=None,
        help="Basic-auth username for the MSA server. Can also come from BOLTZ_MSA_USERNAME.",
    )
    parser.add_argument(
        "--msa-server-password",
        default=None,
        help="Basic-auth password for the MSA server. Can also come from BOLTZ_MSA_PASSWORD.",
    )
    parser.add_argument(
        "--api-key-header",
        default=None,
        help="Optional header name for API key auth. Defaults to X-API-Key if only value is provided.",
    )
    parser.add_argument(
        "--api-key-value",
        default=None,
        help="Optional API key value. Can also come from MSA_API_KEY_VALUE.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a sequence-keyed cache A3M already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from boltz.data.msa.mmseqs2 import run_mmseqs2
    except ImportError as exc:
        raise RuntimeError(
            "This script must be run with a Python environment that has the `boltz` package installed."
        ) from exc

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences: list[str] = [normalize_sequence(seq) for seq in args.sequence if seq.strip()]
    for raw_path in args.sequences_file:
        sequences.extend(load_sequences_from_file(Path(raw_path).expanduser()))
    sequences = [seq for seq in sequences if seq]
    if not sequences:
        raise RuntimeError("No sequences provided. Use --sequence and/or --sequences-file.")

    unique_sequences: list[str] = []
    for seq in sequences:
        if seq not in unique_sequences:
            unique_sequences.append(seq)

    to_fetch: list[str] = []
    skipped = 0
    for seq in unique_sequences:
        cache_path = out_dir / sequence_cache_key(seq) / OF3_MAIN_MSA_FILENAME
        legacy_path = out_dir / f"{sequence_cache_key(seq)}.a3m"
        if not cache_path.exists() and legacy_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(legacy_path.read_text())
        if cache_path.exists() and not args.force:
            skipped += 1
            continue
        to_fetch.append(seq)

    msa_server_username = args.msa_server_username or os.environ.get("BOLTZ_MSA_USERNAME")
    msa_server_password = args.msa_server_password or os.environ.get("BOLTZ_MSA_PASSWORD")
    api_key_value = args.api_key_value or os.environ.get("MSA_API_KEY_VALUE")
    auth_headers = None
    if api_key_value:
        auth_headers = {args.api_key_header or "X-API-Key": api_key_value}

    if not to_fetch:
        print(
            json.dumps(
                {
                    "requested_sequences": len(unique_sequences),
                    "fetched_sequences": 0,
                    "skipped_existing": skipped,
                    "out_dir": str(out_dir),
                },
                indent=2,
            )
        )
        return 0

    with tempfile.TemporaryDirectory(prefix="of3_msa_precompute_") as tmp_dir:
        a3m_results = run_mmseqs2(
            to_fetch,
            prefix=str(Path(tmp_dir) / "unpaired"),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=args.msa_server_url,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )

    written_entries: list[dict[str, str | int]] = []
    for sequence, a3m in zip(to_fetch, a3m_results, strict=True):
        if not a3m.endswith("\n"):
            a3m = f"{a3m}\n"
        cache_path = out_dir / sequence_cache_key(sequence) / OF3_MAIN_MSA_FILENAME
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(a3m)
        rows_written = sum(1 for line in a3m.splitlines() if line.startswith(">"))
        written_entries.append(
            {
                "sequence_key": cache_path.stem,
                "sequence_length": len(sequence),
                "cached_msa": str(cache_path),
                "rows_written": rows_written,
            }
        )

    print(
        json.dumps(
            {
                "requested_sequences": len(unique_sequences),
                "fetched_sequences": len(to_fetch),
                "skipped_existing": skipped,
                "out_dir": str(out_dir),
                "entries": written_entries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
