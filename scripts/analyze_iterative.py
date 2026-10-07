"""Summarize iterative lineages and optionally score folded query structures."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def summarize(protein, generation, predictions=None):
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    info = manifest['proteins'][protein]
    query = info['query']
    metadata = json.loads((generation / 'metadata.json').read_text())
    if metadata['query'] != query:
        raise ValueError('Generation query does not match benchmark protein')
    candidates = [json.loads(line) for line in (generation / 'candidates.jsonl').read_text().splitlines()]
    selected = [row for row in candidates if row['status'] == 'accepted']
    if not selected or not all('lineage' in row for row in selected):
        raise ValueError('Expected accepted iterative lineages')
    rounds = []
    for i in range(len(selected[0]['lineage'])):
        rows = [row['lineage'][i] for row in selected]
        rounds.append(dict(round=i+1, mean_mutations=mean(row['mutations'] for row in rows),
            mean_identity=mean(row['identity'] for row in rows),
            mean_parent_mutations=mean(row['parent_mutations'] for row in rows),
            mean_reversions=mean(row['reversions'] for row in rows),
            unique_sequences=len({row['sequence'] for row in rows})))
    controls = []
    for arm in ('synthetic_40', 'synthetic_60'):
        pool = ROOT / 'data' / protein / 'generation' / arm / 'synthetic.a3m'
        if not pool.exists():
            continue
        seqs = [''.join(block.splitlines()[1:]) for block in pool.read_text().split('>') if block.strip()]
        if seqs[0] != query:
            raise ValueError('Control query mismatch')
        counts = [sum(a != b for a,b in zip(query, seq)) for seq in seqs[1:]]
        controls.append(dict(arm=arm, variants=len(counts), mean_mutations=mean(counts),
                             mean_identity=1-mean(counts)/len(query)))
    result = dict(protein=protein, run_id=metadata['run_id'], complete=metadata['complete'],
                  selected_lineages=len(selected), rounds=rounds, generation_controls=controls,
                  caveat='Single generated pool; rounds have different divergence and use more compute than one-shot controls.')
    if predictions is not None:
        from analysis_metrics import get_ca, fit, ca_lddt
        from tmtools import tm_align
        rs, ref, _ = get_ca(ROOT / info['reference_file'])
        samples = []
        for path in sorted(predictions.rglob('*_model.cif')):
            seq, xyz, _ = get_ca(path)
            if seq != query or len(rs) != len(seq):
                raise ValueError(f'Structure/query mismatch: {path}')
            confidence = json.loads(path.with_name(path.name.replace('_model.cif', '_confidences_aggregated.json')).read_text())
            samples.append(dict(file=str(path), ca_rmsd=fit(xyz,ref)[1],
                tm_score=float(tm_align(xyz,ref,seq,rs).tm_norm_chain2), ca_lddt=ca_lddt(xyz,ref),
                avg_plddt=confidence['avg_plddt'], ptm=confidence['ptm']))
        if not samples:
            raise ValueError('No *_model.cif prediction files found')
        result['fold_samples'] = samples
        result['fold_mean'] = {key:mean(row[key] for row in samples)
                               for key in ('ca_rmsd','tm_score','ca_lddt','avg_plddt','ptm')}
        result['fold_controls'] = [row for row in csv.DictReader((ROOT/'results/summary.csv').open())
                                   if row['protein'] == protein and row['category'] == 'main']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protein', required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.protein, args.generation, args.predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
