"""Verify checksums, packaged input contracts, and recomputed structure metrics."""
from pathlib import Path
import argparse,csv,json,hashlib,sys
sys.dont_write_bytecode=True
from analysis_metrics import get_ca,fit,ca_lddt
from tmtools import tm_align
ROOT=Path(__file__).resolve().parents[1]
def main(root):
 hashes=json.loads((root/'SHA256SUMS.json').read_text())
 for f,digest in hashes.items():
  assert hashlib.sha256((root/f).read_bytes()).hexdigest()==digest,f
 m=json.loads((root/'manifest.json').read_text());rows=list(csv.DictReader((root/'results/per_sample.csv').open()));assert len(rows)==m['counts']['structures']
 seen=set()
 for row in rows:
  key=(row['protein'],row['arm'],row['sample']);assert key not in seen;seen.add(key)
  p=m['proteins'][row['protein']];seq,xyz,_=get_ca(root/row['file']);rs,ref,_=get_ca(root/p['reference_file']);assert seq==p['query'] and len(rs)==len(seq)
  vals=dict(ca_rmsd=fit(xyz,ref)[1],tm_score=float(tm_align(xyz,ref,seq,rs).tm_norm_chain2),ca_lddt=ca_lddt(xyz,ref))
  for k,v in vals.items():assert abs(v-float(row[k]))<1e-5,(key,k)
 for key,c in m['conditions'].items():
  assert sum(r['protein']==c['protein'] and r['arm']==c['arm'] for r in rows)==5
  if c['msa']:assert (root/c['msa']).exists()
  d=root/'data'/c['protein']/'generation'/c['arm']
  if not d.exists():continue
  meta=json.loads((d/'metadata.json').read_text());s=meta['settings'];assert meta['complete']
  vs=[''.join(b.splitlines()[1:]) for b in (d/'variants.fasta').read_text().split('>') if b.strip()];assert len(vs)==len(set(vs))==s['num_variants']
  query=m['proteins'][c['protein']]['query'];assert all(len(v)==len(query) and v!=query for v in vs)
  if c['category']!='exploratory':assert s['fixed_positions']==s['mutable_positions']=='' and not s['filter_identity'] and not s['adaptive_mask'] and s['sampling_strategy']=='gumbel_argmax' and s['temperature']==1
 print(json.dumps(dict(files_verified=len(hashes),structures_recomputed=len(rows),conditions=len(m['conditions']),proteins=len(m['proteins']))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();main(a.root)
