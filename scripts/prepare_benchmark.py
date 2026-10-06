"""Resolve packaged MSAs into an OpenFold JSON; no job is submitted."""
import argparse,json
from pathlib import Path
from prepare_comparison import payload
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protein',action='append');p.add_argument('--include-exploratory',action='store_true');p.add_argument('--include-depth250',action='store_true');p.add_argument('--out',type=Path,default=Path('out/benchmark_queries.json'));a=p.parse_args();m=json.loads((ROOT/'manifest.json').read_text());queries={}
 if a.protein and set(a.protein)-set(m['proteins']):p.error('Unknown protein name')
 for name,c in m['conditions'].items():
  if a.protein and c['protein'] not in a.protein:continue
  if c['category']=='exploratory' and not a.include_exploratory:continue
  if c['category']=='depth_extension' and not a.include_depth250:continue
  seq=m['proteins'][c['protein']]['query'];msa=ROOT/c['msa'] if c['msa'] else None
  queries[name]=payload(name,seq,msa)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 if a.out.exists():raise FileExistsError('Choose a new output JSON')
 a.out.write_text(json.dumps(dict(seeds=[42],queries=queries),indent=2));print(f'{len(queries)} conditions: {a.out}')
