"""Create matched no-MSA, query-only, synthetic and optional natural-MSA inputs."""
import argparse,json,shutil
from pathlib import Path
AA=set('ACDEFGHIKLMNPQRSTVWY')
def fasta(path):
 text=Path(path).read_text();return [''.join(b.splitlines()[1:]) for b in text.split('>') if b.strip()]
def payload(name,query,msa):
 return dict(query_name=name,use_msas=msa is not None,use_main_msas=msa is not None,use_paired_msas=False,chains=[dict(chain_ids=['A'],molecule_type='PROTEIN',sequence=query,main_msa_file_paths=[str(msa)] if msa else [],paired_msa_file_paths=[],template_alignment_file_path=None)])
def validate_msa(path,query):
 rows=fasta(path)
 aligned=[''.join(c for c in s if not c.islower() and c!='.') for s in rows]
 if not aligned or aligned[0]!=query:raise ValueError('The first MSA row must match the query')
 if any(len(s)!=len(query) or not set(s)<=AA|{'-','X'} for s in aligned):raise ValueError('Invalid aligned MSA row')
 return len(rows)
def prepare(fasta_path,synthetic,out,natural=None,name='query'):
 seqs=fasta(fasta_path)
 if len(seqs)!=1 or not seqs[0] or not set(seqs[0])<=AA:raise ValueError('Supply one canonical protein query')
 query=seqs[0];validate_msa(synthetic,query)
 if natural:validate_msa(natural,query)
 out=Path(out).absolute();out.mkdir(parents=True,exist_ok=True)
 if (out/'queries.json').exists():raise FileExistsError('Choose a new output directory')
 queries={}
 for arm,source in [('no_msa',None),('query_only',None),('synthetic',synthetic)]+([('natural',natural)] if natural else []):
  msa=None
  if arm!='no_msa':
   msa=out/arm/'custom_database_hits.a3m';msa.parent.mkdir(exist_ok=True)
   if arm=='query_only':msa.write_text(f'>query\n{query}\n')
   else:shutil.copy2(source,msa)
  key=f'{name}__{arm}';queries[key]=payload(key,query,msa)
 result=out/'queries.json';result.write_text(json.dumps(dict(seeds=[42],queries=queries),indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fasta',type=Path,required=True);p.add_argument('--synthetic-msa',type=Path,required=True);p.add_argument('--natural-msa',type=Path);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--name',default='query');a=p.parse_args();print(prepare(a.fasta,a.synthetic_msa,a.out_dir,a.natural_msa,a.name))
