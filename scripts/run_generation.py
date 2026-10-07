"""Print a reproducible Modal generation command; --execute submits the GPU job."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALLOWED=set('fasta sequence out_dir num_variants target_identity identity_tolerance filter_identity fixed_positions mutable_positions mask_fraction keep_fractions adaptive_mask model_name model_revision max_iter sampling_strategy temperature seed batch_size max_candidates run_id recover_run_id'.split())
BOOL={'filter_identity','adaptive_mask'}
def command(config):
 unknown=set(config)-ALLOWED
 if unknown:raise ValueError(f'Unknown settings: {sorted(unknown)}')
 cmd=[sys.executable,'-m','modal','run','--detach',str(ROOT/'scripts/modal_dplm_homologs.py')]
 for k,v in config.items():
  flag=k.replace('_','-')
  if k in BOOL:
   if not isinstance(v,bool):raise ValueError(f'{k} must be a JSON boolean')
   cmd.append('--'+('' if v else 'no-')+flag)
  else:cmd.extend(['--'+flag,str(v)])
 return cmd
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 cfg=json.loads(a.config.read_text());cmd=command(cfg);print(shlex.join(cmd),flush=True)
 if a.execute:subprocess.run(cmd,cwd=ROOT,check=True)
