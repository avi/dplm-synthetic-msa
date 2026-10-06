import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_comparison import prepare,validate_msa
from run_generation import command
class ReleaseInputs(unittest.TestCase):
 def test_matched_inputs_keep_query_and_disable_only_no_msa(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'q.fa').write_text('>q\nACDE\n');(p/'s.a3m').write_text('>q\nACDE\n>v\nACDF\n')
   out=prepare(p/'q.fa',p/'s.a3m',p/'out');qs=json.loads(out.read_text())['queries'];self.assertEqual(len(qs),3)
   for k,q in qs.items():
    self.assertEqual(q['chains'][0]['sequence'],'ACDE');self.assertEqual(q['use_msas'],not k.endswith('no_msa'))
    for path in q['chains'][0]['main_msa_file_paths']:self.assertEqual(Path(path).name,'custom_database_hits.a3m');self.assertTrue(Path(path).exists())
   with self.assertRaises(FileExistsError):prepare(p/'q.fa',p/'s.a3m',p/'out')
 def test_wrong_query_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.a3m';p.write_text('>q\nCCDE\n')
   with self.assertRaises(ValueError):validate_msa(p,'ACDE')
 def test_config_no_filter_flags(self):
  c=command(dict(filter_identity=False,adaptive_mask=False,num_variants=50))
  self.assertIn('--no-filter-identity',c);self.assertIn('--no-adaptive-mask',c)
  with self.assertRaises(ValueError):command(dict(filter_identity='false'))
  with self.assertRaises(ValueError):command(dict(typo=50))
if __name__=='__main__':unittest.main()
