"""Guard the identity/constraint contract before any paid GPU generation."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("dplm_wrapper", Path(__file__).resolve().parents[1] / "scripts/modal_dplm_homologs.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ConstraintContract(unittest.TestCase):
    def test_unfiltered_mode_retains_native_mutation_counts(self):
        self.assertEqual(m.candidate_status('CAAA', 'AAAA', None, 0, set(), set()), ('accepted', 1))
        self.assertEqual(m.candidate_status('CCCC', 'AAAA', None, 0, set(), set()), ('accepted', 4))
        self.assertEqual(m.candidate_status('AAAA', 'AAAA', None, 0, set(), set())[0], 'duplicate')

    def test_unfiltered_settings_ignore_target_and_disallow_adaptation(self):
        settings=dict(filter_identity=False,target_identity=1.0,identity_tolerance=0,
                      num_variants=1,batch_size=1,max_candidates=1,max_iter=1,
                      mask_fraction=.4,sampling_strategy='gumbel_argmax',temperature=1,
                      fixed_positions='',mutable_positions='',adaptive_mask=False)
        self.assertEqual(m.validate_settings('ACDE',settings),[0,1,2,3])
        settings['adaptive_mask']=True
        with self.assertRaises(ValueError):m.validate_settings('ACDE',settings)

    def test_identity_rounding_for_benchmark(self):
        query = "A" * 159
        seq = "C" * 24 + "A" * 135
        target = round((1 - .85) * len(query))
        self.assertEqual(m.candidate_status(seq, query, target, 0, {158}, set()), ("accepted", 24))
        self.assertEqual(m.candidate_status("C" * 23 + "A" * 136, query, target, 0, {158}, set())[0], "identity_rejected")

    def test_protected_violation_is_fatal_not_silent_overwrite(self):
        with self.assertRaises(RuntimeError):
            m.candidate_status("CAAA", "AAAA", 1, 0, {0}, set())

    def test_invalid_and_duplicate_candidates_excluded(self):
        self.assertEqual(m.candidate_status("XAAA", "AAAA", 1, 0, set(), set())[0], "invalid_sequence")
        self.assertEqual(m.candidate_status("CAAA", "AAAA", 1, 0, set(), {"CAAA"})[0], "duplicate")

    def test_position_ranges(self):
        self.assertEqual(m.parse_positions("1,3-5", 5), {0, 2, 3, 4})
        for bad in ["0", "6", "4-2"]:
            with self.assertRaises(ValueError):
                m.parse_positions(bad, 5)

    def test_iterative_parents_accumulate_and_allow_reversions(self):
        import random
        calls = []
        def sampler(parents, masks):
            calls.append(list(parents))
            # Full masking lets this controlled sampler test inheritance/reversion.
            return [['CAAA', 'ACAA'], ['CCAA', 'ADAA'], ['ACAA', 'ADCA']][len(calls)-1]
        seqs, histories = m.evolve_batch('AAAA', 2, list(range(4)), [0, 0, 0], sampler, random.Random(42))
        self.assertEqual(calls, [['AAAA', 'AAAA'], ['CAAA', 'ACAA'], ['CCAA', 'ADAA']])
        self.assertEqual(seqs, ['ACAA', 'ADCA'])
        self.assertEqual([r['mutations'] for r in histories[0]], [1, 2, 1])
        self.assertEqual(histories[0][-1]['reversions'], 1)

    def test_iterative_masks_and_protection(self):
        import random
        sizes = []
        def sampler(parents, masks):
            sizes.append([len(mask) for mask in masks])
            return [''.join('C' if i in mask else aa for i, aa in enumerate(parent))
                    for parent, mask in zip(parents, masks)]
        seqs, histories = m.evolve_batch('A'*11, 2, list(range(1,11)), [.5, .8, .9], sampler, random.Random(42))
        self.assertEqual(sizes, [[5,5], [2,2], [1,1]])
        self.assertTrue(all(seq[0] == 'A' for seq in seqs))
        with self.assertRaises(RuntimeError):
            m.evolve_batch('AAAA', 1, [1,2,3], [.9], lambda p,m: ['CCCC'], random.Random(42))

    def test_invalid_intermediate_rejects_only_its_lineage(self):
        import random
        calls = []
        def sampler(parents, masks):
            calls.append(list(parents))
            return [['CAAA', 'XAAA', 'ACAA'], ['CCAA', 'ACCA'], ['CDCA', 'ACCD']][len(calls)-1]
        seqs, histories = m.evolve_batch('AAAA', 3, [0,1,2,3], [0,0,0], sampler, random.Random(42))
        self.assertEqual(calls, [['AAAA']*3, ['CAAA','ACAA'], ['CCAA','ACCA']])
        self.assertEqual(seqs, ['CDCA','XAAA','ACCD'])
        self.assertEqual([len(h) for h in histories], [3,1,3])
        self.assertEqual(histories[1][-1]['status'], 'invalid_sequence')
        self.assertEqual(m.candidate_status(seqs[1], 'AAAA', None, 0, set(), set()), ('invalid_sequence',None))

    def test_late_invalid_intermediate_preserves_history_without_accepting_ancestor(self):
        import random
        calls = []
        def sampler(parents, masks):
            calls.append(list(parents))
            return [['CAAA','ACAA'], ['XAAA','ACCA'], ['ACCD']][len(calls)-1]
        seqs, histories = m.evolve_batch('AAAA', 2, [0,1,2,3], [0,0,0], sampler, random.Random(42))
        self.assertEqual(calls[-1], ['ACCA'])
        self.assertEqual(seqs, ['XAAA','ACCD'])
        self.assertEqual([step['sequence'] for step in histories[0]], ['CAAA','XAAA'])
        self.assertEqual(histories[0][-1]['round'], 2)
        self.assertEqual(m.candidate_status(seqs[0], 'AAAA', None, 0, set(), set())[0], 'invalid_sequence')
        self.assertEqual(len(histories[1]), 3)

    def test_all_invalid_lineages_stop_without_resampling(self):
        import random
        calls = []
        def sampler(parents, masks):
            calls.append(parents)
            return ['CAA', 'AXAA']
        seqs, histories = m.evolve_batch('AAAA', 2, [0,1,2,3], [0]*50, sampler, random.Random(42))
        self.assertEqual(len(calls), 1)
        self.assertEqual(seqs, ['CAA','AXAA'])
        self.assertTrue(all(len(h)==1 and h[0]['mutations'] is None for h in histories))

    def test_round_exports_follow_selected_lineages(self):
        import tempfile
        import contextlib
        import io
        import random
        seqs, histories = m.evolve_batch('AAAA', 1, [0,1,2,3], [0,0],
            lambda parents,masks: ['CAAA' if parents[0] == 'AAAA' else 'CCAA'], random.Random(42))
        row=dict(candidate=1, sequence=seqs[0], mutations=2, identity=.5, lineage=histories[0])
        result=dict(metadata=dict(query='AAAA', complete=True), accepted=[row], candidates=[row])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)
            m.write_outputs(result, root)
            self.assertEqual((root/'synthetic.a3m').read_text().splitlines()[-1], 'CCAA')
            self.assertEqual((root/'rounds/round_01.a3m').read_text().splitlines()[-1], 'CAAA')
            self.assertEqual((root/'rounds/round_02.a3m').read_text().splitlines()[-1], 'CCAA')

    def test_keep_schedule_validation(self):
        self.assertEqual(m.keep_schedule(''), [])
        self.assertEqual(m.keep_schedule('0.8, 0.9'), [.8, .9])
        for bad in ['1', '-.1', 'nan', '.9,', 'inf']:
            with self.assertRaises(ValueError): m.keep_schedule(bad)
        settings=dict(filter_identity=True,target_identity=.5,identity_tolerance=0,
                      num_variants=1,batch_size=1,max_candidates=1,max_iter=1,
                      mask_fraction=.1,keep_fractions='.9,.9',sampling_strategy='gumbel_argmax',
                      temperature=1,fixed_positions='',mutable_positions='',adaptive_mask=False)
        self.assertEqual(m.validate_settings('ACDE', settings), [0,1,2,3])
        settings['adaptive_mask'] = True
        with self.assertRaises(ValueError): m.validate_settings('ACDE', settings)

    def test_mask_controller_brackets_sharp_response(self):
        mask, lo, hi = m.next_mask_size(66, 25, 5, 164, None, None)
        self.assertEqual(mask, 115)  # Previously jumped directly to full masking.
        mask, lo, hi = m.next_mask_size(mask, 25, 150, 164, lo, hi)
        self.assertTrue(lo < mask < hi)
        for _ in range(12):
            observed = 5 if mask < 100 else 150
            mask, lo, hi = m.next_mask_size(mask, 25, observed, 164, lo, hi)
        self.assertIn(mask, (99, 100))


if __name__ == "__main__":
    unittest.main()
