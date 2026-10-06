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
