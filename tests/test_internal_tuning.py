import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tuning',ROOT/'scripts/prepare_internal_tuning.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TuningTests(unittest.TestCase):
    def test_fold_never_selects_main_validation_and_preserves_order(self):
        original=['%06d'%x for x in range(20)]
        validation=['%06d'%x for x in range(20,30)]
        train,heldout=module.split_ids(original,validation,5)
        self.assertEqual(len(train),15);self.assertEqual(len(heldout),5)
        self.assertEqual(set(train)|set(heldout),set(original))
        self.assertFalse(set(train)&set(heldout));self.assertFalse(set(heldout)&set(validation))
        self.assertEqual(train,[x for x in original if x in train])
        self.assertEqual(heldout,[x for x in original if x in heldout])
        self.assertEqual((train,heldout),module.split_ids(original,validation,5))
        with self.assertRaisesRegex(ValueError,'overlap'):
            module.split_ids(original,original[:2],5)
