"""Public crop, input isolation and complete received-byte provenance CPU gates."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import contract as c


class Checks(unittest.TestCase):
    def test_exact_public_crop_keeps_unclipped_FP32(self):
        left = np.linspace(-.5, 1.5, 1*3*64*96, dtype=np.float32).reshape(1, 3, 64, 96)
        right = np.flip(left, -1).copy()
        result = c.crop_arrays(left, right, [37, 65], [64, 96])
        self.assertTrue(np.array_equal(result['left'], left[:, :, :37, :65]))
        self.assertTrue(np.array_equal(result['right'], right[:, :, :37, :65]))
        self.assertEqual(result['left'].dtype, np.float32)
        self.assertLess(float(result['left'].min()), 0)
        self.assertGreater(float(result['right'].max()), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.npz'; np.savez(path, **result)
            with np.load(path, allow_pickle=False) as f:
                self.assertEqual(set(f.files), {'left', 'right'})
                self.assertTrue(np.array_equal(f['right'], result['right']))

    def test_invalid_dimensions_shape_dtype_nonfinite_rejected(self):
        left = np.ones((1, 3, 64, 96), np.float32)
        for original, padded in [([0, 65], [64, 96]), ([32, 65], [64, 96]), ([65, 65], [64, 96]),
                                 ([37, 65], [63, 96]), ([37., 65], [64, 96]), ([37], [64, 96])]:
            with self.assertRaises(ValueError): c.crop_arrays(left, left, original, padded)
        for bad in (left.astype(np.float64), left[..., :-1], left[:, :2], np.full_like(left, np.nan)):
            with self.assertRaises(ValueError): c.crop_arrays(bad, left, [37, 65], [64, 96])

    def test_fresh_process_read_barrier_only_allows_own_decoded_NPZ(self):
        with tempfile.TemporaryDirectory() as folder:
            own = Path(folder) / 'reconstructed.npz'; other = Path(folder) / 'source.npz'
            np.savez(own, left=np.ones(1)); np.savez(other, left=np.zeros(1))
            code = ('import sys,numpy as np; from contract import input_guard; '
                    'sys.addaudithook(input_guard([sys.argv[1]])); '
                    'np.load(sys.argv[2],allow_pickle=False).close()')
            for path, good in ((own, True), (other, False)):
                run = subprocess.run([sys.executable, '-c', code, str(own), str(path)], cwd=c.HERE, capture_output=True, text=True)
                self.assertEqual(run.returncode == 0, good)
                if not good: self.assertIn('crop tried foreign/source NPZ', run.stderr)

    def test_all_actual_received_bytes_and_public_headers(self):
        manifest = c.read(c.ROOT / 'data/engineering/artemis-ecsic-digital-CPU-001.json')
        index = c.read(c.ROOT / 'data/provenance/ecsic-received-transfer-index-001.json')
        parser = c.codec(); received = erased = 0
        for x in manifest['packets']:
            if x['reception']['state'] == 'erased':
                self.assertIsNone(x['received_paths']); erased += 1; continue
            received += 1
            p = x['received_paths']; payload = c.ROOT / p['payload']; wire = c.ROOT / p['wire']
            self.assertEqual(c.sha(payload), p['payload_sha256'])
            self.assertEqual(c.sha(wire), p['wire_sha256'])
            self.assertEqual(wire.read_bytes()[20:], payload.read_bytes())
            parsed = parser.unpack(payload.read_bytes(), c.CDF_SHA)
            self.assertEqual(parsed['original_hw'], x['reception']['original_hw'])
            self.assertEqual(parsed['padded_hw'], x['reception']['padded_hw'])
        self.assertEqual((received, erased, len(index['files_sha256'])), (11, 9, 22))

    def test_locked_sources_protocol_and_ast_syntax(self):
        self.assertTrue(all(c.sha(c.ROOT / p) == h for p, h in c.LOCKED.items()))
        for path in c.HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
        before = c.identities(); self.assertEqual(before, c.identities())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert not args.output.exists()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    value = dict(state='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                 failures=[str(t) + trace for t, trace in result.failures], errors=[str(t) + trace for t, trace in result.errors],
                 sources=c.identities(), numpy=np.__version__, protocol_sha256=c.sha(c.PROTOCOL),
                 scope='Actual accepted byte/header identities plus synthetic exact-crop and separate-process source isolation; no native neural/AP result')
    c.save(args.output, value)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
