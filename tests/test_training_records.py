import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_records', ROOT/'scripts/summarize_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainingRecordTests(unittest.TestCase):
    def test_duplicate_nonfinite_and_conflicting_scalars(self):
        summary, values = module.scalar_summary([(1, 5.), (1, 5.), (2, 3.)])
        self.assertEqual(summary['unique_steps'], 2)
        self.assertEqual(summary['first_window_median'], 4.)
        self.assertEqual(values, {1: 5., 2: 3.})
        for rows in [[(1, 5.), (1, 4.)], [(1, math.inf)], [(1, math.nan)]]:
            with self.assertRaises(ValueError):
                module.scalar_summary(rows)

    def test_partial_tail_and_crc_corruption(self):
        # A supplied independent checksum function isolates record parsing.
        payload = b'independent record contents'
        header = struct.pack('<Q', len(payload))
        record = header + struct.pack('<I', zlib.crc32(header)) + payload + struct.pack('<I', zlib.crc32(payload))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'events'
            path.write_bytes(record + record[:10])
            self.assertEqual(list(module.record_payloads(path, zlib.crc32)), [payload])
            corrupt = bytearray(record)
            corrupt[12] ^= 1
            path.write_bytes(corrupt)
            with self.assertRaises(ValueError):
                list(module.record_payloads(path, zlib.crc32))


if __name__ == '__main__':
    unittest.main()
