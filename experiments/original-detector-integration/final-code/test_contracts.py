"""Corruption/scope tests, CPU synthetic arrays; no native detection/AP claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import common
from common import tensor_identity, sha256
from audit_cache import expected_resources, audit_frame_resources
from audit_endpoint import validate_prediction


class Contracts(unittest.TestCase):
    def test_received_cache_retains_out_of_range_floats_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outputs = (torch.full((1, 3, 4, 6), 1.7), torch.full((1, 3, 4, 6), -.2))
            frame = common.ordered_ids()[0]
            payload = base / 'received.pth'
            torch.save(dict(frame_id=frame, channel='awgn', outputs=outputs), payload)
            row = dict(native_shape=[1, 3, 4, 6], erasure=None, received_path=str(payload),
                       received_file_sha256=sha256(payload), received_tensors=[tensor_identity(v) for v in outputs])
            record = dict(state='finished', channel='awgn', ordered_ids=common.ordered_ids(),
                          frames_complete=372, frames={frame:row})
            manifest = base / 'cache.json'
            manifest.write_text(json.dumps(record))
            result, _ = common.load_received(manifest, frame)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(result, outputs)))
            self.assertEqual(result[0].max(), 1.7)
            self.assertEqual(result[1].min(), -.2)
            corrupted = copy.deepcopy(record)
            corrupted['frames'][frame]['received_tensors'][0]['array_sha256'] = '0' * 64
            manifest.write_text(json.dumps(corrupted))
            with self.assertRaises(RuntimeError): common.load_received(manifest, frame)
            manifest.write_text(json.dumps(record))
            payload.write_bytes(payload.read_bytes() + b'changed')
            with self.assertRaises(RuntimeError): common.load_received(manifest, frame)

    def test_erasure_cannot_substitute_clean_or_previous_received_array(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame = common.ordered_ids()[0]
            path = base / 'erased.pth'
            torch.save(dict(frame_id=frame, channel='awgn', outputs=None), path)
            row = dict(erasure='CRC failure', received_path=str(path), received_file_sha256=sha256(path), received_tensors=None)
            record = dict(state='finished', channel='awgn', ordered_ids=common.ordered_ids(), frames_complete=372, frames={frame:row})
            manifest = base / 'cache.json'; manifest.write_text(json.dumps(record))
            self.assertIsNone(common.load_received(manifest, frame)[0])
            torch.save(dict(frame_id=frame, channel='awgn', outputs=(torch.ones(1,3,4,6),)*2), path)
            row['received_file_sha256'] = sha256(path); manifest.write_text(json.dumps(record))
            with self.assertRaises(RuntimeError): common.load_received(manifest, frame)

    def test_union_resource_accounting_charges_empty_views_and_control_without_double_count(self):
        roi = dict(views=[dict(shape=[5,7], boxes=[dict(xyxy=[0,0,4,2]), dict(xyxy=[0,0,4,2])]),
                          dict(shape=[5,7], boxes=[])])
        expected = expected_resources(roi)
        self.assertEqual(expected['key_cells'], [2,0])
        self.assertEqual(expected['data_real_values'], 54)
        self.assertEqual(expected['data_uses'], 27)
        self.assertEqual(expected['control_uses'], 2520)
        self.assertEqual(expected['total_uses'], 2547)
        actual = dict(expected, total_energy=2547.)
        audit_frame_resources(actual, roi)
        for key in ('control_uses', 'pilot_uses', 'total_uses', 'total_energy'):
            changed = dict(actual); changed[key] += 1 if key != 'total_energy' else 100
            with self.assertRaises(RuntimeError): audit_frame_resources(changed, roi)

    def test_native_prediction_schema_keeps_other_LIGA_classes_and_rejects_erasure_AP_drop(self):
        line = 'Pedestrian -1 -1 0 0 0 10 10 1.7 .6 .6 0 1 10 0 .7\n'
        self.assertEqual(validate_prediction(line, 1, None), 1)
        with self.assertRaises(RuntimeError): validate_prediction(line, 1, None, ('Car',))
        with self.assertRaises(RuntimeError): validate_prediction(line, 1, 'CRC failure')
        bad = line.split(); bad[-1] = 'nan'
        with self.assertRaises(RuntimeError): validate_prediction(' '.join(bad), 1, None)
        self.assertEqual(validate_prediction('', 0, 'CRC failure'), 0)

    def test_final_checkpoint_gate_rejects_live_original_before_loading_partial_states(self):
        with patch.object(common, 'terminal', return_value=False), patch.object(common, 'current_sources') as sources:
            with self.assertRaises(RuntimeError): common.final_original_chain()
            sources.assert_not_called()
        with self.assertRaises(RuntimeError): tensor_identity(torch.zeros(1,3,4,6,dtype=torch.float64))
        with self.assertRaises(RuntimeError): tensor_identity(torch.full((1,3,4,6), float('nan')))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
