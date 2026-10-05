import sys
from pathlib import Path
import unittest
import torch
import json
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geocomm.inference import sensor_only_prediction, SENSOR_INPUT_KEYS
from geocomm.evidence import prediction_link_accounting


class InferenceTests(unittest.TestCase):
    def test_channel_record_survives_rebuilt_DDP_input_dictionary(self):
        outer = dict(frame_id=['000001'],gt_boxes=object())
        sensor = dict(frame_id=['000001'],communication_accounting={
            'total_complex_uses':62400,'tx_energy_per_frame':torch.tensor([62400.])})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'communication.jsonl'
            prediction_link_accounting(path,[dict(batch_dict=sensor)],outer)
            row=json.loads(path.read_text())
            self.assertEqual(row['total_complex_uses'],62400)
            self.assertEqual(row['frame_id'],['000001'])
            self.assertNotIn('gt_boxes',row)

    def test_gt_lidar_excluded_and_recall_only_after_prediction(self):
        batch = dict(batch_size=1,left_img=torch.ones(1,3,2,2),right_img=torch.ones(1,3,2,2),
            calib=['camera'],image_shape=[[2,2]],frame_id=['000001'],
            gt_boxes=object(),depth_gt_img=object(),voxels=object(),points=object())
        events=[]
        boxes=torch.tensor([[20.,0.,0.,3.9,1.6,1.5,0.]])
        def forward(inputs):
            self.assertEqual(set(inputs),set(SENSOR_INPUT_KEYS))
            self.assertFalse(torch.is_grad_enabled())
            events.append('prediction_fixed')
            inputs['communication_accounting']={'total_complex_uses':17}
            inputs['left_img']=torch.zeros_like(inputs['left_img'])
            return [dict(pred_boxes=boxes,pred_scores=torch.tensor([.9]))],{}
        def recall(**kwargs):
            self.assertEqual(events,['prediction_fixed'])
            self.assertIs(kwargs['data_dict'],batch)
            self.assertIs(kwargs['box_preds'],boxes)
            events.append('evaluate_GT')
            return {'gt':1},[.8],[.9]
        predictions, diagnostics=sensor_only_prediction(forward,batch,recall,[.7])
        self.assertEqual(events,['prediction_fixed','evaluate_GT'])
        self.assertEqual(diagnostics,{'gt':1})
        self.assertEqual(predictions[0]['iou_results'],[.8])
        self.assertEqual(batch['communication_accounting']['total_complex_uses'],17)
        self.assertTrue(torch.equal(batch['left_img'],torch.ones(1,3,2,2)))


if __name__=='__main__':
    unittest.main()
