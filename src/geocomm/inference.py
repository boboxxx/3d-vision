"""Keep benchmark labels/LiDAR outside the detector inference boundary."""
import torch

SENSOR_INPUT_KEYS = ('batch_size', 'left_img', 'right_img', 'calib', 'image_shape', 'frame_id')


def sensor_only_prediction(forward, batch, recall, thresholds):
    inputs = {key: batch[key] for key in SENSOR_INPUT_KEYS if key in batch}
    if 'random_T' in batch:
        raise ValueError('sensor-only evaluation supports the released crop-only test augmentation')
    with torch.no_grad():
        predictions, diagnostics = forward(inputs)
        # Evaluation labels are accessed only after all predictions are fixed.
        if 'gt_boxes' in batch:
            for index, prediction in enumerate(predictions):
                diagnostics, iou, bev = recall(
                    box_preds=prediction['pred_boxes'], recall_dict=diagnostics,
                    batch_index=index, data_dict=batch, thresh_list=thresholds)
                if iou is not None:
                    prediction['iou_results'], prediction['ioubev_results'] = iou, bev
        for prediction in predictions:
            for key in ('pred_boxes', 'pred_scores'):
                if not torch.isfinite(prediction[key]).all():
                    raise RuntimeError('nonfinite benchmark prediction: ' + key)
    if 'communication_accounting' in inputs:
        batch['communication_accounting'] = inputs['communication_accounting']
    return predictions, diagnostics
