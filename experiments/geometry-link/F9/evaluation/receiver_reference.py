"""Engineering-only exact receiver equality using the SAME received BEV prefix."""
import hashlib
import json

import numpy as np
import torch

from common import check, rng_record


def value_identity(value):
    """Passive identity of all original prefix values, including nested tensors."""
    if isinstance(value, torch.Tensor): value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        return dict(shape=list(value.shape), dtype=value.dtype.str,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, (tuple, list)): return [value_identity(v) for v in value]
    if isinstance(value, dict): return {str(k): value_identity(v) for k, v in value.items()}
    if isinstance(value, np.generic): return value.item()
    if value is None or type(value) in (bool, int, float, str): return value
    # Native public calibration object; no opaque features silently omitted.
    check(type(value).__name__ == 'Calibration', 'unsupported receiver-prefix value: ' + str(type(value)))
    return dict(calibration=value_identity(vars(value)))


def prefix_identity(batch):
    return {k: value_identity(v) for k, v in batch.items()}


def prediction_arrays(predictions):
    check(len(predictions) == 1, 'single reference prediction')
    arrays = {k: predictions[0][k].detach().cpu().numpy().copy()
              for k in ('pred_boxes', 'pred_scores', 'pred_labels')}
    check(all(np.isfinite(v).all() for v in arrays.values()), 'finite exact3D outputs')
    return arrays


class ReceiverReference:
    def __init__(self, model, observer):
        self.model, self.observer = model, observer
        self.prefix = None
        self.identity = None
        self.handle = model.dense_head.register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        if self.prefix is None:
            check(self.observer.calls['frames'] == 1, 'capture first received prefix only')
            self.prefix = dict(inputs[0])
            check('spatial_features_2d' in self.prefix, 'received BEV input required')
            check(not any('gt' in k or k == 'random_T' for k in self.prefix), 'reference label barrier')
            self.identity = prefix_identity(self.prefix)

    def compare(self, predictions):
        self.handle.remove()
        check(self.prefix is not None and prefix_identity(self.prefix) == self.identity, 'explicit3D head mutated prefix inputs')
        expected = prediction_arrays(predictions)
        device = self.prefix['spatial_features_2d'].device
        rng_before = dict(cpu=rng_record(torch.device('cpu')), receiver=rng_record(device))
        reference = dict(self.prefix)
        with self.observer.reference_scope():
            reference = self.model.dense_head_2d(reference)
            check(set(reference) - set(self.prefix) == {'head_outs', 'boxes_2d_pred'}, 'native auxiliary written keys differ')
            check(all(k in reference and value_identity(reference[k]) == v for k, v in self.identity.items()), 'auxiliary changed original receiver input')
            reference = self.model.dense_head(reference)
            depth_before = prefix_identity(reference)
            reference = self.model.depth_loss_head(reference)
            check(prefix_identity(reference) == depth_before, 'depth head withoutGT changed reference')
            received_predictions, _ = self.model.post_processing(reference)
        actual = prediction_arrays(received_predictions)
        check(all(np.array_equal(expected[k], actual[k]) and expected[k].dtype == actual[k].dtype for k in expected), 'native auxiliary reference changed3D output')
        check(prefix_identity(self.prefix) == self.identity, 'reference changed original received prefix')
        rng_after = dict(cpu=rng_record(torch.device('cpu')), receiver=rng_record(device))
        check(rng_after == rng_before, 'reference consumed new randomness')
        payload = json.dumps(self.identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        report = dict(state='passed', all_boxes_scores_classes_exact=True, original_prefix_unchanged=True,
                      prefix_value_identities=self.identity, prefix_identity_sha256=hashlib.sha256(payload).hexdigest(),
                      allowed_auxiliary_written_keys=['boxes_2d_pred', 'head_outs'], auxiliary_calls=self.observer.reference_calls.copy(),
                      new_sender_channel_cost_calls=0, no_new_randomness=rng_before == rng_after,
                      RNG_before=rng_before, RNG_after=rng_after,
                      scope='Same actual received prefix; eight first-frame engineering references only, no AP')
        self.prefix = None
        return report, {**{'explicit_' + k: v for k, v in expected.items()},
                        **{'reference_' + k: v for k, v in actual.items()}}

    def close(self):
        self.handle.remove()
        self.prefix = None
