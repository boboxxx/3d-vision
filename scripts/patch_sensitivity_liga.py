#!/usr/bin/env python3
"""Expose native 3D loss for optional training-only sensitivity supervision."""
import argparse
from pathlib import Path


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('sensitivity patch anchor mismatch: ' + old[:100])
    return text.replace(old, new, 1)


def patch(root):
    head = root / 'liga/models/dense_heads/anchor_head_template.py'
    detector = root / 'liga/models/detectors_stereo/liga.py'
    h, d = head.read_text(), detector.read_text()
    if '# GEOCOMM_NATIVE_3D_LOSS' in h:
        if '# GEOCOMM_SENSITIVITY_LOSS' not in d:
            raise RuntimeError('partial sensitivity patch')
        return False
    if '# GEOCOMM_TASK_LOSS' not in d:
        raise RuntimeError('apply geometry link patch first')
    h = replace(h, '        rpn_loss = cls_loss + box_loss\n', '''        rpn_loss = cls_loss + box_loss
        # GEOCOMM_NATIVE_3D_LOSS: a separate sum excludes in-place imitation additions.
        if getattr(self, 'communication_sensitivity_enabled', False):
            self.forward_ret_dict['communication_detection_loss'] = cls_loss + box_loss
''')
    d = replace(d, '        self.module_list = self.build_networks()\n', '''        self.module_list = self.build_networks()
        link = self.backbone_3d.semantic_link
        self.dense_head.communication_sensitivity_enabled = (
            link is not None and link.sensitivity_predictor is not None)
''')
    d = replace(d, '        loss = loss_rpn + loss_depth\n', '''        loss = loss_rpn + loss_depth
        # GEOCOMM_SENSITIVITY_LOSS: no GT gradients or labels at inference.
        if 'communication_sensitivity_context' in batch_dict:
            received, importance = batch_dict.pop('communication_sensitivity_context')
            link = self.backbone_3d.semantic_link
            sensitivity_loss, target_log = link.sensitivity_predictor.distillation_loss(
                self.dense_head.forward_ret_dict.pop('communication_detection_loss'),
                received, importance)
            loss = loss + link.sensitivity_weight * sensitivity_loss
            tb_dict['loss_communication_sensitivity'] = sensitivity_loss.item()
            tb_dict['communication_sensitivity_target_log_std'] = target_log.std().item()
''')
    compile(h, str(head), 'exec')
    compile(d, str(detector), 'exec')
    head.write_text(h)
    detector.write_text(d)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('third_party/LIGA-Stereo'))
    args = parser.parse_args()
    print('patched optional sensitivity' if patch(args.root) else 'already patched')
