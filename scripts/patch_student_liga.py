#!/usr/bin/env python3
"""Optional distilled sender; preserve the complete original receiver."""
import argparse
from pathlib import Path


def replace(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError('pinned student patch anchor mismatch: '+old[:100])
    return text.replace(old,new,1)


def patch(root):
    path = root/'liga/models/backbones_3d_stereo/liga_backbone.py'
    detector = root/'liga/models/detectors_stereo/liga.py'
    text,det = path.read_text(),detector.read_text()
    if '# GEOCOMM_STUDENT_ENCODER' in text:
        if '# GEOCOMM_STUDENT_LOSS' not in det:
            raise RuntimeError('partial student patch')
        if '# GEOCOMM_UNCOMPRESSED_DIAGNOSTIC' not in text:
            old="""            if self.semantic_link is None or self.semantic_link_boundary != 'raw_cost':
                raise ValueError('student requires a raw-cost geometry communication boundary')"""
            new="""            # GEOCOMM_UNCOMPRESSED_DIAGNOSTIC: explicitly configured no-channel ablation.
            if self.semantic_link is None:
                if not student_cfg.get('allow_uncompressed_diagnostic', False):
                    raise ValueError('student without communication requires explicit diagnostic flag')
            elif self.semantic_link_boundary != 'raw_cost':
                raise ValueError('student requires a raw-cost geometry communication boundary')"""
            text=replace(text,old,new)
            compile(text,str(path),'exec');path.write_text(text)
            return True
        return False
    if '# GEOCOMM_EARLIER_BOUNDARY' not in text or '# GEOCOMM_TASK_LOSS' not in det:
        raise RuntimeError('apply geometry boundary patch first')
    text = replace(text,'    def build_depth_pred_module(self):', '''        # GEOCOMM_STUDENT_ENCODER: original features are a training-only teacher.
        self.student_semantic_link_encoder = None
        student_cfg = self.model_cfg.get('STUDENT_ENCODER', {})
        if student_cfg.get('enabled', False):
            if self.semantic_link is None or self.semantic_link_boundary != 'raw_cost':
                raise ValueError('student requires a raw-cost geometry communication boundary')
            if not self.fullres_stereo_feature:
                raise ValueError('student supports the released full-resolution stereo feature interface')
            from geocomm.student import StereoStudentEncoder
            self.student_semantic_link_encoder = StereoStudentEncoder(
                self.feature_neck.stereo_dim[-1], self.feature_neck.sem_dim[-1])
            self.student_distill_weight = float(student_cfg.get('distill_weight', 0.1))
            if self.student_distill_weight < 0:
                raise ValueError('negative distillation weight')
            for module in [self.feature_backbone, self.feature_neck]:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def build_depth_pred_module(self):''')
    original = '''        left_features = self.feature_backbone(left)
        left_features = [left] + list(left_features)
        right_features = self.feature_backbone(right)
        right_features = [right] + list(right_features)

        left_stereo_feat, left_sem_feat = self.feature_neck(left_features)
        right_stereo_feat, _ = self.feature_neck(right_features)
'''
    new = '''        if self.student_semantic_link_encoder is None:
            left_features = self.feature_backbone(left)
            left_features = [left] + list(left_features)
            right_features = self.feature_backbone(right)
            right_features = [right] + list(right_features)
            left_stereo_feat, left_sem_feat = self.feature_neck(left_features)
            right_stereo_feat, _ = self.feature_neck(right_features)
        else:
            left_stereo_feat, left_sem_feat = self.student_semantic_link_encoder(left)
            right_stereo_feat, _ = self.student_semantic_link_encoder(right)
            if self.training and self.student_distill_weight > 0:
                from geocomm.student import teacher_feature_targets, feature_distillation
                targets = teacher_feature_targets(self.feature_backbone, self.feature_neck, left, right)
                batch_dict['communication_student_loss'] = self.student_distill_weight * feature_distillation(
                    (left_stereo_feat, right_stereo_feat, left_sem_feat), targets)
'''
    text = replace(text,original,new)
    det = replace(det,'        loss = loss_rpn + loss_depth\n', '''        loss = loss_rpn + loss_depth
        # GEOCOMM_STUDENT_LOSS: feature supervision only during training.
        if 'communication_student_loss' in batch_dict:
            loss += batch_dict['communication_student_loss']
            tb_dict['loss_communication_student'] = batch_dict['communication_student_loss'].item()
''')
    compile(text,str(path),'exec')
    compile(det,str(detector),'exec')
    path.write_text(text)
    detector.write_text(det)
    patch(root)
    return True


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('third_party/LIGA-Stereo'))
    args = parser.parse_args()
    print('patched optional student encoder' if patch(args.root) else 'already patched')
