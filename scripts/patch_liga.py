#!/usr/bin/env python3
"""Install a reversible optional link into the pinned official LIGA checkout."""
import argparse
from pathlib import Path

PIN = "aee3731a24a0ab1667e633e520cc89be2f135272"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Upstream anchor mismatch; refusing partial patch: " + old[:80])
    return text.replace(old, new, 1)


def patch(root):
    backbone = root / "liga/models/backbones_3d_stereo/liga_backbone.py"
    detector = root / "liga/models/detectors_stereo/liga.py"
    src = backbone.read_text(encoding="utf-8")
    det = detector.read_text(encoding="utf-8")
    evaluation = root / "tools/eval_utils/eval_utils.py"
    ev = evaluation.read_text(encoding="utf-8")
    if "# GEOCOMM_EVIDENCE" not in ev:
        ev = replace_once(ev, "    start_time = time.time()\n", '''    # GEOCOMM_EVIDENCE: keep per-batch channel/energy measurements.
    from geocomm.evidence import save_link_accounting
    communication_file = result_dir / ('communication_rank%d.jsonl' % cfg.LOCAL_RANK)
    communication_file.write_text('', encoding='utf-8')
    start_time = time.time()
''')
        ev = replace_once(ev, "        disp_dict = {}\n", "        save_link_accounting(communication_file, batch_dict)\n        disp_dict = {}\n")
        compile(ev, str(evaluation), "exec")
        evaluation.write_text(ev, encoding="utf-8")
    # DDP can rebuild input dictionaries; the returned sensor batch is authoritative.
    if 'save_link_accounting(communication_file, batch_dict)' in ev:
        ev = replace_once(ev, '        save_link_accounting(communication_file, batch_dict)\n',
            "        # GEOCOMM_PREDICTION_ACCOUNTING: survive DDP input dictionary copies.\n"
            "        from geocomm.evidence import prediction_link_accounting\n"
            "        prediction_link_accounting(communication_file, pred_dicts, batch_dict)\n")
        compile(ev,str(evaluation),'exec')
        evaluation.write_text(ev,encoding='utf-8')
    if "# GEOCOMM_BOUNDARY" in src and "# GEOCOMM_TASK_LOSS" in det:
        changed = False
        if '# GEOCOMM_EARLIER_BOUNDARY' not in src:
            src = replace_once(src,"            options.pop('enabled')\n",'''            options.pop('enabled')
            # GEOCOMM_EARLIER_BOUNDARY: move original dres0/dres1 to receiver.
            self.semantic_link_boundary = options.pop('boundary', 'processed_cost')
            if self.semantic_link_boundary not in ('processed_cost', 'raw_cost'):
                raise ValueError('unknown communication boundary')
''')
            src = replace_once(src,'                cost_channels=self.cv_dim,',
                "                cost_channels=CV_INPUT_DIM if self.semantic_link_boundary == 'raw_cost' else self.cv_dim,")
            src = replace_once(src,'        cost0 = self.dres0(cost_raw)\n', '''        if self.semantic_link is not None and self.semantic_link_boundary == 'raw_cost':
            cost_raw, left_sem_feat = self.semantic_link(
                cost_raw, left_sem_feat, batch_dict, downsampled_depth)
        cost0 = self.dres0(cost_raw)
''')
            src = replace_once(src,'        if self.semantic_link is not None:\n',
                "        if self.semantic_link is not None and self.semantic_link_boundary == 'processed_cost':\n")
            changed = True
        if '# GEOCOMM_NONLINEAR_POSTERIOR' not in src:
            src = replace_once(src,'            self.semantic_link = GeometryLink(\n', '''            # GEOCOMM_NONLINEAR_POSTERIOR: raw concat requires left/right coupling.
            options.setdefault('posterior_hidden', 16 if self.semantic_link_boundary == 'raw_cost' else 0)
            self.semantic_link = GeometryLink(
''')
            changed = True
        if changed:
            compile(src,str(backbone),'exec')
            backbone.write_text(src,encoding='utf-8')
        return changed
    if "# GEOCOMM_BOUNDARY" in src or "# GEOCOMM_TASK_LOSS" in det:
        raise RuntimeError("Partial existing patch; inspect before proceeding")
    init_anchor = "    def build_depth_pred_module(self):"
    init = '''        # GEOCOMM_BOUNDARY: initialize after upstream weight initialization.
        self.semantic_link = None
        link_cfg = self.model_cfg.get('SEMANTIC_LINK', None)
        if link_cfg is not None and link_cfg.get('enabled', False):
            from geocomm.link import GeometryLink
            options = dict(link_cfg)
            options.pop('enabled')
            self.semantic_link = GeometryLink(
                cost_channels=self.cv_dim,
                appearance_channels=self.feature_neck.sem_dim[-1], **options)

'''
    src = replace_once(src, init_anchor, init + init_anchor)
    clean_semantics = '''        if self.sem_neck is not None:
            batch_dict['sem_features'] = self.sem_neck([left_sem_feat])
        else:
            batch_dict['sem_features'] = [left_sem_feat]

        batch_dict['rpn_feature'] = left_sem_feat

'''
    src = replace_once(src, clean_semantics, "")
    cost_anchor = "        cost0 = self.dres1(cost0) + cost0\n"
    link_call = '''        if self.semantic_link is not None:
            cost0, left_sem_feat = self.semantic_link(
                cost0, left_sem_feat, batch_dict, downsampled_depth)

'''
    src = replace_once(src, cost_anchor, cost_anchor + link_call + clean_semantics)
    loss_anchor = "        loss = loss_rpn + loss_depth\n"
    loss = '''        # GEOCOMM_TASK_LOSS: task gradients pass through received latents.
        if 'communication_aux_loss' in batch_dict:
            loss += batch_dict['communication_aux_loss']
            tb_dict['loss_communication_posterior'] = batch_dict['communication_aux_loss'].item()
        if 'communication_accounting' in batch_dict:
            account = batch_dict['communication_accounting']
            tb_dict['communication_complex_uses'] = account['total_complex_uses']
            tb_dict['communication_cbr'] = account['cbr_complex_per_input_real_scalar']
            tb_dict['communication_energy'] = account['tx_energy_per_frame'].mean().item()

'''
    det = replace_once(det, loss_anchor, loss_anchor + loss)
    # Both edits are validated in memory before either file is changed.
    compile(src, str(backbone), "exec")
    compile(det, str(detector), "exec")
    backbone.write_text(src, encoding="utf-8")
    detector.write_text(det, encoding="utf-8")
    # Apply the optional raw-cost boundary migration to fresh checkouts as well.
    patch(root)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print("Patched optional communication boundary" if patch(args.root) else "Already patched")
