#!/usr/bin/env python3
"""Activate configured gradient clipping and reject nonfinite updates."""
import argparse
from pathlib import Path


def patch(root):
    path=root/'tools/train_utils/train_utils.py'
    text=path.read_text()
    if '# GEOCOMM_FINITE_TRAINING_UPDATE' in text:return False
    old='''        loss.backward()
        # if getattr('optim_cfg', 'GRAD_NORM_CLIP', None) is not None:
        #     clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()'''
    if text.count(old)!=1:raise RuntimeError('pinned training update anchor mismatch')
    new='''        # GEOCOMM_FINITE_TRAINING_UPDATE: apply the declared optimization protocol.
        if not torch.isfinite(loss).all():
            raise FloatingPointError('nonfinite training loss before optimizer update')
        loss.backward()
        gradient_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP,
                                       error_if_nonfinite=True)
        tb_dict['gradient_norm_before_clip'] = gradient_norm.item()
        optimizer.step()'''
    text=text.replace(old,new,1);compile(text,str(path),'exec');path.write_text(text)
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print('patched finite updates and configured clipping' if patch(args.root) else 'already patched')
