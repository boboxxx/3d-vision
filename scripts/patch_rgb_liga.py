#!/usr/bin/env python3
"""Optional RGB comparator before all complete detector computation."""
import argparse
from pathlib import Path


def patch(root):
    path = root/'liga/models/detectors_stereo/liga.py'
    text = path.read_text()
    if '# GEOCOMM_RGB_BOUNDARY' in text:
        return False
    init = '        self.module_list = self.build_networks()\n'
    forward = '    def forward(self, batch_dict):\n'
    if text.count(init)!=1 or text.count(forward)!=1:
        raise RuntimeError('pinned complete detector anchors do not match')
    text = text.replace(init,init+'''        # GEOCOMM_RGB_BOUNDARY: exclusive with geometry communication.
        self.rgb_semantic_link = None
        rgb_cfg = model_cfg.get('RGB_LINK', None)
        if rgb_cfg is not None and rgb_cfg.get('enabled', False):
            if model_cfg.BACKBONE_3D.get('SEMANTIC_LINK', {}).get('enabled', False):
                raise ValueError('RGB and geometry boundaries cannot both be enabled')
            from geocomm.rgb_link import RGBLink
            options = dict(rgb_cfg)
            options.pop('enabled')
            self.rgb_semantic_link = RGBLink(**options)

''',1)
    text = text.replace(forward,forward+'''        if self.rgb_semantic_link is not None:
            batch_dict['left_img'], batch_dict['right_img'] = self.rgb_semantic_link(
                batch_dict['left_img'], batch_dict['right_img'], batch_dict)
''',1)
    compile(text,str(path),'exec')
    path.write_text(text)
    return True


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('third_party/LIGA-Stereo'))
    args = parser.parse_args()
    print('patched RGB comparator' if patch(args.root) else 'already patched')
