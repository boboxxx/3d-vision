#!/usr/bin/env python3
"""Check full result-file reporter at both thresholds using a perfect fixture."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from eval_stereo_rcnn import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('retain evidence; use a unique output')
    report = dict(evidence_type='engineering_only',state='running',
                  fixture='100 identical exact synthetic cars; not KITTI benchmark AP')
    try:
        # Require coexistence with an initialized PyTorch CUDA context in WSL.
        torch.zeros(1,device='cuda')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'kitti'
            run = Path(directory)/'run'
            (root/'training/label_2').mkdir(parents=True)
            (run/'data').mkdir(parents=True)
            ids = [f'{i:06d}' for i in range(100)]
            for frame in ids:
                label = 'Car 0 0 0 40 40 200 200 1.6 1.6 4 0 1.6 20 0'
                (root/'training/label_2'/f'{frame}.txt').write_text(label+'\n')
                (run/'data'/f'{frame}.txt').write_text(label+' 0.9\n')
            result = metrics(root,ids,run)
            values = list(result['primary_Car_IoU_0_7_R40'].values())
            values += [value for row in result['secondary_Car_IoU_0_5'].values() for value in row]
            assert all(abs(value-100)<1e-6 for value in values),values
            report.update(state='passed',checked_metrics=result,
                          expected_percent=100,torch_context_initialized=True)
    except BaseException as error:
        report.update(state='failed',error=repr(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
