#!/usr/bin/env python3
"""Audit all saved full-validation predictions against their per-frame records."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve previous audits; use a unique output')
    run,root = args.run.resolve(),args.root.resolve()
    report = json.loads((run/'run.json').read_text())
    ids = (root/'ImageSets/val.txt').read_text().split()
    assert report['state']=='finished' and report['frames_complete']==len(ids)==3769
    assert report['identity']['validation_split_sha256']==sha256(root/'ImageSets/val.txt')
    assert {p.stem for p in (run/'data').glob('*.txt')}==set(ids)
    assert {p.stem for p in (run/'frames').glob('*.json')}==set(ids)
    source = json.loads(args.source_manifest.read_text())
    for tree,key in [('project','source'),('stereo_rcnn','detector_source')]:
        for name,digest in report['identity'][key]['file_hashes'].items():
            assert source[tree]['actual_sources']['file_hashes'][name]==digest,(tree,name)
    predictions,empty,nonpositive_dimensions = 0,0,0
    files = {}
    for frame in ids:
        path = run/'data'/(frame+'.txt')
        record_path = run/'frames'/(frame+'.json')
        record = json.loads(record_path.read_text())
        digest = sha256(path)
        assert record['frame_id']==frame and record['prediction_sha256']==digest
        lines = path.read_text().splitlines()
        assert len(lines)==len(record['predictions'])==record['counts']['dense_solutions']
        for line in lines:
            parts = line.split()
            assert len(parts)==16 and parts[0]=='Car'
            values = [float(v) for v in parts[1:]]
            assert all(math.isfinite(v) for v in values)
            assert -1e-6<=values[-1]<=1+1e-6
            nonpositive_dimensions += any(float(v)<=0 for v in parts[8:11])
        for folder in ['label_2','calib']:
            assert sha256(root/'training'/folder/(frame+'.txt'))==record['inputs_sha256'][folder]
        predictions += len(lines)
        empty += not lines
        files[frame] = dict(prediction_sha256=digest,frame_record_sha256=sha256(record_path))
    metrics = json.loads((run/'metrics.json').read_text())
    assert metrics==report['metrics']
    values = list(metrics['primary_Car_IoU_0_7_R40'].values())
    values += [v for row in metrics['secondary_Car_IoU_0_5'].values() for v in row]
    assert all(math.isfinite(v) and 0<=v<=100 for v in values)
    identity = json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    result = dict(evidence_type='full_KITTI_prediction_artifact_audit',state='passed',
        run=str(run),frames=3769,predictions=predictions,empty_frames=empty,
        nonpositive_dimensions=nonpositive_dimensions,run_sha256=sha256(run/'run.json'),
        metrics_sha256=sha256(run/'metrics.json'),split_sha256=sha256(root/'ImageSets/val.txt'),
        source_manifest_sha256=sha256(args.source_manifest),
        all_prediction_and_frame_hashes_sha256=hashlib.sha256(identity).hexdigest(),files=files,
        note='all saved files and label/calibration identities checked; original stereo image hashes remain in records and verified archive identity; does not reproduce the communication codec')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':
    main()
