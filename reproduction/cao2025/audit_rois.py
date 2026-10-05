"""Independent complete-fold image/ROI/mask identity audit, without GT access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('records','fold','data-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve prior audit; choose unique output')
    manifest_path = args.records.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((Path(__file__).parent/'upstream/yolov5.json').read_text())
    assert manifest['state']=='finished' and manifest['evidence_type']=='sensor_only_YOLO_ROI_fold_extraction'
    assert manifest['checkpoint_sha256']==metadata['checkpoint_sha256']
    assert manifest['upstream_revision']==metadata['revision']
    assert manifest['metadata_sha256']==digest(Path(__file__).parent/'upstream/yolov5.json')
    assert manifest['source_sha256']==digest(Path(__file__).parent/'extract_rois.py')
    assert manifest['records_sha256']==digest(args.records)
    assert manifest['fold_sha256']==digest(args.fold)
    fold = json.loads(args.fold.read_text())
    assert manifest['split'] in ('geocomm_tune_train','geocomm_tune_holdout')
    ids = fold['folds'][manifest['split']]['ids']
    rows = [json.loads(line) for line in args.records.read_text().splitlines()]
    assert [row['frame_id'] for row in rows]==ids
    assert len(rows)==manifest['frames']
    areas,counts = [],[]
    for row in rows:
        assert [view['camera'] for view in row['views']]==['image_2','image_3']
        for view in row['views']:
            path=args.data_root/'training'/view['camera']/(row['frame_id']+'.png')
            assert digest(path)==view['image_sha256']
            with Image.open(path) as image:
                w,h=image.size
            assert view['shape']==[h,w]
            assert view['letterbox_shape'][:2]==[1,3]
            assert all(size%32==0 and 0<size<=640 for size in view['letterbox_shape'][2:])
            mask=np.zeros((h,w),dtype=np.uint8)
            for box in view['boxes']:
                x1,y1,x2,y2=box['xyxy']
                assert all(isinstance(value,int) for value in box['xyxy'])
                assert 0<=x1<x2<=w and 0<=y1<y2<=h
                assert box['class_id'] in (2,5,7) and np.isfinite(box['confidence'])
                assert .25<=box['confidence']<=1
                mask[y1:y2,x1:x2]=1
            assert len(view['boxes'])<=300
            assert int(mask.sum())==view['union_area_pixels']
            assert hashlib.sha256(mask.tobytes()).hexdigest()==view['mask_uint8_sha256']
            areas.append(float(mask.mean()))
            counts.append(len(view['boxes']))
    result=dict(state='passed',evidence_type='independent_full_fold_ROI_coverage_and_union_mask_audit',
        frames=len(rows),views=len(areas),split=manifest['split'],records_sha256=digest(args.records),
        manifest_sha256=digest(manifest_path),fold_sha256=digest(args.fold),
        mean_union_mask_fraction=float(np.mean(areas)),empty_views=sum(value==0 for value in counts),
        mean_boxes_per_view=float(np.mean(counts)),max_boxes_per_view=max(counts),
        limitations='verifies image identity and saved ROI/mask geometry; no fresh YOLO inference, GT quality, communication or AP claim')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
