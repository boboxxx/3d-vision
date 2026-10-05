"""Compare received cleanRGB relay with true native LIGA evaluation preprocessing."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti'),str(ROOT/'reproduction/cao2025')]
from adapter import liga_preprocess
from data import StereoRGB
from geocomm.evidence import sha256,source_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output=args.output.resolve()
    if args.output.exists():p.error('preserve previous evidence')
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    before=source_identity(ROOT,['src','scripts','configs','pyproject.toml'])
    os.chdir(LIGA)
    from easydict import EasyDict
    from liga.config import cfg_from_yaml_file
    from liga.datasets import build_dataloader
    from liga.datasets.augmentor.stereo_data_augmentor import StereoDataAugmentor
    from liga.datasets.stereo_dataset_template import StereoDatasetTemplate
    from liga.utils.calibration_kitti import Calibration
    from liga.utils.common_utils import create_logger
    cfg=cfg_from_yaml_file(str(ROOT/'configs/diagnostic/clean_holdout.yaml'),EasyDict())
    dataset,loader,_=build_dataloader(cfg.DATA_CONFIG,cfg.CLASS_NAMES,batch_size=1,dist=False,workers=0,training=False,logger=create_logger())
    reference=next(iter(loader))
    if str(reference['frame_id'][0])!='000036':raise RuntimeError('fixed firstnative reference frame differs')
    cache=StereoRGB('/mnt/d/paper6/data/kitti',ROOT/'data/engineering/cao2025-roi-holdout-001.jsonl',
        ROOT/'data/engineering/cao2025-roi-holdout-audit-001.json',ROOT/'data/internal-tuning-fold-001.json','geocomm_tune_holdout')
    frame=cache[0]
    calib_file=Path('/mnt/d/paper6/data/kitti/training/calib/000036.txt');calibration=Calibration(calib_file)
    initialP2=calibration.P2.copy();initialP3=calibration.P3.copy()
    augmentor=StereoDataAugmentor(dataset.root_path,cfg.DATA_CONFIG.TEST_DATA_AUGMENTOR,cfg.CLASS_NAMES)
    actual=liga_preprocess((frame['left'],frame['right']),calibration,frame['frame_id'],augmentor,StereoDatasetTemplate.collate_batch)
    for name in ('left_img','right_img'):
        if not torch.equal(actual[name],torch.as_tensor(reference[name])):
            raise RuntimeError('exact true native cleanrelay preprocessing differs: '+name)
    for name in ('image_shape','frame_id'):
        if not np.array_equal(actual[name],reference[name]):raise RuntimeError('native metadata differs: '+name)
    for name in ('calib','calib_ori'):
        for field in ('P2','P3','R0','V2C'):
            if not np.array_equal(getattr(actual[name][0],field),getattr(reference[name][0],field)):
                raise RuntimeError('native public calibration differs: '+name+'.'+field)
    again=liga_preprocess((frame['left'],frame['right']),calibration,frame['frame_id'],augmentor,StereoDatasetTemplate.collate_batch)
    if not np.array_equal(again['calib'][0].P2,actual['calib'][0].P2) or not np.array_equal(initialP2,calibration.P2) or not np.array_equal(initialP3,calibration.P3):
        raise RuntimeError('caller calibration accumulated crop transforms')
    high=torch.full_like(frame['left'],1.25)
    unclipped=liga_preprocess((high,high),calibration,frame['frame_id'],augmentor,StereoDatasetTemplate.collate_batch)
    value=float(unclipped['left_img'][0,0,0,0]);expected=(1.25-.485)/.229
    if not np.isclose(value,expected,rtol=1e-6):raise RuntimeError('decodedfloat range was clipped/quantized')
    if source_identity(ROOT,['src','scripts','configs','pyproject.toml'])!=before:raise RuntimeError('frozen root changed')
    original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    if len(original)!=21 or any(sha256(ROOT/'reproduction/cao2025'/name)!=v for name,v in original.items()):
        raise RuntimeError('original21 sources changed')
    result=dict(state='passed',scope='true native LIGA CPU preprocessing comparison only, no model/GPU/AP',frame_id='000036',
        native_sensor_RGB_shape=list(frame['left'].shape),processed_shape=list(actual['left_img'].shape),
        original_image_shape=actual['image_shape'].tolist(),crop_calibration_offsets=actual['calib'][0].offsets,
        original_calibration_offsets=actual['calib_ori'][0].offsets,caller_calibration_unchanged=True,
        exact_native_left_right_tensors=True,exact_native_calibration_and_metadata=True,
        normalized_out_of_range_input=value,helper_input_keys=['received_RGB','public_calibration','frame_id','native_crop','native_collation'],
        prepared_batch_keys=sorted(actual),reference_GT_LiDAR_never_passed_to_helper_or_model=True,
        model_inference_calls=0,config_sha256=sha256(ROOT/'configs/diagnostic/clean_holdout.yaml'),calibration_sha256=sha256(calib_file),
        adapter_sources=source_identity(HERE,['adapter.py','probe_liga_rgb_preprocess_cpu.py']),
        project_sources_unchanged=before,original21_sources_unchanged=original,
        protocol_sha256=sha256(HERE/'LIGA-RGB-endpoint.md'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['state','processed_shape','original_image_shape','crop_calibration_offsets','model_inference_calls']}))

if __name__=='__main__':main()
