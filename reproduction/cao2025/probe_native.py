"""Native KITTI full-model GPU sanity, not formal training or quality evaluation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from model import SemanticVariant
from wireless import WirelessVariant
from stages import configure, forward_loss, frozen_state_keys, optimizer_groups, phase


def state_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--roi-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('preserve earlier evidence')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    audit=json.loads(args.roi_audit.read_text())
    records_bytes=args.records.read_bytes()
    assert audit['state']=='passed' and audit['records_sha256']==hashlib.sha256(records_bytes).hexdigest()
    row=json.loads(records_bytes.splitlines()[0])
    images=[];boxes=[];image_hashes={}
    for view in row['views']:
        raw=(args.data_root/'training'/view['camera']/(row['frame_id']+'.png')).read_bytes()
        image_hashes[view['camera']]=hashlib.sha256(raw).hexdigest()
        assert image_hashes[view['camera']]==view['image_sha256']
        with Image.open(io.BytesIO(raw)) as image:
            assert image.mode=='RGB' and [image.height,image.width]==view['shape']
            array=np.asarray(image).copy()
        images.append(torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().cuda()/255.)
        boxes.append(view['boxes'])
    assert len(images)==2 and images[0].shape==images[1].shape
    torch.set_num_threads(4);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    model=WirelessVariant(SemanticVariant(args.checkpoint)).cuda()
    results=[]
    for stage,epoch,expected in [(1,1,562),(5,26,718)]:
        current=phase(stage,epoch);active=configure(model,current)
        assert len(active)==expected
        frozen=frozen_state_keys(model)
        before={name:state_hash(model.state_dict()[name]) for name in frozen}
        optimizer=torch.optim.Adam(optimizer_groups(model,stage))
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.time()
        outcome=forward_loss(model,*images,boxes,current,snr_db=10.)
        assert outcome['erasure'] is None and torch.isfinite(outcome['loss'])
        outcome['loss'].backward()
        parameters=dict(model.named_parameters())
        assert all(parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all()
                   for name in active)
        assert all(parameters[name].grad is None for name in parameters if name not in active)
        norm=torch.nn.utils.clip_grad_norm_([parameters[name] for name in active],10.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        assert all(state_hash(model.state_dict()[name])==value for name,value in before.items())
        results.append(dict(stage=stage,epoch=epoch,active_parameter_tensors=len(active),
            frozen_state_tensors=len(frozen),frozen_states_identical=True,all_active_gradients_finite=True,
            loss_type=current['loss'],untrained_sanity_loss=outcome['loss'].item(),
            gradient_norm_before_clip=norm.item(),elapsed_seconds=time.time()-started,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),accounting=outcome['accounting']))
        del optimizer,outcome
    root=Path(__file__).parent
    result=dict(state='passed',scope='two untrained native-frame GPU sanity updates; weights discarded; no formal training or AP',
        frame_id=row['frame_id'],native_shape=list(images[0].shape),image_sha256=image_hashes,
        records_sha256=hashlib.sha256(records_bytes).hexdigest(),device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,phases=results,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('model.py','radio.py','wireless.py','stages.py','probe_native.py')})
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    main()
