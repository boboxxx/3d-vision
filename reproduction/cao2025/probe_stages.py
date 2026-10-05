"""Real full-model phase/optimizer/frozen-state engineering; no reusable training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from model import SemanticVariant
from wireless import WirelessVariant
from stages import EPOCHS, configure, forward_loss, frozen_state_keys, optimizer_groups, phase


def state_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    started=time.time();torch.set_num_threads(4);torch.manual_seed(17)
    model=WirelessVariant(SemanticVariant(args.checkpoint))
    # Separate synthetic models/updates are never saved as formal initialization.
    left,right=torch.rand(1,3,197,203),torch.rand(1,3,197,203)
    boxes=([dict(xyxy=[40,30,160,150],confidence=.8)],
           [dict(xyxy=[35,35,155,155],confidence=.75)])
    expected={(1,1):562,(1,3):622,(2,1):16,(3,1):12,(4,1):638,(4,6):638,(5,1):80,(5,26):718}
    records=[]
    optimizers={}
    for (stage,epoch),expected_count in expected.items():
        current=phase(stage,epoch);active=configure(model,current)
        assert len(active)==expected_count,(stage,epoch,len(active),expected_count)
        if stage not in optimizers:
            optimizers[stage]=torch.optim.Adam(optimizer_groups(model,stage))
        optimizer=optimizers[stage]
        for group in optimizer.param_groups:
            if group['component'] in current['rates']:
                assert group['lr']==current['rates'][group['component']]
        frozen=frozen_state_keys(model)
        before={name:state_hash(model.state_dict()[name]) for name in frozen}
        model.zero_grad(set_to_none=True)
        outcome=forward_loss(model,left,right,boxes,current,snr_db=10.)
        assert outcome['erasure'] is None and torch.isfinite(outcome['loss'])
        outcome['loss'].backward()
        parameters=dict(model.named_parameters())
        assert all(parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all()
                   for name in active),(stage,epoch,'missing active gradient')
        assert all(parameters[name].grad is None for name in parameters if name not in active)
        norm=torch.nn.utils.clip_grad_norm_([parameters[name] for name in active],10.,error_if_nonfinite=True)
        optimizer.step()
        assert all(state_hash(model.state_dict()[name])==value for name,value in before.items())
        if stage==1 and epoch==1:
            assert not any(name.startswith('semantic.global_decoder.flow.') and
                parameter in optimizer.state for name,parameter in model.named_parameters())
        if stage==5 and epoch==1:
            assert not any(name.startswith('semantic.') and parameter in optimizer.state
                           for name,parameter in model.named_parameters())
        records.append(dict(stage=stage,epoch=epoch,loss_type=current['loss'],rates=current['rates'],
            active_parameter_tensors=len(active),frozen_state_tensors=len(frozen),
            frozen_states_identical=True,active_gradients_finite=True,
            synthetic_untrained_loss=outcome['loss'].item(),gradient_norm_before_clip=norm.item()))
    root=Path(__file__).parent
    result=dict(state='passed',scope='synthetic full-model phase/update engineering only; no formal weights or AP',
        stage_epochs=EPOCHS,total_formal_epochs=sum(EPOCHS.values()),torch_version=torch.__version__,
        device='cpu',phases=records,source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('model.py','radio.py','wireless.py','stages.py','probe_stages.py')},
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        elapsed_seconds=time.time()-started)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    main()
