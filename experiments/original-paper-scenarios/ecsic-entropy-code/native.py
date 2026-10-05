"""Encode sealed fixtures once; a fresh neural receiver reads only container bytes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import codec as c

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/ecsic-recovery-code'))
import recover as a

STAGE_A = Path('/mnt/d/paper6/runs/ecsic-recovery-stageA-001')


def save_arrays_without_readback(path, arrays):
    buffer=io.BytesIO()
    np.savez(buffer,**arrays)
    blob=buffer.getvalue()
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def source_identity():
    return {str(p.relative_to(ROOT)):a.sha(p) for p in c.HERE.iterdir() if p.is_file()}


def guards():
    audit=json.loads((ROOT/'data/provenance/ecsic-recovery-stageA-001-audit.json').read_text())
    a.check(audit['state']=='passed' and audit['actual_native_terminal'], 'stage A audit gate')
    manifest=json.loads((STAGE_A/'manifest.json').read_text())
    a.check(a.sha(STAGE_A/'manifest.json')==audit['manifest_sha256'], 'sealed stage A')
    for name in ('ecsic-entropy-local-CPU-001.json','ecsic-entropy-sheng-CPU-001.json'):
        gate=json.loads((ROOT/'data/engineering'/name).read_text())
        a.check(gate['state']=='passed' and len(gate['rejected_cases'])==204, 'CPU gate')
        a.check(all(a.sha(c.HERE/key)==value for key,value in gate['source_sha256'].items()), 'CPU source identity')
        a.check(gate['CDF_sha256']==c.tables()[1], 'frozen CDF gate')
    return audit,manifest


@torch.inference_mode()
def receiver(container, directory):
    # Read and validate bounded lengths/CRC/identities before constructing a model.
    def deny_inputs(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0]); mode=args[1]; flags=args[2]
            reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE == os.O_RDONLY)
            a.check('/data/kitti/' not in name and not (reading and name.endswith('.npz')), 'receiver tried source/NPZ input')
    sys.addaudithook(deny_inputs)
    cdfs,cdf_sha=c.tables()
    a.check(container.stat().st_size <= c.LIMIT, 'bounded container file')
    blob=container.read_bytes(); parsed=c.unpack(blob,cdf_sha)
    model,utils,meta=a.load(); before=a.state(model)
    counts,_=a.hooks(model,reject_encoder=True)
    h,w=parsed['padded_hw']; pos=utils.get_positional_fourier_encoding(h,w).unsqueeze(0)
    values={}; decoded_symbols={}; indices={}
    def pull(index,loc,scale):
        name=c.ORDER[index]; shape=c.shape_of(index,[h,w])
        residual=c.decode_stream(*parsed['streams'][name],a.arr(scale),shape,cdfs)
        decoded_symbols[name]=residual
        indices[name]=a.describe(c.scale_ids(a.arr(scale),shape))
        values[name+'_symbols']=torch.from_numpy(residual)
        values[name+'_loc']=loc; values[name+'_scale']=scale
        values[name+'_hat']=values[name+'_symbols'].float()+loc
        return values[name+'_hat']
    zl=pull(0,model.zl_loc,model.zl_scale)
    zr=pull(1,*model.zr_entropy(zl))
    left,right,_,_=model.HD(zl,zr,pos=pos,return_attn=True)
    left_loc,left_scale=model.hd_out_left(left).chunk(2,1)
    yl=pull(2,left_loc,left_scale)
    right_loc,right_scale,_,_=model.yr_entropy(yl,model.hd_out_right(right),pos=pos)
    yr=pull(3,right_loc,right_scale)
    values['pred_left'],values['pred_right'],_,_=model.D(yl,yr,pos=pos,return_attn=True)
    arrays={k:a.arr(v) for k,v in values.items()}
    a.check(all(np.isfinite(v).all() for v in arrays.values()), 'finite receiver arrays')
    a.check(a.state(model)==before and all(p.grad is None for p in model.parameters()), 'receiver state/gradient immutability')
    a.check(counts==dict(E=0,HE=0,HD=1,D=1), 'receiver source encoder barrier')
    output_sha=save_arrays_without_readback(directory/'reconstructed.npz',arrays)
    meta.update(state='passed',container_sha256=hashlib.sha256(blob).hexdigest(),container_bytes=len(blob),
                public_dimensions={k:parsed[k] for k in ('original_hw','padded_hw')},calls=counts,
                full_states_before=before,full_states_after=a.state(model),no_parameter_gradients=True,
                CDF_sha256=cdf_sha,derived_scale_indices=indices,arrays={k:a.describe(v) for k,v in arrays.items()},
                sources_sha256=source_identity(),source_NPZ_read_barrier=True,
                output_sha256=output_sha)
    a.save_json(directory/'receiver.json',meta)


def run(directory):
    directory.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',pid=os.getpid(),started_unix=time.time(),cases={})
    a.save_json(directory/'manifest.json',report)
    try:
        audit,manifest=guards(); cdfs,cdf_sha=c.tables(); sources=source_identity()
        report.update(sources_sha256=sources,CDF_sha256=cdf_sha,stage_A_audit_sha256=a.sha(ROOT/'data/provenance/ecsic-recovery-stageA-001-audit.json'),
                      protocol_sha256=a.sha(ROOT/'experiments/original-paper-scenarios/ecsic-entropy-protocol-001.md'))
        for case in ('synthetic32x64','training000000'):
            original=STAGE_A/case; target=directory/case; target.mkdir()
            locked=manifest['cases'][case]
            a.check(all(a.sha(original/k)==v for k,v in locked['files'].items()), 'sealed case identity')
            public=json.loads((original/'public.json').read_text())
            streams={}; details={}
            print(json.dumps(dict(case=case,stage='entropy_encoding')),flush=True)
            with np.load(original/'symbols.npz',allow_pickle=False) as residual, np.load(original/'reference.npz',allow_pickle=False) as reference:
                for name in c.ORDER:
                    q=residual[name]; scale=reference[name+'_scale']
                    rb,eb=c.encode_stream(q,scale,cdfs); streams[name]=(rb,eb)
                    ids=c.scale_ids(scale,q.shape)
                    values=q.ravel(); codes=np.where((values < -255)|(values > 255),511,values+255)
                    freq=np.asarray([cdfs[int(t)][int(k)+1]-cdfs[int(t)][int(k)] for t,k in zip(ids,codes)],dtype=np.float64)
                    ideal=float(-np.log2(freq/c.TOTAL).sum())
                    details[name]=dict(symbols=int(q.size),rans_bytes=len(rb),escape_bytes=len(eb),
                                       escape_symbols=int(np.count_nonzero(codes==511)),finite_CDF_ideal_bits=ideal,
                                       derived_scale_indices=a.describe(ids))
                blob=c.pack(public['original_hw'],public['padded_hw'],streams,cdf_sha)
                payload=target/'source.p6ec';payload.write_bytes(blob)
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--receiver','--container',str(payload),'--directory',str(target)],
                                      capture_output=True,text=True,env=os.environ.copy())
                (target/'receiver.log').write_text(result.stdout+result.stderr)
                a.check(result.returncode==0,'byte-only receiver failure: '+case)
                rec=json.loads((target/'receiver.json').read_text())
                a.check(rec['state']=='passed' and rec['container_sha256']==a.sha(payload),'receiver evidence identity')
                a.check(rec['derived_scale_indices']=={k:v['derived_scale_indices'] for k,v in details.items()},'receiver independently derived scales')
                with np.load(target/'reconstructed.npz',allow_pickle=False) as recovered:
                    a.check(set(recovered.files)==set(reference.files),'all decoded arrays')
                    equal={k:bool(np.array_equal(reference[k],recovered[k])) for k in reference.files}
                    a.check(all(equal.values()),'byte receiver exact equality failed: '+str(equal))
            estimated=sum(v['finite_CDF_ideal_bits']+8*v['escape_bytes'] for v in details.values())
            actual=len(blob)*8
            item=dict(state='passed',container_bytes=len(blob),container_bits=actual,streams=details,
                      header_CRC_bytes=166,estimated_CDF_plus_literal_bits=estimated,
                      actual_minus_estimate_bits=actual-estimated,exact_arrays=equal,
                      original_hw=public['original_hw'],padded_hw=public['padded_hw'],
                      files={p.name:a.sha(p) for p in target.iterdir()})
            report['cases'][case]=item;a.save_json(directory/'manifest.json',report)
            print(json.dumps(dict(case=case,state='passed',container_bytes=len(blob),exact_arrays=len(equal))),flush=True)
        a.check(source_identity()==sources,'entropy source mutation')
        report.update(state='passed',finished_unix=time.time(),actual_entropy_source_bytes=True,
                      physical_channel_executed=False,KITTI_AP_measured=False,
                      limitation='Declared finite-CDF source codec; probabilities not billed as physical uses. Fixed two engineering inputs only.')
    except Exception:
        report.update(state='failed',failed_unix=time.time(),traceback=traceback.format_exc());raise
    finally:
        a.save_json(directory/'manifest.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--receiver',action='store_true');p.add_argument('--container',type=Path)
    args=p.parse_args()
    if args.receiver:receiver(args.container,args.directory)
    else:run(args.directory)
