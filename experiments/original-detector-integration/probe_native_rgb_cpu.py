"""One true native RGB frame through full original variant; engineering only."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ORIGINAL=ROOT/'reproduction/cao2025'
sys.path[:0]=[str(HERE),str(ORIGINAL),str(ROOT/'src')]
from adapter import transmit_rgb,stereo_rcnn_preprocess
from data import StereoRGB
from wireless import WirelessVariant
import radio
from geocomm.evidence import sha256,source_identity


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def hashes(model):
    result={}
    for name,value in model.state_dict().items():
        value=value.detach().cpu().contiguous()
        if not torch.isfinite(value).all():raise RuntimeError('nonfinite original model state')
        result[name]=hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.numpy().tobytes()).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('preserve evidence; unique output required')
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(mem['MemAvailable'].strip().split()[0])*1024
    if available<8*2**30:raise RuntimeError('need8GiB available host memory while GPUtraining continues')
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    source=source_identity(ROOT,['src','scripts','configs','pyproject.toml'])
    original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    if len(original)!=21 or any(sha256(ORIGINAL/name)!=value for name,value in original.items()):
        raise RuntimeError('original isolated source differs')
    initializer=Path('/mnt/d/paper6/runs/cao2025-full-native-seed17-001-stage1/initialization.pth')
    initial_sha='f4d0d5d5bcc203ee5fd0554a0e691b2a530c56cb55d48fe21b6fc0f97331d3cb'
    if sha256(initializer)!=initial_sha:raise RuntimeError('fixed freshoriginal initialization differs')
    spynet=Path('/mnt/d/paper6/checkpoints/spynet_sintel_final-3d2a1287.pth')
    if sha256(spynet)!='3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714':
        raise RuntimeError('fixed SpyNet identity differs')
    semantic=load_file('paper6_cao2025_semantic',ORIGINAL/'model.py')
    # No bare original model import: the native Stereo-RCNN model namespace remains available.
    if 'model' in sys.modules:raise RuntimeError('original import occupied detector model namespace')
    model=WirelessVariant(semantic.SemanticVariant(spynet)).cpu().eval()
    initial=torch.load(initializer,map_location='cpu',weights_only=False)['model_state']
    actual=model.state_dict()
    if len(initial)!=768 or set(initial)!=set(actual) or any(v.shape!=actual[k].shape or v.dtype!=actual[k].dtype for k,v in initial.items()):
        raise RuntimeError('complete768-state nativearchitecture differs')
    model.load_state_dict(initial,strict=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in initial.items()):raise RuntimeError('initial values differ')
    before=hashes(model);del initial,actual
    cache=ROOT/'data/engineering/cao2025-roi-holdout-001.jsonl'
    audit=ROOT/'data/engineering/cao2025-roi-holdout-audit-001.json';fold=ROOT/'data/internal-tuning-fold-001.json'
    dataset=StereoRGB('/mnt/d/paper6/data/kitti',cache,audit,fold,'geocomm_tune_holdout')
    frame=dataset[0]
    if frame['frame_id']!='000036' or set(frame)!={'frame_id','left','right','boxes'}:
        raise RuntimeError('fixed sensor-only engineering frame differs')
    model.forward=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('broad model.forward forbidden'))
    model.semantic_mse=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('cleansemanticMSE forbidden'))
    events=[];current={'clean':None,'received':None}
    encode=model.semantic.encode;transmit=model.transmit_payload;receive=model.receive_payload;decode=model.semantic.decode
    def traced_encode(*args):
        events.append('sender_encode');payload=encode(*args);current['clean']=payload;return payload
    def traced_transmit(payload,*args):
        if payload is not current['clean']:raise RuntimeError('sender payload boundary differs')
        events.append('wire_transmit');return transmit(payload,*args)
    def traced_receive(observation):
        if not isinstance(observation,radio.Observation):raise RuntimeError('receiver requires onlyobservation')
        events.append('received_payload');payload=receive(observation);current['received']=payload;return payload
    def traced_decode(payload):
        if payload is not current['received'] or payload is current['clean']:raise RuntimeError('clean semantic bypass')
        events.append('received_RGB_decode');return decode(payload)
    model.semantic.encode=traced_encode;model.transmit_payload=traced_transmit;model.receive_payload=traced_receive;model.semantic.decode=traced_decode
    # The actual detector package is importable alongside the original semantic class.
    sys.path.insert(0,str(ROOT/'third_party/Stereo-RCNN/lib'))
    from model.utils.blob import prep_im_for_blob
    from model.utils.config import cfg
    rows=[]
    with torch.no_grad():
        for channel in ('identity','awgn'):
            events.clear();current.update(clean=None,received=None);start=time.time()
            result=transmit_rgb(model,radio,frame['left'],frame['right'],frame['boxes'],channel,10.,torch.Generator().manual_seed(17))
            expected=['sender_encode','wire_transmit','received_payload']+([] if result['erasure'] else ['received_RGB_decode'])
            if events!=expected:raise RuntimeError('actual native receiver chronology differs')
            row=dict(channel=channel,events=list(events),accounting=result['accounting'],erasure=result['erasure'],decoder_range=result['decoder_range'])
            if result['outputs'] is not None:
                images,info,shape=stereo_rcnn_preprocess(result['outputs'],prep_im_for_blob,cfg.PIXEL_MEANS,cfg.TRAIN.SCALES[0],cfg.TRAIN.MAX_SIZE)
                row.update(native_RGB_shape=list(result['outputs'][0].shape),detector_preprocess_shapes=[list(v.shape) for v in images],
                    detector_info=info.tolist(),original_detector_shape=list(shape),
                    processed_received_tensor_sha256=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in images])
            if hashes(model)!=before:raise RuntimeError('native codec inference mutated states')
            row['CPU_seconds_engineering_not_latency_benchmark']=time.time()-start;rows.append(row)
            print(json.dumps({'channel':channel,'erasure':row['erasure'],'total_uses':row['accounting']['total_uses']}),flush=True)
            del result,current['clean'],current['received'];current.update(clean=None,received=None)
    if source_identity(ROOT,['src','scripts','configs','pyproject.toml'])!=source:
        raise RuntimeError('F6 frozenroot sources changed')
    if any(sha256(ORIGINAL/name)!=value for name,value in original.items()):raise RuntimeError('original sources changed')
    output=dict(state='passed',evidence_type='full_native_original_RGB_CPU_engineering_only',seed=17,device='cpu',
        model_states_identical=768,all768_state_hashes=before,initialization_sha256=initial_sha,updates=0,
        frame_id=frame['frame_id'],native_sensor_shape=list(frame['left'].shape),conditions=rows,
        ROI_records_sha256=sha256(cache),ROI_audit_sha256=sha256(audit),fold_sha256=sha256(fold),
        available_host_memory_before_bytes=available,project_source_unchanged=source,
        original21_sources_unchanged=original,adapter_sources=source_identity(HERE,['adapter.py','probe_native_rgb_cpu.py']),
        protocol_sha256=sha256(HERE/'native-CPU-engineering.md'),
        limitations='Fresh untrained original initializer; nativecodec+officialpreprocessing only, no3Ddetector/GPU/AP/latency or learnedbaseline claim')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['state','updates','model_states_identical','frame_id','native_sensor_shape']}))

if __name__=='__main__':main()
