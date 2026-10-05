"""Fixed F9 condition scheduling and paired-data evidence; no native model edits.

Experimental helpers remain isolated until pending detector engineering closes.
Formal runners must include this entire directory in their source manifest.
"""
import hashlib,json,random,re
from pathlib import Path
import numpy as np
import torch

STEPS=3340
PARENT_SHA='6e54ebc5fa696db26b3902cdd129ff42544635db4cfb56dfdb7f4b94841c5193'
PROTOCOL_SHA='c4bb907e4b7663bb639675c3dd218b16f1f52084d7597b6da0a3719c42a6b89a'
CONFIG_SHA='8d8053cc19d69eceb89280182b30ccb2d60d9d1a7ff6971c2343fe3bee2c247f'
SNR_SEED=1927
NOISE_SEED=1928
VERSION='F9-paired-conditions-v1'
SENSOR_KEYS={'batch_size','left_img','right_img','calib','image_shape','frame_id'}

def check(condition,message):
    if not condition:raise ValueError(message)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def fixed_schedule():
    rng=random.Random(SNR_SEED)
    return [rng.uniform(0.,20.) for _ in range(STEPS)]

def schedule_manifest():
    values=fixed_schedule()
    payload=dict(version=VERSION,steps=STEPS,SNR_seed=SNR_SEED,noise_seed=NOISE_SEED,
                 SNR_uniform_dB=[0.,20.],SNR_values=values)
    return dict(**payload,sha256=hashlib.sha256(canonical(payload)).hexdigest())

def array_evidence(value):
    if isinstance(value,torch.Tensor):
        check(not value.requires_grad,'audit array must not be trainable')
        value=value.detach().cpu().numpy()
    array=np.asarray(value)
    check(array.dtype.kind in 'bifu' and np.isfinite(array).all(),'finite numeric audit array required')
    array=np.ascontiguousarray(array)
    shape=list(array.shape);dtype=array.dtype.str
    digest=hashlib.sha256(canonical(dict(dtype=dtype,shape=shape))+b'\0'+array.tobytes()).hexdigest()
    return dict(shape=shape,dtype=dtype,sha256=digest)

def augmented_evidence(sensors,targets):
    """Hash actual separate sensor/target values, never add audit fields to model input."""
    check(set(sensors) in (SENSOR_KEYS,SENSOR_KEYS|{'random_T'}),'unexpected sensor keys')
    check(sensors['batch_size']==1 and len(sensors['frame_id'])==len(sensors['calib'])==1,'batch1 required')
    frame=str(sensors['frame_id'][0]);check(re.fullmatch(r'\d{6}',frame) is not None,'KITTI ID required')
    check(tuple(sensors['left_img'].shape)==tuple(sensors['right_img'].shape)
          and len(sensors['left_img'].shape)==4 and tuple(sensors['left_img'].shape[:2])==(1,3),'paired sensor layout')
    check(len(targets.shape)==3 and targets.shape[0]==1 and targets.shape[2]==8,'native separate GT layout')
    arrays={k:array_evidence(sensors[k]) for k in ('left_img','right_img','image_shape')}
    arrays['gt_boxes']=array_evidence(targets)
    if 'random_T' in sensors:arrays['random_T']=array_evidence(sensors['random_T'])
    calibration=vars(sensors['calib'][0])
    expected={'P2','P3','offsets','flipped'}| (set() if calibration.get('flipped') else {'R0','V2C'})
    check(set(calibration)==expected,'unexpected native calibration state')
    calib={k:array_evidence(v) for k,v in calibration.items()}
    check(calib['P2']['shape']==calib['P3']['shape']==[3,4] and calib['offsets']['shape']==[2],'native projection shape')
    payload=dict(version=VERSION,frame_id=frame,arrays=arrays,calibration=calib)
    return dict(**payload,sha256=hashlib.sha256(canonical(payload)).hexdigest())

def verify_augmented_evidence(value):
    payload={k:v for k,v in value.items() if k!='sha256'}
    check(set(payload)=={'version','frame_id','arrays','calibration'} and payload['version']==VERSION,'augmented evidence schema')
    check(hashlib.sha256(canonical(payload)).hexdigest()==value['sha256'],'augmented fingerprint digest differs')
    check(re.fullmatch(r'\d{6}',payload['frame_id']) is not None,'augmented evidence ID')
    arrays=payload['arrays'];calib=payload['calibration']
    check(set(arrays) in ({'left_img','right_img','image_shape','gt_boxes'},
                         {'left_img','right_img','image_shape','gt_boxes','random_T'}),'array coverage')
    check(set(calib) in ({'P2','P3','offsets','flipped'}, {'P2','P3','R0','V2C','offsets','flipped'}),'calibration coverage')
    for a in [*arrays.values(),*calib.values()]:
        check(set(a)=={'shape','dtype','sha256'} and re.fullmatch(r'[a-f0-9]{64}',a['sha256']) is not None,'array identity')
        check(isinstance(a['shape'],list) and all(type(v) is int and v>=0 for v in a['shape']),'array shape')
        check(np.dtype(a['dtype']).kind in 'bifu','array dtype')
    check(arrays['left_img']['shape']==arrays['right_img']['shape'] and len(arrays['left_img']['shape'])==4
          and arrays['left_img']['shape'][:2]==[1,3],'native paired layout')
    gt=arrays['gt_boxes']['shape'];check(len(gt)==3 and gt[0]==1 and gt[2]==8,'native GT evidence layout')
    check(calib['P2']['shape']==calib['P3']['shape']==[3,4] and calib['offsets']['shape']==[2],'native calib evidence layout')

def rng_digest(generator):
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()

def generator_device(device):
    """Resolve CUDA alias before strict comparison; preserve explicit indexes."""
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:
        return torch.device('cuda',torch.cuda.current_device())
    return device


class PairedChannel:
    """Bind dedicated noise RNG to existing channel without parameter/state additions.

    Native module __call__ hooks still observe the actual original input. Each
    step must be prepared/consumed once; no global RNG reset or hidden replay.
    """
    def __init__(self,link,arm,device='cuda'):
        check(arm in ('identity','awgn') and link.channel.kind==arm,'explicit matching arm required')
        check(not link.training and not link.channel.training and link.channel.pilots==8,'locked eval channel required')
        self.link=link;self.arm=arm;self.schedule=schedule_manifest();self.generator=torch.Generator(device=generator_device(device)).manual_seed(NOISE_SEED)
        self.initial_rng_state=self.generator.get_state().clone();self.initial_rng_sha256=rng_digest(self.generator)
        self.steps=0;self.calls=0;self.pending=None;self.last_evidence=None;self.closed=False
        self.original_forward=link.channel.forward;self.original_snr=link.snr_db
        link.channel.forward=self.forward

    def prepare(self,step):
        check(not self.closed and self.pending is None and type(step) is int and step==self.steps+1 and 1<=step<=STEPS,'noncontiguous/overlapping channel step')
        self.link.snr_db=self.schedule['SNR_values'][step-1]
        self.pending=dict(step=step,snr_db=self.link.snr_db,noise_rng_before_sha256=rng_digest(self.generator))
        return self.link.snr_db

    def forward(self,symbols,snr_db,generator=None):
        check(not self.closed and self.pending is not None,'channel call outside prepared F9 step')
        check(generator is None and float(snr_db)==self.pending['snr_db'],'external RNG or changed SNR')
        check(symbols.shape==(1,62400,2) and symbols.device==self.generator.device
              and symbols.dtype==torch.float32 and torch.isfinite(symbols).all(),'native symbol layout/device/dtype/finite')
        result=self.original_forward(symbols,snr_db,generator=self.generator)
        self.calls+=1;self.steps+=1
        self.last_evidence=dict(**self.pending,channel=self.arm,schedule_sha256=self.schedule['sha256'],
          noise_seed=NOISE_SEED,noise_rng_after_sha256=rng_digest(self.generator),isolated_noise_generator=True)
        check((self.last_evidence['noise_rng_before_sha256']==self.last_evidence['noise_rng_after_sha256'])==(self.arm=='identity'),'noise RNG consumption differs')
        self.pending=None
        return result

    def checkpoint_rng(self):
        check(not self.closed and self.pending is None and self.calls==self.steps,'unconsumed channel step')
        return dict(device=str(self.generator.device),seed=NOISE_SEED,initial_state=self.initial_rng_state,
                    final_state=self.generator.get_state().clone(),initial_sha256=self.initial_rng_sha256,
                    final_sha256=rng_digest(self.generator),actual_calls=self.calls)

    def close(self):
        if not self.closed:
            self.link.channel.forward=self.original_forward;self.link.snr_db=self.original_snr;self.closed=True

def replay_rng_checkpoint(saved,arm,steps,records):
    """Replay only fixed channel draws, separately from model/loss forward."""
    check(arm in ('identity','awgn') and steps in (6,STEPS),'fixed RNG replay scope')
    check(saved['seed']==NOISE_SEED and saved['actual_calls']==steps and len(records)==steps,'saved RNG scope')
    device=torch.device(saved['device']);generator=torch.Generator(device=device).manual_seed(NOISE_SEED)
    check(torch.equal(saved['initial_state'].cpu(),generator.get_state().cpu())
          and saved['initial_sha256']==rng_digest(generator),'actual dedicated generator initialization')
    for index,row in enumerate(records,1):
        e=row['channel_condition'];check(e['step']==index and e['channel']==arm,'RNG replay chronology')
        check(e['noise_rng_before_sha256']==rng_digest(generator),'RNG state before actual draw')
        if arm=='awgn':torch.randn((1,62400,2),dtype=torch.float32,device=device,generator=generator)
        check(e['noise_rng_after_sha256']==rng_digest(generator),'RNG state after actual draw')
    check(torch.equal(saved['final_state'].cpu(),generator.get_state().cpu())
          and saved['final_sha256']==rng_digest(generator),'saved final noise RNG state')
    return dict(state='passed',actual_calls=steps,device=str(device),seed=NOISE_SEED,initial_sha256=saved['initial_sha256'],final_sha256=saved['final_sha256'])

def paired_records(paths,steps):
    """All four actual sensor/GT/calibration fingerprints and noise streams must pair."""
    check(type(steps) is int and steps in (6,STEPS),'fixed paired budget')
    check(set(paths)=={'U','G','P','S'},'complete four-arm comparison required')
    rows={arm:[json.loads(line) for line in Path(path).read_text().splitlines()] for arm,path in paths.items()}
    check(all(len(values)==steps for values in rows.values()),'complete four-arm records')
    schedule=schedule_manifest();previous={};fingerprints=[]
    for index in range(steps):
        baseline=rows['U'][index]
        for arm,values in rows.items():
            row=values[index];check(row['step']==index+1 and row['epoch']==1 and row['optimization_scope']==arm,'step/arm identity')
            verify_augmented_evidence(row['augmented_evidence'])
            check(row['frame_id']==row['augmented_evidence']['frame_id'] and row['GT_shape']==row['augmented_evidence']['arrays']['gt_boxes']['shape'],'actual ID/target fingerprint')
            e=row['channel_condition']
            check(row['channel']=='awgn' and row['snr_db']==schedule['SNR_values'][index] and e['step']==index+1 and e['channel']=='awgn' and e['snr_db']==row['snr_db'],'locked SNR/channel')
            check(e['schedule_sha256']==schedule['sha256'] and e['noise_seed']==NOISE_SEED and e['isolated_noise_generator'],'isolated noise identity')
            for name in ('noise_rng_before_sha256','noise_rng_after_sha256'):
                check(re.fullmatch(r'[a-f0-9]{64}',e[name]) is not None,'noise state hash')
            if arm in previous:check(previous[arm]==e['noise_rng_before_sha256'],'continuous noise generator')
            check(e['noise_rng_before_sha256']!=e['noise_rng_after_sha256'],'AWGN consumes noise once')
            previous[arm]=e['noise_rng_after_sha256']
            check(row['augmented_evidence']==baseline['augmented_evidence'],'actual four-arm data/augmentation differ')
            check(e==baseline['channel_condition'],'actual four-arm noise/SNR differ')
        fingerprints.append(baseline['augmented_evidence']['sha256'])
    return dict(state='passed',steps=steps,arms=['U','G','P','S'],schedule_sha256=schedule['sha256'],ordered_fingerprints_sha256=hashlib.sha256(canonical(fingerprints)).hexdigest(),final_noise_rng_sha256=previous,
      limitations='Producer fingerprints checked across all raw rows; separate full state/native/AP audits are required.')
