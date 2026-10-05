"""Real first training-update writer replay and independent coordinate checks."""
import argparse,hashlib,importlib.util,json,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'experiments/cost-field/validation_code_005/writer_replay_006.py'
spec=importlib.util.spec_from_file_location('writer006',HELPER);writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
 row=json.loads(args.metadata.read_text());assert sha(args.artifact)==row['sha256'] and args.artifact.stat().st_size==row['bytes']
 keys=('original_P2','original_P3','cropped_P2','cropped_P3','crop_offsets','image_shape','pred_boxes','pred_scores','pred_labels')
 with np.load(args.artifact,allow_pickle=False) as f:arrays={k:f[k] for k in keys}
 for key,v in arrays.items():assert row['arrays'][key]==dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())
 with tempfile.TemporaryDirectory(prefix='writer-real-probe-') as temp:text=writer.replay(ROOT,arrays,row['frame_id'],Path(temp))
 fields=[line.split() for line in text.decode().splitlines()];assert len(fields)==len(arrays['pred_boxes'])==row['prediction_count']>0
 eps=np.finfo(np.float32).eps;maximum=0.
 for i,x in enumerate(fields):
  box=arrays['pred_boxes'][i].astype(np.float64);values=np.asarray([float(v) for v in x[1:]])
  assert len(x)==16 and x[0]==('Car','Pedestrian','Cyclist')[int(arrays['pred_labels'][i])-1] and np.isfinite(values).all()
  assert x[1:3]==['-1','-1']
  dims=box[[5,4,3]];location=np.array([-box[1],-box[2]+box[5]/2,box[0]]);ry=-box[6]-np.pi/2;alpha=-np.arctan2(-box[1],box[0])+ry
  expected=np.r_[dims,location,ry]
  actual=np.asarray([float(v) for v in x[8:15]])
  error=np.abs(actual-expected);assert (error<=.5e-6+64*eps*(1+np.abs(expected))).all()
  assert abs(float(x[3])-alpha)<=.5e-4+64*eps*(1+abs(alpha))
  assert abs(float(x[15])-float(arrays['pred_scores'][i]))<=.5e-8+1e-15
  h,w=arrays['image_shape'][0];bbox=np.asarray([float(v) for v in x[4:8]])
  assert (bbox>=0).all() and (bbox[[0,2]]<=w-1).all() and (bbox[[1,3]]<=h-1).all()
  maximum=max(maximum,float(error.max()))
 result=dict(state='passed_real_training_first_update_pure_author_writer_and_coordinates',frame=row['frame_id'],boxes=len(fields),text_sha256=hashlib.sha256(text).hexdigest(),artifact_sha256=sha(args.artifact),metadata_sha256=sha(args.metadata),helper_sha256=sha(HELPER),probe_sha256=sha(__file__),max_coordinate_text_error=maximum,numpy=np.__version__,not_validation_AP=True,limitation='Independent location/dimension/rotation/alpha/score checks; bbox uses original pure writer with range checks. One engineering training frame, not full validation.')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
