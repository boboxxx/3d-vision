"""Fresh CPU actual PNG/literal-wire audit, separate from future GPU reception."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'experiments/original-paper-scenarios/srcnn-native-cache-code'
sys.path.insert(0,str(CODE))
import common as c
import audit as oracle


def main():
    assert ROOT == Path('/home/sheng/paper6') and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    source=c.CPU_gate(); _, dependencies=c.gate('engineering'); prefix=c.prefix('engineering')
    manifest=ROOT/'data/runs'/(prefix+'-encode.json'); output=ROOT/'data/provenance'/(prefix+'-wire-audit.json')
    assert not output.exists(); run=c.read(manifest)
    assert run['state']=='finished_all_three_source_wire_conditions' and run['sources']==source and run['dependencies']==dependencies
    assert run['scope']=='engineering' and run['frame_ids']==['000000','000003'] and run['pairs']==6
    process=subprocess.run(['ps','-p',str(run['pid']),'-o','pid,stat,args'],capture_output=True,text=True)
    assert process.returncode==1, 'Source encoder must actually exit'
    assert not run['GPU_used'] and run['no_GT_calibration_LiDAR_or_detector']
    inputs=[c.DATA/'training'/camera/(frame+'.png') for frame in run['frame_ids'] for camera in ('image_2','image_3')]
    inputs += [Path(row['wire_path']) for condition in run['conditions'].values() for row in condition['rows']]
    control={'active':False};sys.addaudithook(c.sensor_guard(inputs,control));control['active']=True
    rows=[]; RGB_values=0; summaries={}
    for rate in (10,30,50):
        condition=run['conditions'][str(rate)]; raw_total=wire_total=0
        assert [row['frame_id'] for row in condition['rows']]==run['frame_ids']
        for row in condition['rows']:
            frame=row['frame_id'];left,l=c.png(c.DATA/'training/image_2'/(frame+'.png'));right,r=c.png(c.DATA/'training/image_3'/(frame+'.png'))
            assert l==row['source_left'] and r==row['source_right']
            path=Path(row['wire_path']);assert path==c.NATIVE/prefix/('cr'+str(rate))/'wire'/(frame+'.p6sr')
            wire=path.read_bytes();assert c.r.sha_bytes(wire)==row['wire_sha256'] and wire==oracle.independent_wire(left,right,rate)
            low,hw,received_rate=oracle.independent_unpack(wire);assert hw==left.shape[:2]==right.shape[:2] and received_rate==rate
            meta=row['source_metadata'];assert meta['native_hw']==list(hw) and meta['low_hw']==list(low[0].shape[:2])
            assert meta['source_bytes']==len(wire) and meta['source_bits']==8*len(wire) and meta['framing_bytes']==32
            assert meta['payload_bytes']==[low[0].size,low[1].size] and meta['raw_RGB8_bits']==8*(left.size+right.size)
            assert meta['actual_raw_to_wire_ratio']==(left.size+right.size)/len(wire)
            assert meta['source_only_PHY_uses'] is None and meta['source_only_PHY_energy'] is None
            raw_total+=left.size+right.size;wire_total+=len(wire);RGB_values+=left.size+right.size
            rows.append(dict(rate=rate,frame_id=frame,wire_path=str(path),wire_sha256=row['wire_sha256'],source_metadata=meta,
                             source_PNGs=[l,r]))
        assert raw_total==condition['raw_RGB8_bytes'] and wire_total==condition['wire_bytes']
        assert raw_total/wire_total==condition['actual_pooled_raw_to_wire_ratio']
        summaries[str(rate)]=dict(raw_RGB8_bytes=raw_total,wire_bytes=wire_total,actual_pooled_raw_to_wire_ratio=raw_total/wire_total)
    control['active']=False;assert c.sources()==source
    result=dict(state='passed_all6_native_SRCNN_engineering_wires_actual_encoder_terminal',checked_unix=time.time(),
                actual_encoder_terminal=True,encoder_pid=run['pid'],encoder_manifest_sha256=c.sha(manifest),sources=source,
                dependencies=dependencies,pairs=6,unique_source_views=4,RGB_values_checked_across_three_rates=RGB_values,
                wire_files_sha256={row['wire_path']:row['wire_sha256'] for row in rows},rows=rows,summaries=summaries,
                audit_helper_sha256=c.sha(__file__),GPU_used=False,
                limitation='Fresh full PNG/RGB and independently literal six actual wire files, same Pillow resize backend; no GPU SRCNN reception, quality, radio or AP')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ('state','pairs','RGB_values_checked_across_three_rates','summaries')}),flush=True)


if __name__=='__main__':main()
