"""All six actual transferred SRCNN engineering byte files and native lineage."""
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib

ROOT=Path(__file__).resolve().parents[2]
PREFIX='srcnn-source-cache-engineering-001'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    output=ROOT/'data/provenance'/(PREFIX+'-wire-local-verification.json');assert not output.exists()
    encode_path=ROOT/'data/runs'/(PREFIX+'-encode.json');audit_path=ROOT/'data/provenance'/(PREFIX+'-wire-audit.json')
    encode,audit=read(encode_path),read(audit_path)
    assert encode['state']=='finished_all_three_source_wire_conditions'
    assert audit['state']=='passed_all6_native_SRCNN_engineering_wires_actual_encoder_terminal' and audit['actual_encoder_terminal']
    assert audit['encoder_pid']==encode['pid'] and audit['encoder_manifest_sha256']==sha(encode_path)
    assert audit['pairs']==encode['pairs']==6 and encode['frame_ids']==['000000','000003']
    assert audit['unique_source_views']==4 and not audit['GPU_used'] and not encode['GPU_used']
    assert encode['sources']==audit['sources'] and encode['dependencies']==audit['dependencies']
    for group in audit['sources'].values():
        for path,digest in group.items():assert sha(ROOT/path)==digest
    assert audit['audit_helper_sha256']==sha(ROOT/'data/provenance/audit-srcnn-engineering-wires-001.py')
    assert sha(ROOT/'data/provenance/srcnn-kitti-engineering-seed17-002-closure.json')==audit['dependencies']['training_closure_sha256']
    assert sha(ROOT/'data/provenance/srcnn-kitti-engineering-seed17-002-local-verification.json')==audit['dependencies']['training_local_proof_sha256']
    assert sha(ROOT/'data/provenance/srcnn-float-reception-transferred-verification-001.json')==audit['dependencies']['float_CPU_local_proof_sha256']
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in (encode_path,audit_path)};total_values=0
    assert len(audit['rows'])==len(audit['wire_files_sha256'])==6
    for rate in (10,30,50):
        parent=encode['conditions'][str(rate)];native=[row for row in audit['rows'] if row['rate']==rate]
        assert [row['frame_id'] for row in native]==['000000','000003']
        raw_total=wire_total=0
        for row,n in zip(parent['rows'],native):
            assert row['frame_id']==n['frame_id'] and row['wire_path']==n['wire_path']
            path=Path(row['wire_path']);expected=Path('/mnt/d/paper6/runs')/PREFIX/('cr'+str(rate))/'wire'/(row['frame_id']+'.p6sr')
            assert path==expected
            local=ROOT/'data/engineering'/(PREFIX+'-wire-transfer')/str(path).lstrip('/')
            blob=local.read_bytes();assert sha(local)==row['wire_sha256']==n['wire_sha256']==audit['wire_files_sha256'][str(path)]
            magic,version,algorithm,r,reserved,h,w,lh,lw,nl,nr,crc=struct.unpack('>4sBBBBIIHHIII',blob[:32])
            assert (magic,version,algorithm,r,reserved)==(b'P6SR',1,1,rate,0)
            assert (lh,lw)==(math.floor(h/math.sqrt(rate)),math.floor(w/math.sqrt(rate))) and nl==nr==lh*lw*3
            assert len(blob)==32+nl+nr and zlib.crc32(blob[32:])==crc
            meta=row['source_metadata'];assert meta==n['source_metadata']
            assert meta['native_hw']==[h,w] and meta['low_hw']==[lh,lw] and meta['payload_bytes']==[nl,nr]
            assert meta['source_bytes']==len(blob) and meta['framing_bytes']==32 and meta['source_bits']==8*len(blob)
            assert meta['raw_RGB8_bits']==8*h*w*6 and meta['actual_raw_to_wire_ratio']==h*w*6/len(blob)
            assert n['source_PNGs']==[row['source_left'],row['source_right']]
            for sensor in n['source_PNGs']:
                assert sensor['RGB8']['dtype']=='|u1' and sensor['RGB8']['shape']==[h,w,3]
            raw_total+=h*w*6;wire_total+=len(blob);total_values+=h*w*6
            artifacts[str(local.relative_to(ROOT))]=sha(local)
        summary=dict(raw_RGB8_bytes=raw_total,wire_bytes=wire_total,actual_pooled_raw_to_wire_ratio=raw_total/wire_total)
        assert summary==audit['summaries'][str(rate)]
        assert all(parent[key]==value for key,value in summary.items())
    assert len(artifacts)==8 and total_values==audit['RGB_values_checked_across_three_rates']==16535340
    result=dict(state='passed_all6_actual_transferred_SRCNN_engineering_wire_files_and8_artifacts',checked_unix=time.time(),
                pairs=6,unique_source_views=4,artifacts_verified=8,artifacts_sha256=artifacts,
                sources=audit['sources'],dependencies=audit['dependencies'],summaries=audit['summaries'],
                native_wire_audit_sha256=sha(audit_path),verifier_sha256=sha(__file__),
                limitation='Every actual wire/framing/CRC and full metadata lineage checked locally; fresh source PNG/RGB replay native only, no GPU SRCNN receiver/cache, formal result, quality, radio or AP')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],artifacts=8)))


if __name__=='__main__':main()
