"""Verify all transferred metadata and actual packets; native arrays stay on sheng."""
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    output=ROOT/'data/provenance/ecsic-transferred-verification-001.json';assert not output.exists()
    results={};files={}
    for prefix in ('ecsic-recovery-stageA-001','ecsic-entropy-stageB-002'):
        base=ROOT/'data/engineering'/prefix;manifest=read(base/'manifest.json')
        audit=read(ROOT/'data/provenance'/f'{prefix}-audit.json')
        assert manifest['state']==audit['state']=='passed' and audit['actual_native_terminal']
        assert sha(base/'manifest.json')==audit['manifest_sha256']
        assert sha(ROOT/'data/provenance'/('verify-ecsic-recovery-001.py' if 'stageA' in prefix else 'verify-ecsic-entropy-002.py'))==audit['verifier_sha256']
        files[str((base/'manifest.json').relative_to(ROOT))]=sha(base/'manifest.json')
        for case in ('synthetic32x64','training000000'):
            item=manifest['cases'][case];receiver=read(base/case/'receiver.json')
            assert item['state']==receiver['state']=='passed'
            assert len(item['exact_arrays'])==18 and all(item['exact_arrays'].values())
            assert len(receiver['full_states_before'])==225 and receiver['full_states_before']==receiver['full_states_after']
            assert receiver['calls']==dict(E=0,HE=0,HD=1,D=1)
            for name,value in item['files'].items():
                if name.endswith('.npz'):continue
                file=base/case/name;assert sha(file)==value;files[str(file.relative_to(ROOT))]=value
            if 'stageB' in prefix:
                p=base/case/'source.p6ec';blob=p.read_bytes()
                assert len(blob)==item['container_bytes']==audit['cases'][case]['container_bytes']
                assert sha(p)==receiver['container_sha256']==audit['cases'][case]['payload_sha256']
                assert zlib.crc32(blob[:-4])==int.from_bytes(blob[-4:],'big')
        results[prefix]=dict(manifest_sha256=sha(base/'manifest.json'),server_audit_sha256=sha(ROOT/'data/provenance'/f'{prefix}-audit.json'))
    b=read(ROOT/'data/engineering/ecsic-entropy-stageB-002/manifest.json')
    assert all(sha(ROOT/k)==v for k,v in b['sources_sha256'].items())
    local=read(ROOT/'data/engineering/ecsic-entropy-local-CPU-001.json');server=read(ROOT/'data/engineering/ecsic-entropy-sheng-CPU-001.json')
    for key in ('state','checks','rejected_cases','CDF_sha256','reference_fixtures','source_sha256'):
        assert local[key]==server[key]
    assert local['state']=='passed' and len(local['rejected_cases'])==204
    output.write_text(json.dumps(dict(state='passed',verifier_sha256=sha(Path(__file__)),runs=results,
                        files=files,scope='Complete transferred JSON/logs/real P6EC bytes and source identities; full NPZ/checkpoint/C++ native-stream audit ran on sheng.'),indent=2)+'\n')
    print(json.dumps(dict(state='passed',transferred_files=len(files),native_containers_bytes=[b['cases'][k]['container_bytes'] for k in ('synthetic32x64','training000000')])))


if __name__=='__main__':main()
