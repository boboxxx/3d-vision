"""No fitting until full native observation audit and actual process closure pass."""
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
from risk_stats import common_rows,fit_compare
ROOT=Path(__file__).resolve().parents[3];PREFIX='geometry-risk-pilot-001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=ROOT/'data/runs'/f'{PREFIX}.json';run=json.loads(path.read_text())
    auditpath=ROOT/'data/provenance'/f'{PREFIX}-native-audit.json';audit=json.loads(auditpath.read_text())
    assert run['state']=='finished64_native_observations_independent_audit_pending'
    assert audit['state']=='passed_full64_independent_native_audit' and audit['manifest_sha256']==sha(path) and audit['native_rows']==8320
    assert audit['actual_terminal_PID']==run['pid']
    protocol=ROOT/'experiments/geometry-risk/pilot-execution-001.md';assert sha(protocol)==run['protocol_sha256']
    lock=json.loads((ROOT/'experiments/geometry-risk/native-engineering-inputs-001.json').read_text())
    ids=lock['future_fit_ids']+lock['future_out_of_fit_ids']
    assert run['fit_ids']==lock['future_fit_ids'] and run['out_of_fit_ids']==lock['future_out_of_fit_ids']
    reports=[];reporthashes={}
    for frame in ids:
        p=Path(run['frames'][frame]['report_path']);assert sha(p)==run['frames'][frame]['report_sha256']==audit['summaries'][frame]['frame_report_sha256']
        reporthashes[frame]=sha(p);reports.append(common_rows(json.loads(p.read_text())))
    result,indices=fit_compare(reports)
    output=ROOT/'data/analysis'/f'{PREFIX}-ridge-001.json';indicespath=ROOT/'data/analysis'/f'{PREFIX}-bootstrap-001.npy'
    assert not output.exists() and not indicespath.exists();output.parent.mkdir(parents=True,exist_ok=True)
    np.save(indicespath,indices,allow_pickle=False)
    result.update(scope='training_only_auxiliary_out_of_fit_loss_damage_not_mainval_AP_or_inference_gradient_availability',
                  manifest_sha256=sha(path),native_audit_sha256=sha(auditpath),protocol_sha256=sha(protocol),
                  report_sha256=reporthashes,bootstrap_indices_path=str(indicespath),bootstrap_indices_sha256=sha(indicespath),
                  analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  numpy=np.__version__,finished_unix=time.time())
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'state':result['state'],'models':{k:v['summary'] for k,v in result['models'].items()}}))


if __name__=='__main__':main()
