"""Descriptive Monte Carlo uncertainty and linear residuals; no new fit/test."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-antithetic-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    auditpath=ROOT/'data/provenance'/f'{PREFIX}-transferred-verification-001.json'
    audit=json.loads(auditpath.read_text());assert audit['state']=='passed_complete_transferred_raw_rows_and_reports'
    reports=ROOT/'data/engineering'/f'{PREFIX}-records';rows=[]
    for path in sorted(reports.glob('*.json')):
        assert sha(path)==audit['artifacts_sha256'][str(path.relative_to(ROOT))]
        r=json.loads(path.read_text());q=np.asarray([d['symmetric'] for d in r['damage']]);s=np.asarray([d['antisymmetric'] for d in r['damage']]);linear=np.asarray([d['linear'] for d in r['damage']])
        assert q.shape==s.shape==linear.shape==(32,16)
        means=q.mean(1);se=np.sqrt(q.var(1,ddof=1)/16);residual=s-linear
        denominator=float(np.square(s).mean())
        rows.append(dict(frame_id=r['frame_id'],signed_group_means=means.tolist(),group_mean_MC_standard_errors=se.tolist(),
          symmetric_group_variances=q.var(1,ddof=1).tolist(),antisymmetric_group_variances=s.var(1,ddof=1).tolist(),
          linear_group_variances=linear.var(1,ddof=1).tolist(),positive_group_means=int((means>0).sum()),
          antisymmetric_linear_relative_RMS=float(np.sqrt(np.square(residual).mean()/denominator)) if denominator else None,
          mean_signed_group_increment=float(means.mean()),mean_group_MC_standard_error=float(se.mean()),
          report_sha256=sha(path)))
    assert len(rows)==8
    out=ROOT/'data/analysis'/f'{PREFIX}-descriptive-001.json';assert not out.exists()
    result=dict(frames=rows,source_sha256=sha(Path(__file__)),transferred_audit_sha256=sha(auditpath),
      scope='Descriptive summaries of prelocked paired quantities; MC standard errors use 16 independent pairs per group. Relative RMS compares actual s with g dot e across all 512 pairs within a frame; not a calibration or AP metric. No cross-frame significance test or selection.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
