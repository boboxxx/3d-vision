"""Supplement source review: every log-scale transition and representable half ties."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'experiments/original-paper-scenarios/ecsic-entropy-code'
sys.path.insert(0,str(CODE))
import codec


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    assert not args.output.exists()
    with localcontext() as context:
        context.prec=60
        lower=Decimal('0.1').ln();span=Decimal(2560).ln()
        midpoints=[float((lower+(Decimal(i)+Decimal('0.5'))*span/Decimal(255)).exp()) for i in range(255)]
    left=np.asarray(midpoints)*(1-1e-9);right=np.asarray(midpoints)*(1+1e-9)
    assert np.array_equal(codec.scale_ids(left,(255,)),np.arange(255))
    assert np.array_equal(codec.scale_ids(right,(255,)),np.arange(1,256))
    ties=[];unrepresentable=[]
    for i,middle in enumerate(midpoints):
        candidates=[middle];a=b=middle
        for _ in range(32):
            a=np.nextafter(a,0.);b=np.nextafter(b,np.inf);candidates.extend([a,b])
        found=[]
        for value in candidates:
            calculated=(np.log(value)-np.log(.1))*(255/np.log(2560.))
            if calculated==i+.5:
                expected=i if i%2==0 else i+1
                actual=int(codec.scale_ids(value,(1,))[0]);assert actual==expected
                found.append(dict(scale_hex=float(value).hex(),index=actual))
        if found:ties.append(dict(boundary=i,representable=found))
        else:unrepresentable.append(i)
    assert ties and any(t['boundary']%2 for t in ties) and any(t['boundary']%2==0 for t in ties)
    result=dict(state='passed',source_sha256=hashlib.sha256((CODE/'codec.py').read_bytes()).hexdigest(),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy_version=np.__version__,
                decimal_precision=60,transition_sides_checked=510,exact_half_tie_boundaries_checked=len(ties),
                exact_half_ties=ties,no_exact_tie_in_65_neighboring_floats=unrepresentable,
                scope='No native rerun or source/CDF changes; high precision midpoint sides and exact representable computed-log ties.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('state','transition_sides_checked','exact_half_tie_boundaries_checked')}))


if __name__=='__main__':main()
