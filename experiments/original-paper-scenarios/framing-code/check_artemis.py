"""Actual LDPC/QAM/BP packets followed by a receiver without byte-length oracle."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import sionna
import torch

from receiver import ROOT, receive_stereo
from check_local import pair, source_identities
from image_codecs import encode_pair, decode_pair
sys.path.insert(0,str(ROOT/'experiments/original-paper-scenarios/digital-code'))
from transceiver import DigitalTransceiver


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def all_sources():
    out=source_identities(); d=Path(sionna.__file__).parent
    out.update({'sionna/'+str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*'))
                if p.is_file() and p.suffix in ('.py','.csv')})
    return out


def main():
    output=ROOT/'data/engineering/artemis-digital-framing-CPU-001.json'
    artifacts=output.with_suffix('')
    assert not output.exists() and not artifacts.exists(), 'unique result paths required'
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    assert torch.__version__=='2.9.1+cpu' and sionna.__version__=='2.1.0'
    runtime=json.loads((ROOT/'data/engineering/artemis-digital-runtime-001.json').read_text())
    installed=Path(sionna.__file__).parent
    assert all(sha(installed/p)==v for p,v in runtime['installed_source_sha256'].items())
    torch.set_num_threads(4); artifacts.mkdir()
    before=all_sources(); freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    record=dict(state='running',scope='received_header_actual_digital_CPU_engineering_no_AP',
                job_id=os.environ['SLURM_JOB_ID'],started_unix=time.time(),
                protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/framing-protocol-001.md'),
                runtime_sha256=sha(ROOT/'data/engineering/artemis-digital-runtime-001.json'),
                sources_before=before,pip_freeze_before=freeze,
                torch=torch.__version__,numpy=np.__version__,sionna=sionna.__version__,packets=[])
    with output.open('x') as f:json.dump(record,f,indent=2)
    try:
        native=ROOT/'assets/digital-native-001/jpeg2000-30.p6sb'
        assert sha(native)=='a9e92e1c66e138f35497984e763db1b63b58be54878d5413b08d68571d3912cd'
        native_wire=native.read_bytes(); native_images=decode_pair(native_wire)
        small_wire,_,_=encode_pair(*pair(),codec='jpeg2000',parameter=4)
        (artifacts/'native-source.p6sb').write_bytes(native_wire)
        (artifacts/'synthetic-source.p6sb').write_bytes(small_wire)
        for config in ('ldpc_2_3_qam64','ldpc_1_2_qam256'):
            link=DigitalTransceiver(config)
            settings=[('native','identity',10,native_wire)]
            settings.extend(('synthetic',kind,snr,small_wire) for kind in ('awgn','rayleigh') for snr in (6,18))
            for source,kind,snr,payload in settings:
                np_before=copy.deepcopy(np.random.get_state());torch_before=torch.random.get_rng_state().clone()
                # Legacy trimmed bytes are an offline diagnostic ONLY, discarded here.
                _, diagnostic, arrays=link.transmit(payload,kind=kind,snr_db=snr,
                    noise_rng=np.random.Generator(np.random.PCG64(1901)),
                    fading_rng=np.random.Generator(np.random.PCG64(1902)))
                wire,images,reception=receive_stereo(arrays['decoded'],k=link.k)
                assert all(np.array_equal(a,b) for a,b in zip(np_before,np.random.get_state()))
                assert torch.equal(torch_before,torch.random.get_rng_state())
                assert reception['receiver_inputs']==['decoded_information_blocks','public_code_k']
                assert not reception['source_length_side_channel']
                if kind=='identity':
                    assert wire==payload and reception['header_derived_bytes']==90545
                    assert all(np.array_equal(a,b) for a,b in zip(images,native_images))
                    assert images[0].shape==images[1].shape==(370,1224,3)
                name=f'{config}-{source}-{kind}-{snr}'
                np.savez_compressed(artifacts/(name+'.npz'),**arrays)
                if wire is not None: (artifacts/(name+'.p6sb')).write_bytes(wire)
                else: assert images is None and reception['state']=='erased'
                record['packets'].append(dict(name=name,source=source,diagnostic=diagnostic,reception=reception,
                    physical_accounting_on_erasure_unchanged=True,
                    attempted_uses=diagnostic['channel']['uses'],attempted_energy=diagnostic['channel']['actual_energy'],
                    arrays_path=str((artifacts/(name+'.npz')).relative_to(ROOT)),
                    source_wire_path=str((artifacts/(source+'-source.p6sb')).relative_to(ROOT))))
                output.write_text(json.dumps(record,indent=2)+'\n')
        assert len(record['packets'])==10 and all_sources()==before
        afterfreeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        assert afterfreeze==freeze
        record.update(state='passed_actual_receiver_framing_audit_pending',sources_after=all_sources(),
                      pip_freeze_after=afterfreeze,
                      artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(artifacts.iterdir())},
                      successful_packets=sum(p['reception']['state']=='received' for p in record['packets']),
                      erased_packets=sum(p['reception']['state']=='erased' for p in record['packets']))
    except Exception as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record['finished_unix']=time.time();output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('state','successful_packets','erased_packets')}))


if __name__=='__main__':main()
