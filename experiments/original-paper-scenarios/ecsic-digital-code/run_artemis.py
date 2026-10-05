"""Twenty fixed actual digital packets; never invokes a neural ECSIC model."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from common import (ROOT, HERE, AUDIT_SHA, MANIFEST_SHA, PROTOCOL_SHA, RUNTIME_SHA,
                    SOURCES, CONFIGS, identities, source_payloads, packet_plan, read, save, sha, need)
from receiver import wrap_container, receive_container


def runtime_identity():
    import scipy
    import sionna
    import torch
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU Slurm scope required')
    need(not torch.cuda.is_available(), 'CPU-only digital runtime required')
    runtime_path = ROOT/'data/engineering/artemis-digital-runtime-001.json'
    need(sha(runtime_path) == RUNTIME_SHA, 'pinned runtime manifest changed')
    runtime = read(runtime_path)
    need(Path(sys.prefix).resolve() == Path(runtime['environment']).resolve(), 'wrong isolated environment')
    versions = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, sionna=sionna.__version__)
    need(versions == {k: runtime['CPU_imports'][k] for k in versions}, 'pinned PHY runtime versions changed')
    installed = Path(sionna.__file__).parent
    files = {str(p.relative_to(installed)): sha(p) for p in sorted(installed.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.csv')}
    need(files == runtime['installed_source_sha256'], 'installed PHY source identity changed')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    need(hashlib.sha256(freeze.encode()).hexdigest() == runtime['freeze_sha256'], 'dependency inventory changed')
    return dict(versions=versions, environment=sys.prefix, runtime_sha256=RUNTIME_SHA,
                installed_source_sha256=files, pip_freeze=freeze)


def run(output):
    output = Path(output).resolve()
    artifacts = output.with_suffix('')
    need(not output.exists() and not artifacts.exists(), 'unique packet run paths required')
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    record = dict(state='running', pid=os.getpid(), job_id=os.environ.get('SLURM_JOB_ID'), started_unix=time.time(),
                  scope='ECSIC_actual_digital_CPU_no_neural_decoder_no_AP', protocol_sha256=PROTOCOL_SHA,
                  stageB_audit_sha256=AUDIT_SHA, stageB_manifest_sha256=MANIFEST_SHA, packets=[],
                  legacy_transceiver_trimmed_bytes_discarded=True, source_length_side_channel=False,
                  current_packet=None)
    save(output, record)
    try:
        before = identities()
        record['sources_before'] = before
        local_path = ROOT/'data/engineering/ecsic-digital-local-CPU-001.json'
        local = read(local_path)
        need(local['state'] == 'passed_independent_P6EC_framing_checks' and local['checks_count'] == 148,
             'complete local framing gate required')
        need(local['successful_fixtures'] == 4 and local['fatal_dependency_failure_propagated'], 'local gate scope')
        need(local['source_sha256'] == local['source_after_sha256'] == before, 'local gate source identity')
        need(local['protocol_sha256'] == PROTOCOL_SHA, 'local gate protocol')
        for path, digest in local['artifact_sha256'].items():
            need(sha(ROOT/path) == digest, 'local fixture identity')
        record['local_gate_sha256'] = sha(local_path)
        payloads = source_payloads()
        runtime = runtime_identity()
        record['runtime_before'] = runtime
        import torch
        torch.set_num_threads(4)
        # Historical PHY is imported only after all immutable identity gates.
        sys.path.insert(0, str(HERE.parent/'digital-code'))
        from transceiver import DigitalTransceiver
        global_numpy = copy.deepcopy(np.random.get_state())
        global_torch = torch.random.get_rng_state().clone()
        wires = {}
        for source, payload in payloads.items():
            wire = wrap_container(payload)
            (artifacts/(source+'-source.p6ec')).write_bytes(payload)
            (artifacts/(source+'-source.p6sb')).write_bytes(wire)
            wires[source] = wire
        links = {}
        plan = packet_plan()
        need(plan == local['packet_plan'] and len(plan) == 20, 'prelocked complete packet plan')
        record['packet_plan'] = plan
        save(output, record)
        for item in plan:
            name, source, config = item['name'], item['source'], item['configuration']
            record['current_packet'] = item
            save(output, record)
            if config not in links:
                links[config] = DigitalTransceiver(config)
            link = links[config]
            k, n, bps = CONFIGS[config]
            need((link.k, link.n, link.bps) == (k, n, bps), 'physical code configuration changed')
            # Deliberately discard the first return; it depends on transmitter length.
            _, diagnostic, arrays = link.transmit(wires[source], kind=item['channel'], snr_db=item['snr_db'],
                noise_rng=np.random.Generator(np.random.PCG64(item['noise_seed'])),
                fading_rng=np.random.Generator(np.random.PCG64(item['fading_seed'])))
            array_path = artifacts/(name+'.npz')
            np.savez_compressed(array_path, **arrays)
            record['current_packet'] = dict(**item, arrays_path=str(array_path.relative_to(ROOT)), arrays_sha256=sha(array_path))
            save(output, record)
            wire, payload, reception = receive_container(arrays['decoded'], k=k)
            source_bits = len(wires[source])*8
            blocks = (source_bits+k-1)//k
            need(diagnostic['blocks'] == blocks and diagnostic['source_bits'] == source_bits and
                 diagnostic['source_padding_bits'] == blocks*k-source_bits, 'actual complete wire block accounting')
            uses = blocks*n//bps
            energy = float(np.sum(np.abs(arrays['transmitted'])**2))
            need(diagnostic['channel']['uses'] == len(arrays['transmitted']) == uses, 'actual attempted uses')
            need(diagnostic['channel']['actual_energy'] == energy, 'actual attempted energy')
            if item['channel'] == 'identity':
                need(wire == wires[source] and payload == payloads[source], 'identity full-container equality')
                need(diagnostic['channel']['rng_before'] == diagnostic['channel']['rng_after'], 'identity consumed RNG')
            need(all(np.array_equal(a, b) for a, b in zip(global_numpy, np.random.get_state())), 'global NumPy RNG changed')
            need(torch.equal(global_torch, torch.random.get_rng_state()), 'global Torch RNG changed')
            received_paths = None
            if reception['state'] == 'received':
                need(wire is not None and payload is not None, 'received bytes absent')
                wire_path = artifacts/(name+'-received.p6sb')
                payload_path = artifacts/(name+'-received.p6ec')
                wire_path.write_bytes(wire)
                payload_path.write_bytes(payload)
                received_paths = dict(wire=str(wire_path.relative_to(ROOT)), payload=str(payload_path.relative_to(ROOT)),
                                      wire_sha256=sha(wire_path), payload_sha256=sha(payload_path))
            else:
                need(wire is None and payload is None and reception['erasure_reason'], 'erasure fallback forbidden')
            record['packets'].append(dict(**item, diagnostic=diagnostic, reception=reception,
                arrays_path=str(array_path.relative_to(ROOT)), arrays_sha256=sha(array_path),
                source_payload_sha256=SOURCES[source][1], source_wire_sha256=hashlib.sha256(wires[source]).hexdigest(),
                source_payload_bytes=len(payloads[source]), outer_header_bytes=20, inner_header_CRC_bytes=166,
                source_wire_bytes=len(wires[source]), attempted_uses=uses, attempted_energy=energy,
                physical_accounting_on_erasure_unchanged=True, received_paths=received_paths,
                received_payload_equals_source_diagnostic_only=(payload == payloads[source]) if payload is not None else None))
            record['current_packet'] = None
            save(output, record)
            print(json.dumps(dict(packet=name, reception=reception['state'], reason=reception['erasure_reason'],
                                  uses=uses, energy=energy)), flush=True)
        need(len(record['packets']) == 20 and identities() == before, 'complete packets and unchanged source')
        need(runtime_identity() == runtime, 'runtime changed during packet sequence')
        record.update(state='finished20_packets_independent_audit_pending', sources_after=identities(), runtime_after=runtime,
                      received_packets=sum(p['reception']['state'] == 'received' for p in record['packets']),
                      erased_packets=sum(p['reception']['state'] == 'erased' for p in record['packets']),
                      global_numpy_torch_rng_unchanged=True, neural_decoder_executed=False, KITTI_AP_measured=False)
    except BaseException:
        record.update(state='failed', traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in artifacts.iterdir()})
        save(output, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
