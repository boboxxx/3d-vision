"""Twenty fixed received outcomes, eleven unchanged fresh neural+crop processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import contract as c


def execute(command, log_path):
    started = time.time()
    with log_path.open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=c.ROOT)
        returncode = process.wait()
    return dict(pid=process.pid, returncode=returncode, started_unix=started, ended_unix=time.time())


def run(directory):
    directory = directory.resolve(); c.need(not directory.exists(), 'unique native bridge directory')
    c.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('WANDB_MODE') == 'disabled', 'CPU-only isolated logger-free bridge')
    manifest, previous = c.predecessor()
    sources = c.identities()
    for filename in ('ecsic-neural-local-CPU-003.json', 'ecsic-neural-sheng-CPU-001.json'):
        gate = c.read(c.ROOT / 'data/engineering' / filename)
        c.need(gate['state'] == 'passed' and gate['tests'] == 5 and gate['sources'] == sources, 'current five CPU gates')
    reference = c.read(c.STAGE_B / 'manifest.json')
    c.need(reference['state'] == 'passed', 'completed stageB manifest')
    c.need(c.sha(c.STAGE_B / 'manifest.json') == manifest['stageB_manifest_sha256'], 'stageB/digital predecessor binding')
    reference_identities = {}
    for case in ('synthetic32x64', 'training000000'):
        locked_files = reference['cases'][case]['files']
        c.need(c.sha(c.STAGE_B / case / 'receiver.json') == locked_files['receiver.json'] and
               c.sha(c.STAGE_B / case / 'reconstructed.npz') == locked_files['reconstructed.npz'], 'sealed stageB reference files')
        expected = c.read(c.STAGE_B / case / 'receiver.json')
        c.need(expected['state'] == 'passed' and expected['state_count'] == 225 and expected['full_states_before'] == expected['full_states_after'], 'fixed225 stageB receiver')
        c.need(c.sha(c.STAGE_B / case / 'reconstructed.npz') == expected['output_sha256'], 'sealed full18 reference arrays')
        reference_identities[case] = dict(receiver_report_sha256=c.sha(c.STAGE_B / case / 'receiver.json'),
                                        arrays_sha256=expected['output_sha256'], versions=expected['versions'])
    directory.mkdir(parents=True)
    report_path = directory / 'manifest.json'
    info = dict(state='running', pid=os.getpid(), started_unix=time.time(), sources=sources, predecessor=previous,
                reference_identities=reference_identities, output_dir=str(directory), packets=[], current_packet=None,
                fresh_neural_processes=0, fresh_crop_processes=0, KITTI_AP_measured=False,
                no_new_source_encoding_or_channel_draws=True)
    c.save(report_path, info)
    try:
        for packet in manifest['packets']:
            name = packet['name']; info['current_packet'] = name; c.save(report_path, info)
            item = {k: packet[k] for k in ('index', 'name', 'source', 'configuration', 'channel', 'snr_db', 'attempted_uses', 'attempted_energy')}
            item.update(original_reception=packet['reception'], received_paths=packet['received_paths'])
            if packet['reception']['state'] == 'erased':
                c.need(packet['received_paths'] is None and packet['reception']['erasure_reason'], 'preserve erasure without fallback')
                item.update(state='erased_preserved_no_neural_or_crop', neural_calls=0, crop_calls=0, output_dir=None)
            else:
                target = directory / name; target.mkdir()
                container = c.ROOT / packet['received_paths']['payload']
                c.need(c.sha(container) == packet['received_paths']['payload_sha256'], 'only actual received bytes')
                command = [sys.executable, str(c.NATIVE), '--receiver', '--container', str(container), '--directory', str(target)]
                execution = execute(command, target / 'neural.log')
                c.need(execution['returncode'] == 0, 'fresh unchanged neural receiver failed: ' + name)
                info['fresh_neural_processes'] += 1
                record = c.read(target / 'receiver.json')
                c.need(record['state'] == 'passed' and record['container_sha256'] == packet['received_paths']['payload_sha256'], 'received byte causal binding')
                c.need(record['versions'] == reference_identities[packet['source']]['versions'], 'exact pinned CPU neural runtime')
                c.need(record['state_count'] == 225 and record['full_states_before'] == record['full_states_after'] and record['calls'] == dict(E=0, HE=0, HD=1, D=1), 'read-only causal225 state receiver')
                command_crop = [sys.executable, str(c.HERE / 'crop.py'), '--container', str(container), '--directory', str(target)]
                execution_crop = execute(command_crop, target / 'crop.log')
                c.need(execution_crop['returncode'] == 0, 'fresh public-header crop failed: ' + name)
                info['fresh_crop_processes'] += 1
                crop = c.read(target / 'crop.json')
                c.need(crop['state'] == 'passed' and crop['container_sha256'] == record['container_sha256'], 'same-packet crop evidence')
                item.update(state='neural_and_crop_finished_independent_audit_pending', neural_calls=1, crop_calls=1,
                            output_dir=str(target), receiver_command=command, crop_command=command_crop,
                            receiver_execution=execution, crop_execution=execution_crop,
                            files_sha256={p.name: c.sha(p) for p in sorted(target.iterdir()) if p.is_file()})
            info['packets'].append(item); c.save(report_path, info)
            print(json.dumps(dict(packet=name, state=item['state'])), flush=True)
        c.need(len(info['packets']) == 20 and info['fresh_neural_processes'] == info['fresh_crop_processes'] == 11, 'all fixed outcomes and11fresh child processes')
        c.need(c.identities() == sources and c.predecessor()[1] == previous, 'source/input identity changed')
        info.update(state='finished20_outcomes11_neural_crop_pairs_independent_audit_pending', current_packet=None, ended_unix=time.time())
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); raise
    finally: c.save(report_path, info)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); run(p.parse_args().directory)
