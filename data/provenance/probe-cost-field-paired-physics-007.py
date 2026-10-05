"""Fixed G256/P256 actual waveform pairing; no training or AP closure."""
import argparse, hashlib, importlib.util, io, json, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/provenance/audit-cost-field-full-007.py'
spec = importlib.util.spec_from_file_location('audit007', BASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
KEYS = ('wire_tx', 'wire_received', 'channel_noise', 'channel_fading', 'channel_received_baseband')
SEALED = ROOT / 'data/provenance/cost-field-full-local-prefix-006-001.json'

def blobs():
    directory = ROOT / 'data/runs/cost-field-full-seed17-004'
    rows = {'G': [], 'P': []}
    h = hashlib.sha256()
    with (directory / 'updates.jsonl').open('rb') as f:
        for line in f:
            assert line.endswith(b'\n')
            row = json.loads(line)
            arm, step = row['arm'], row['step']
            if arm in rows and step < 256:
                assert step == len(rows[arm])
                rows[arm].append((line, row))
                if arm == 'G': h.update(line)
            if arm == 'P' and step == 255: break
    sealed = json.loads(SEALED.read_text())
    assert len(rows['G']) == len(rows['P']) == 256
    assert h.hexdigest() == sealed['updates_jsonl_sha256']
    yield 'metadata.json', json.dumps(dict(seed=17, count=256, G_jsonl_sha256=h.hexdigest(), sealed_proof_sha256=m.sha(SEALED), verifier_sha256=m.sha(__file__), base_verifier_sha256=m.sha(BASE))).encode()
    for arm in ('G', 'P'):
        for line, row in rows[arm]:
            yield f'{arm}/{row["step"]}.json', line
            yield row['artifact'], (ROOT / row['artifact']).read_bytes()

def exported():
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for name, b in blobs():
            info = tarfile.TarInfo(name); info.size = len(b)
            archive.addfile(info, io.BytesIO(b))

def audit(iterator, output):
    it = iter(iterator)
    def get(name):
        actual, b = next(it); assert actual == name, (actual, name)
        return b
    metadata = json.loads(get('metadata.json'))
    assert metadata['sealed_proof_sha256'] == m.sha(SEALED)
    assert metadata['verifier_sha256'] == m.sha(__file__)
    assert metadata['base_verifier_sha256'] == m.sha(BASE)
    sealed = json.loads(SEALED.read_text())
    assert metadata['G_jsonl_sha256'] == sealed['updates_jsonl_sha256']
    physics = m.Audit.__new__(m.Audit)
    physics.maxima = dict(baseband=0., ZF=0.)
    physics.noise = [0., 0., 0]; physics.fading = [0., 0., 0]
    physics.uses = 0; physics.paired_physics = {}; physics.paired_comparisons = 0
    identities = {}; ledger = output.with_suffix('.records.jsonl')
    h = hashlib.sha256(); selected_values = 0; checked = 0
    result = dict(state='failed', seed=17, paired_cases=0, full_training_closed=False, AP_endpoint_closed=False)
    try:
        with ledger.open('x') as out:
            for arm in ('G', 'P'):
                for step in range(256):
                    line = get(f'{arm}/{step}.json'); row = json.loads(line)
                    assert (row['arm'], row['step'], row['epoch'], row['epoch_index']) == (arm, step, 1, step)
                    identity = (row['frame_id'], row['channel'], row['snr_db'], row['complex_uses'])
                    if arm == 'G': identities[step] = identity; h.update(line)
                    else: assert identity == identities[step]
                    expected = f'data/runs/cost-field-full-seed17-004/updates/{arm}/epoch1/block0000/{row["frame_id"]}-step{step}.npz'
                    assert row['artifact'] == expected
                    b = get(row['artifact'])
                    assert len(b) == row['bytes'] and m.digest(b) == row['sha256']
                    with np.load(io.BytesIO(b), allow_pickle=False) as z:
                        a = {k: z[k] for k in KEYS}
                    for k, v in a.items():
                        assert np.isfinite(v).all()
                        assert dict(dtype=str(v.dtype), shape=list(v.shape), sha256=m.digest(np.ascontiguousarray(v).tobytes())) == row['arrays'][k], k
                        selected_values += v.size
                    physics.physics(a, row)
                    out.write(json.dumps(dict(arm=arm, step=step, frame=row['frame_id'], row_sha256=m.digest(line), artifact_sha256=row['sha256'], noise_sha256=row['arrays']['channel_noise']['sha256'], fading_sha256=row['arrays']['channel_fading']['sha256'])) + '\n')
                    checked += 1
            assert h.hexdigest() == sealed['updates_jsonl_sha256']
            assert len(physics.paired_physics) == physics.paired_comparisons == 256
            try: next(it)
            except StopIteration: pass
            else: raise AssertionError('Unexpected stream suffix')
        result.update(state='passed_fixed_all256_G_P_actual_waveform_pairs', paired_cases=256)
    except BaseException:
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result.update(checked_records=checked, complete_selected_physical_array_values=selected_values, complex_uses=physics.uses, max_errors=physics.maxima, metadata=metadata, checked_record_ledger_sha256=m.sha(ledger), checked_unix=time.time(), limitation='All five saved physical arrays for all fixed first256 G/P records; no independent gradient, complete training, validation, AP or method-superiority claim.')
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('state','paired_cases','checked_records','complete_selected_physical_array_values')}))

def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('native','export','stream'), required=True); p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.mode == 'export': exported(); return
    assert args.output and not args.output.exists()
    audit(m.incoming() if args.mode == 'stream' else blobs(), args.output)

if __name__ == '__main__': main()
