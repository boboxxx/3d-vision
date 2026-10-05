"""Independent full 20-packet audit in the pinned Artemis Sionna CPU runtime.

Usage:
  .../envs/digital-cpu-001/bin/python data/provenance/verify-ecsic-digital-CPU-001.py \
    --terminal-json data/provenance/artemis-ecsic-digital-CPU-001-terminal.json \
    --output data/provenance/ecsic-digital-CPU-001-audit.json

Terminal JSON is independently collected AFTER the original job exits:
  schema: ecsic-digital-terminal-v1; state: completed_exit0_CPU_only
  job_id: original numeric job ID as a string; checked_unix: UTC Unix seconds
  slurm_time_zone: UTC
  sacct_command: actual argv, including env TZ=UTC, -j ID and the six fields
  sacct_raw: raw -n -P rows JobIDRaw|State|ExitCode|AllocTRES|Start|End
  squeue_command: actual argv including -h -j ID; squeue_raw: empty output
  fresh_file_sha256: root-relative path -> fresh SHA256, including the run
    manifest and EVERY run artifact. Other paths such as logs are permitted.

No project receiver, entropy parser, digital transceiver or model is imported.
Every codeword is checked by full mother-code GF(2) syndrome, explicit RV0/
filler/interleaver construction and actual public encoder replay. Every BP
information bit is replayed from the saved, independently checked APP LLRs.
Original global RNG states were not serialized; this audit verifies the sealed
per-packet runtime assertions and independently demonstrates unchanged global
states during the full replay. It does not invent original state digests.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zlib

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'experiments/original-paper-scenarios/ecsic-digital-protocol-001.md'
PROTOCOL_SHA = 'da3c6a000af96628d954bbcdd2af8cce8343c3b7d66d731ceb9d5731af54208a'
AUDIT_SHA = '7fb7ca99b75054e7e099e2435a05ff94dff00fe1e8896ba320ed83d86bd99524'
MANIFEST_SHA = 'e098d9ec03e7783746ee9a150d0caf2b71e2c82816a47b088b3c4e6e54e6b867'
RUNTIME_SHA = '921c0cbb2b28d7671e62dfaf28b4ab504cd7956ebe9bdf53a2bb94762d61ecd1'
MODEL_SHA = 'e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
CONFIG_SHA = 'ba02cc1239b755a26ca0eb920c3014908c97c45fa210c447c47e96f7527102f8'
CDF_SHA = '507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
SOURCES = [('synthetic32x64', 813, 'acbc2be75dc60f74903dc19c57d100f2ea9772a9c4606f3fb7681e75a632a35f'),
           ('training000000', 44824, 'e7242cca0a0edd43dbda26b8892894967a75690efeb1716ee538371c1905eb4a')]
CONFIGURATIONS = [('ldpc_2_3_qam64', 1296, 1944, 6), ('ldpc_1_2_qam256', 972, 1944, 8)]
SETTINGS = [('identity', 10), ('awgn', 6), ('awgn', 18), ('rayleigh', 6), ('rayleigh', 18)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def local_path(value):
    p = (ROOT / value).resolve()
    check(p.is_relative_to(ROOT), 'evidence path outside research root')
    return p


def array_check(array, record):
    check(list(array.shape) == record['shape'] and array.dtype.str == record['dtype'], 'array shape/dtype evidence')
    check(digest(np.ascontiguousarray(array).tobytes()) == record['sha256'], 'array byte identity')


def terminal_check(path, manifest_path, run):
    t = read(path)
    check(t['schema'] == 'ecsic-digital-terminal-v1' and t['state'] == 'completed_exit0_CPU_only', 'actual terminal evidence schema')
    job = str(run['job_id'])
    check(job.isdigit() and str(t['job_id']) == job, 'actual original job identity')
    check(t['slurm_time_zone'] == 'UTC' and t['checked_unix'] >= run['finished_unix'], 'terminal observation time')
    check(t['checked_unix'] <= time.time()+60, 'future terminal evidence')
    command = t['sacct_command']
    check(isinstance(command, list) and 'sacct' in command and 'TZ=UTC' in command and '-j' in command and
          command[command.index('-j')+1] == job and '-n' in command and '-P' in command,
          'actual UTC sacct command for this job')
    fields = 'JobIDRaw,State,ExitCode,AllocTRES,Start,End'
    check('--format='+fields in command or ('--format' in command and command[command.index('--format')+1] == fields), 'sacct raw column contract')
    queue = t['squeue_command']
    check(isinstance(queue, list) and 'squeue' in queue and '-h' in queue and '-j' in queue and
          queue[queue.index('-j')+1] == job and not t['squeue_raw'].strip(), 'job absent from actual squeue')
    rows = []
    for line in t['sacct_raw'].splitlines():
        values = line.strip().split('|')
        if not line.strip():
            continue
        if len(values) == 7 and values[-1] == '':
            values.pop()
        check(len(values) == 6, 'raw sacct row field count')
        jid, state, code, resources, start, end = values
        check(jid == job or jid.startswith(job+'.'), 'unrelated job in terminal evidence')
        check(state == 'COMPLETED' and code == '0:0' and 'gpu' not in resources.lower(), 'all original CPU job steps completed successfully')
        tres = dict(v.split('=', 1) for v in resources.split(',') if '=' in v)
        check(int(tres.get('cpu', '0')) > 0, 'actual allocated CPU evidence')
        start_ts = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
        end_ts = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
        check(start_ts <= end_ts <= t['checked_unix'], 'sacct actual start/end chronology')
        rows.append(dict(job_id=jid, state=state, exit_code=code, AllocTRES=resources, start_UTC=start, end_UTC=end))
    check(any(r['job_id'] == job for r in rows) and any(r['job_id'] == job+'.batch' for r in rows), 'root and batch terminal rows required')
    needed = {str(manifest_path.relative_to(ROOT)): sha(manifest_path), **run['artifact_sha256']}
    check(all(t['fresh_file_sha256'].get(p) == h for p, h in needed.items()), 'post-terminal full artifact binding')
    for p, h in t['fresh_file_sha256'].items():
        check(sha(local_path(p)) == h, 'fresh terminal file changed: '+p)
    return dict(job_id=job, terminal_sha256=sha(path), checked_unix=t['checked_unix'], rows=rows,
                raw_sacct_sha256=digest(t['sacct_raw'].encode()), raw_squeue_sha256=digest(t['squeue_raw'].encode()))


def inner_parse(blob):
    def require(ok, why):
        if not ok:
            raise ValueError('inner_P6EC_failure:'+why)
    require(182 <= len(blob) <= 33554432, 'container length bound')
    require(zlib.crc32(blob[:-4]) == int.from_bytes(blob[-4:], 'big'), 'container CRC')
    require(blob[:6] == b'P6EC\x01\x04', 'container format')
    require(tuple(blob[p:p+32].hex() for p in (14, 46, 78)) == (MODEL_SHA, CONFIG_SHA, CDF_SHA), 'container model/config/CDF identity')
    h, w, ph, pw = [int.from_bytes(blob[p:p+2], 'big') for p in (6, 8, 10, 12)]
    require(0 < h <= ph <= 2048 and 0 < w <= pw <= 4096, 'dimension bounds')
    require(ph % 32 == pw % 32 == 0 and ph-h < 32 and pw-w < 32, 'padding rule')
    offset, streams = 162, []
    for index in range(4):
        pos = 110+13*index
        sid = blob[pos]
        count, nr, ne = [int.from_bytes(blob[p:p+4], 'big') for p in (pos+1, pos+5, pos+9)]
        factor = 32 if index < 2 else 8
        expected = 48*(ph//factor)*(pw//factor)
        require(sid == index and count == expected, 'stream id/count')
        require(nr >= 4 and ne <= 4*expected and offset+nr+ne <= len(blob)-4, 'stream lengths')
        streams.append(dict(id=index, symbols=count, rans_bytes=nr, escape_bytes=ne,
                            rans_sha256=digest(blob[offset:offset+nr]), escapes_sha256=digest(blob[offset+nr:offset+nr+ne])))
        offset += nr+ne
    require(offset == len(blob)-4, 'unused container bytes')
    return dict(original_hw=[h, w], padded_hw=[ph, pw], inner_overhead_bytes=166, streams=streams)


def receive_bits(decoded, k):
    # This receiver has no source-length/source-byte argument or imported parser.
    def require(ok, why):
        if not ok:
            raise ValueError(why)
    try:
        require(type(k) is int and k in (1296, 972), 'unsupported_public_k')
        require(decoded.ndim == 2 and len(decoded) > 0 and decoded.shape[1] == k, 'decoded_block_shape')
        require(len(decoded) <= (8*(33554432+20)+k-1)//k, 'observed_capacity_limit')
        require(decoded.dtype.kind in 'biuf' and np.isfinite(decoded).all() and np.isin(decoded, [0, 1]).all(), 'nonbinary_decoded_bits')
        bits = decoded.astype(np.uint8).reshape(-1)
        require(bits.size >= 160, 'truncated_header')
        header = np.packbits(bits[:160], bitorder='big').tobytes()
        for ok, why in [(header[:4] == b'P6SB', 'header_magic'), (header[4] == 2, 'header_version'),
                        (header[5] == 3, 'header_codec'), (header[6:8] == bytes(2), 'header_reserved')]:
            require(ok, why)
        length, second, crc = [int.from_bytes(header[p:p+4], 'big') for p in (8, 12, 16)]
        require(182 <= length <= 33554432, 'header_payload_length')
        require(second == 0, 'header_second_length')
        end = 8*(20+length)
        require(end <= bits.size, 'header_capacity_overflow')
        require((end+k-1)//k == len(decoded), 'observed_block_count_mismatch')
        require(not bits[end:].any(), 'nonzero_decoded_padding')
        wire = np.packbits(bits[:end], bitorder='big').tobytes()
        payload = wire[20:]
        require(zlib.crc32(payload) == crc, 'outer_payload_CRC_failure')
        inner = inner_parse(payload)
        metadata = dict(observed_blocks=len(decoded), public_k=k, decoded_header_hex=header.hex(), header_bytes=20,
                        header_derived_wire_bytes=len(wire), header_derived_payload_bytes=length,
                        source_padding_bits=int(bits.size-end), codec='ecsic_P6EC', outer_version=2,
                        payload_lengths=[length, 0], received_wire_sha256=digest(wire), received_payload_sha256=digest(payload),
                        **{key: inner[key] for key in ('original_hw', 'padded_hw', 'inner_overhead_bytes')})
        return None, wire, payload, metadata, inner
    except ValueError as error:
        return str(error), None, None, None, None


def pam(bits):
    if len(bits) == 1:
        return 1-2*int(bits[0])
    return (1-2*int(bits[0]))*(2**(len(bits)-1)-pam(bits[1:]))


def runtime_check(run):
    import scipy
    import sionna
    import torch
    path = ROOT/'data/engineering/artemis-digital-runtime-001.json'
    check(sha(path) == RUNTIME_SHA, 'pinned environment manifest')
    runtime = read(path)
    check(sys.prefix == runtime['environment'] and torch.__version__ == '2.9.1+cpu' and not torch.cuda.is_available(), 'pinned CPU environment')
    versions = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, sionna=sionna.__version__)
    check(versions == {k: runtime['CPU_imports'][k] for k in versions}, 'runtime version identity')
    package = Path(sionna.__file__).parent
    files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py', '.csv')}
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    check(files == runtime['installed_source_sha256'] and digest(freeze.encode()) == runtime['freeze_sha256'], 'complete installed PHY/dependency identity')
    expected = dict(versions=versions, environment=sys.prefix, runtime_sha256=RUNTIME_SHA, installed_source_sha256=files, pip_freeze=freeze)
    check(run['runtime_before'] == run['runtime_after'] == expected, 'actual run and replay environments equal')
    return expected


def audit_local_fixtures(run):
    path = ROOT/'data/engineering/ecsic-digital-local-CPU-001.json'
    gate = read(path)
    check(sha(path) == run['local_gate_sha256'] and gate['state'] == 'passed_independent_P6EC_framing_checks', 'local gate binding')
    check(gate['source_sha256'] == gate['source_after_sha256'] == run['sources_before'], 'local gate source identity')
    check(gate['fatal_dependency_failure_propagated'] and gate['global_numpy_rng_unchanged'], 'local dependency/RNG assertions')
    for p, h in gate['artifact_sha256'].items():
        check(sha(local_path(p)) == h, 'local fixture file identity')
    fixtures = path.with_suffix('.artifacts')/'decoded-fixtures.npz'
    with np.load(fixtures, allow_pickle=False) as data:
        check(len(data.files) == len(gate['checks']) == gate['checks_count'] == 148, 'complete148 fixtures')
        check(set(data.files) == {r['name'] for r in gate['checks']}, 'local fixture names')
        for row in gate['checks']:
            reason, wire, payload, metadata, _ = receive_bits(data[row['name']], row['public_k'])
            expected = row['expected_reason']
            check(reason == expected or (expected == 'inner_P6EC_failure' and reason.startswith(expected+':')), 'independent malformed fixture reason')
            check(reason == row['reception']['erasure_reason'], 'recorded local reception reason')
            if reason is None:
                source = row['name'].split('-', 1)[0]
                check(wire == (path.with_suffix('.artifacts')/(source+'.p6sb')).read_bytes(), 'local intact wire')
                check(all(row['reception'][k] == v for k, v in metadata.items()), 'local received metadata')
            else:
                check(wire is None and payload is None, 'local erasure no fallback')
    return dict(fixtures=148, valid=4, erased=144, manifest_sha256=sha(path))


def audit_packet(item, expected, folder, encoder, decoder, public_qam, torch):
    source, size, source_sha, config, k, n, bps, j, kind, snr = expected
    name = f'{j:02d}-{source}-{config}-{kind}-{snr}'
    plan = dict(index=j, source=source, configuration=config, channel=kind, snr_db=snr,
                noise_seed=1911+2*j, fading_seed=1912+2*j, name=name)
    check(all(item[field] == value for field, value in plan.items()), 'fixed packet condition/order/seed')
    payload = (folder/(source+'-source.p6ec')).read_bytes()
    check(len(payload) == size and digest(payload) == source_sha, 'fixed actual entropy source')
    inner_parse(payload)
    expected_wire = b'P6SB\x02\x03\0\0'+size.to_bytes(4, 'big')+bytes(4)+zlib.crc32(payload).to_bytes(4, 'big')+payload
    wire = (folder/(source+'-source.p6sb')).read_bytes()
    check(wire == expected_wire, 'full outer wire including counted header')
    for field, value in [('source_payload_sha256', source_sha), ('source_wire_sha256', digest(wire)),
                         ('source_payload_bytes', size), ('source_wire_bytes', size+20), ('outer_header_bytes', 20), ('inner_header_CRC_bytes', 166)]:
        check(item[field] == value, 'source accounting '+field)
    path = local_path(item['arrays_path'])
    check(path == folder/(name+'.npz') and sha(path) == item['arrays_sha256'], 'actual packet array file')
    with np.load(path, allow_pickle=False) as raw:
        a = {key: raw[key] for key in raw.files}
    check(set(a) == {'source_padded', 'coded', 'decoded', 'transmitted', 'received', 'noise', 'fading', 'effective_noise_variance', 'llr'}, 'complete raw packet arrays')
    bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8), bitorder='big')
    B = (len(bits)+k-1)//k
    padded = np.pad(bits, (0, B*k-len(bits))).reshape(B, k)
    np.testing.assert_array_equal(a['source_padded'], padded)
    for key, shape in [('coded', (B, n)), ('decoded', (B, k))]:
        check(a[key].shape == shape and a[key].dtype == np.uint8 and np.isin(a[key], [0, 1]).all(), 'binary code array layout')
        array_check(a[key], item['diagnostic'][key])
    r, c = item['diagnostic'], item['diagnostic']['channel']
    check(all(r[key] == value for key, value in dict(source_bytes=len(wire), source_bits=len(bits), blocks=B,
          source_padding_bits=B*k-len(bits), coded_bits=B*n, QAM_padding_bits=0, reliable_uses=0, reliable_energy=0.0).items()), 'actual block/padding/overhead accounting')
    ident = r['identity']
    fields = dict(configuration=config, k=k, n=n, bits_per_symbol=bps, basegraph=encoder._bg,
                  lifting_factor=encoder.z, k_ldpc=encoder.k_ldpc, n_ldpc=encoder.n_ldpc, k_filler=encoder.k_filler,
                  punctured_first_bits=2*encoder.z, RV=0, code_family='NR_38.212_declared_ambiguity_variant',
                  pcm_shape=list(encoder.pcm.shape), decoder_iterations=20, decoder_llr_clip=20.0,
                  decoder_rule='boxplus-phi', decoder_schedule='flooding', precision='double', device='cpu')
    check(all(ident[key] == value for key, value in fields.items()), 'full actual code configuration')
    interleaver = encoder.out_int.cpu().numpy()
    for key, value in [('output_interleaver', interleaver), ('pcm_indptr', encoder.pcm.indptr),
                       ('pcm_indices', encoder.pcm.indices), ('pcm_values', encoder.pcm.data), ('points', public_qam)]:
        array_check(value, ident[key])
    # Every actual block: public encoding, full mother PCM, systematic/filler,
    # independent rate matching and exact output interleaving.
    for start in range(0, B, 32):
        data = padded[start:start+32].astype(np.float64)
        filled = np.pad(data, ((0, 0), (0, encoder.k_ldpc-k)))
        with torch.no_grad():
            public = encoder(torch.from_numpy(data)).cpu().numpy().astype(np.uint8)
            mother = encoder._encode_fast(torch.from_numpy(filled)).cpu().numpy().reshape(len(data), encoder.n_ldpc).astype(np.int64)
        check(not (encoder.pcm.astype(np.int64).dot(mother.T) % 2).any(), 'complete mother PCM syndrome')
        np.testing.assert_array_equal(mother[:, :encoder.k_ldpc], filled)
        no_filler = np.concatenate((mother[:, :k], mother[:, encoder.k_ldpc:]), axis=1)
        matched = no_filler[:, 2*encoder.z:][:, :n][:, interleaver]
        np.testing.assert_array_equal(public, matched)
        np.testing.assert_array_equal(a['coded'][start:start+32], public)
    labels = ((np.arange(2**bps)[:, None] >> np.arange(bps-1, -1, -1)) & 1).astype(np.uint8)
    array_check(labels, ident['labels'])
    points = np.asarray([pam(b[0::2])+1j*pam(b[1::2]) for b in labels], dtype=np.complex128)/math.sqrt(2*(2**bps-1)/3)
    check(abs(float(np.mean(np.abs(points)**2))-1) < 1e-14, 'population Es normalization')
    np.testing.assert_allclose(public_qam, points, rtol=1e-15, atol=1e-15)
    indices = a['coded'].reshape(-1, bps).astype(np.int64).dot(2**np.arange(bps-1, -1, -1))
    N = len(indices)
    check(N == B*n//bps, 'all mapped coded bits charged')
    np.testing.assert_allclose(a['transmitted'], points[indices], rtol=1e-15, atol=1e-15)
    for key in ('transmitted', 'received', 'noise', 'fading'):
        check(a[key].shape == (N,) and a[key].dtype == np.complex128 and np.isfinite(a[key]).all(), 'physical complex array')
        array_check(a[key], c[key])
    check(a['effective_noise_variance'].shape == (N,) and a['effective_noise_variance'].dtype == np.float64, 'variance layout')
    array_check(a['effective_noise_variance'], c['effective_noise_variance'])
    n0 = 1e-6 if kind == 'identity' else 10**(-snr/10)
    channel_fields = dict(channel=kind, SNR_dB=float(snr), nominal_population_Es=1.0, N0=n0,
                          identity_demapper_variance_only=kind == 'identity', uses=N, pilot_uses=0,
                          equalizer='exact_ZF', CSI='perfect' if kind == 'rayleigh' else 'not_needed',
                          coherence_symbols=1 if kind == 'rayleigh' else None)
    check(all(c[key] == value for key, value in channel_fields.items()), 'channel convention')
    nrng, hrng = np.random.Generator(np.random.PCG64(1911+2*j)), np.random.Generator(np.random.PCG64(1912+2*j))
    check(c['rng_before'] == dict(noise=nrng.bit_generator.state, fading=hrng.bit_generator.state), 'packet dedicated seeds')
    noise, h = np.zeros(N, dtype=np.complex128), np.ones(N, dtype=np.complex128)
    if kind != 'identity':
        normal = nrng.standard_normal((N, 2))*math.sqrt(n0/2)
        noise = normal[:, 0]+1j*normal[:, 1]
        if kind == 'rayleigh':
            normal = hrng.standard_normal((N, 2))/math.sqrt(2)
            h = normal[:, 0]+1j*normal[:, 1]
    np.testing.assert_array_equal(a['noise'], noise)
    np.testing.assert_array_equal(a['fading'], h)
    check(c['rng_after'] == dict(noise=nrng.bit_generator.state, fading=hrng.bit_generator.state), 'exact dedicated RNG advancement')
    reconstructed = (h*a['transmitted']+noise)/h
    np.testing.assert_allclose(a['received'], reconstructed, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(a['effective_noise_variance'], n0/np.abs(h)**2, rtol=3e-15, atol=0)
    check(a['llr'].shape == (B, n) and a['llr'].dtype == np.float64 and np.isfinite(a['llr']).all(), 'all actual APP LLRs')
    llr = a['llr'].reshape(N, bps)
    max_error = 0.0
    for start in range(0, N, 2048):
        end = min(start+2048, N)
        score = -np.abs(a['received'][start:end, None]-points[None, :])**2/a['effective_noise_variance'][start:end, None]
        wanted = np.column_stack([logsumexp(score[:, labels[:, bit] == 1], axis=1)-logsumexp(score[:, labels[:, bit] == 0], axis=1) for bit in range(bps)])
        np.testing.assert_allclose(llr[start:end], wanted, rtol=5e-13, atol=1e-9)
        max_error = max(max_error, float(np.max(np.abs(llr[start:end]-wanted))))
    for start in range(0, B, 32):
        with torch.no_grad():
            hard = decoder(torch.from_numpy(a['llr'][start:start+32])).cpu().numpy().astype(np.uint8)
        np.testing.assert_array_equal(a['decoded'][start:start+32], hard)
    energy = float(np.sum(np.abs(a['transmitted'])**2))
    np.testing.assert_allclose([item['attempted_energy'], c['actual_energy']], energy, rtol=1e-15, atol=1e-10)
    np.testing.assert_allclose(c['actual_mean_Es'], float(np.mean(np.abs(a['transmitted'])**2)), rtol=1e-15, atol=0)
    check(item['attempted_uses'] == N and item['physical_accounting_on_erasure_unchanged'], 'failed attempt charging')
    offline = a['decoded'].ravel()[:len(bits)]
    offline_bytes = np.packbits(offline, bitorder='big').tobytes()
    check(r['source_bit_errors'] == int(np.count_nonzero(offline != bits)), 'diagnostic bit errors')
    check(r['source_block_errors'] == int(np.count_nonzero(np.any(a['decoded'] != padded, axis=1))), 'diagnostic block errors')
    check(r['source_padding_valid'] == bool(not a['decoded'].ravel()[len(bits):].any()), 'diagnostic source tail')
    check(r['exact_payload_recovered'] == (offline_bytes == wire) and r['source_sha256'] == digest(wire) and
          r['received_sha256'] == digest(offline_bytes), 'legacy diagnostic identity only')
    reason, received_wire, received_payload, metadata, inner = receive_bits(a['decoded'], k)
    reception = item['reception']
    check(reception['receiver_inputs'] == ['decoded_information_blocks', 'public_code_k'] and
          not reception['source_length_side_channel'] and not reception['neural_decoder_called'], 'actual receiver scope')
    check(reception['erasure_reason'] == reason and reception['state'] == ('erased' if reason else 'received'), 'independent actual reception')
    paths = item['received_paths']
    if reason:
        check(received_wire is None and received_payload is None and paths is None, 'whole-pair erasure without fallback')
        check(not (folder/(name+'-received.p6sb')).exists() and not (folder/(name+'-received.p6ec')).exists(), 'no hidden output on erasure')
        check(item['received_payload_equals_source_diagnostic_only'] is None, 'erasure no source fallback comparison')
    else:
        check(all(reception[key] == value for key, value in metadata.items()), 'header-derived actual metadata')
        check(paths['wire'] == str((folder/(name+'-received.p6sb')).relative_to(ROOT)) and
              paths['payload'] == str((folder/(name+'-received.p6ec')).relative_to(ROOT)), 'exact successful receive file paths')
        check(local_path(paths['wire']).read_bytes() == received_wire and local_path(paths['payload']).read_bytes() == received_payload, 'independent successful receive bytes')
        check(paths['wire_sha256'] == digest(received_wire) and paths['payload_sha256'] == digest(received_payload), 'successful receive file SHA')
        check(item['received_payload_equals_source_diagnostic_only'] == (received_payload == payload), 'noisy source comparison is diagnostic only')
    if kind == 'identity':
        check(reason is None and received_wire == wire and received_payload == payload and c['rng_before'] == c['rng_after'], 'identity exact complete source and no RNG')
    return dict(name=name, full_source_blocks=B, all_mother_syndromes_zero=True, all_coded_bits_replayed=B*n,
                all_BP_information_bits_replayed=B*k, APP_max_roundoff=max_error,
                channel_max_roundoff=float(np.max(np.abs(a['received']-reconstructed))),
                source_bit_errors=r['source_bit_errors'], erased=reason is not None, erasure_reason=reason,
                uses=N, energy=energy, receiver_source_length_oracle=False, independent_inner=inner)


def main(args):
    output, path, terminal_path = args.output.resolve(), args.manifest.resolve(), args.terminal_json.resolve()
    check(output.is_relative_to(ROOT), 'audit output within project')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(state='running', started_unix=time.time(), verifier_sha256=sha(Path(__file__)),
                  scope='full20_actual_PHY_framing_audit_no_neural_decoder_or_AP', verified=[])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    try:
        run = read(path)
        check(run['state'] == 'finished20_packets_independent_audit_pending' and run['current_packet'] is None, 'complete packet manifest')
        result['terminal_evidence'] = terminal_check(terminal_path, path, run)
        check(run['protocol_sha256'] == sha(ROOT/PROTOCOL) == PROTOCOL_SHA, 'locked protocol')
        check(run['stageB_audit_sha256'] == sha(ROOT/'data/provenance/ecsic-entropy-stageB-002-audit.json') == AUDIT_SHA, 'stageB terminal audit identity')
        check(run['stageB_manifest_sha256'] == sha(ROOT/'data/engineering/ecsic-entropy-stageB-002/manifest.json') == MANIFEST_SHA, 'stageB source manifest identity')
        check(run['sources_before'] == run['sources_after'], 'sealed source before/after')
        for p, h in run['sources_before'].items():
            check(sha(local_path(p)) == h, 'sealed source changed: '+p)
        check(run['legacy_transceiver_trimmed_bytes_discarded'] and not run['source_length_side_channel'], 'no transmitter length receiver')
        check(run['global_numpy_torch_rng_unchanged'] and not run['neural_decoder_executed'] and not run['KITTI_AP_measured'], 'original run declared scope')
        folder = path.with_suffix('')
        actual = {str(p.relative_to(ROOT)): sha(p) for p in folder.iterdir() if p.is_file()}
        check(actual == run['artifact_sha256'], 'complete actual artifact set including all failures')
        result['runtime'] = runtime_check(run)
        result['local_fixture_audit'] = audit_local_fixtures(run)
        import torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        from sionna.phy.mapping import qam
        torch.set_num_threads(4)
        np_state, torch_state = copy.deepcopy(np.random.get_state()), torch.random.get_rng_state().clone()
        expected, configurations = [], {}
        for source, size, source_sha in SOURCES:
            for config, k, n, bps in CONFIGURATIONS:
                for kind, snr in SETTINGS:
                    expected.append((source, size, source_sha, config, k, n, bps, len(expected), kind, snr))
        check(len(run['packets']) == len(run['packet_plan']) == len(expected) == 20, 'no packet subsampling')
        for item, entry, planned in zip(run['packets'], expected, run['packet_plan']):
            config, k, n, bps = entry[3:7]
            if config not in configurations:
                encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=bps, precision='double', device='cpu')
                decoder = LDPC5GDecoder(encoder, cn_update='boxplus-phi', vn_update='sum', cn_schedule='flooding',
                                        hard_out=True, return_infobits=True, num_iter=20, llr_max=20.0,
                                        prune_pcm=True, precision='double', device='cpu')
                configurations[config] = (encoder, decoder, qam(bps, normalize=True, precision='double'))
            check(all(item[key] == value for key, value in planned.items()), 'manifest plan/packet identity')
            result['verified'].append(audit_packet(item, entry, folder, *configurations[config], torch))
            check(all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state())) and
                  torch.equal(torch_state, torch.random.get_rng_state()), 'independent full replay changed global RNG')
            save(output, result)
            print(json.dumps(dict(packet=item['name'], independent_audit='passed')), flush=True)
        erased = sum(row['erased'] for row in result['verified'])
        check(run['received_packets'] == 20-erased and run['erased_packets'] == erased, 'complete denominator')
        runtime_check(run)
        check(sha(path) == read(terminal_path)['fresh_file_sha256'][str(path.relative_to(ROOT))], 'manifest unchanged during audit')
        for p, h in {**run['sources_before'], **run['artifact_sha256']}.items():
            check(sha(local_path(p)) == h, 'evidence changed during audit')
        result.update(state='passed_all20_independent_actual_PHY_and_P6EC_frame_audits', manifest_sha256=sha(path),
                      terminal_sha256=sha(terminal_path), artifact_count=len(actual), packets=20,
                      received_packets=20-erased, erased_packets=erased, replay_global_numpy_torch_rng_unchanged=True,
                      original_global_RNG_evidence='sealed runtime per-packet equality assertions; original state bytes were not serialized',
                      actual_terminal=True, neural_reconstruction_audited=False, KITTI_AP_measured=False)
    except BaseException:
        result.update(state='failed', traceback=traceback.format_exc())
        raise
    finally:
        result['finished_unix'] = time.time()
        save(output, result)
    print(json.dumps({key: result[key] for key in ('state', 'packets', 'received_packets', 'erased_packets')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/engineering/artemis-ecsic-digital-CPU-001.json')
    parser.add_argument('--terminal-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
