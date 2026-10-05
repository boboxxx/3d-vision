"""Narrow full-byte native SRCNN blob stream with one-member memory residency."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tarfile
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
MAX_MEMBER_BYTES=64*2**20
NATIVE_PATTERN=re.compile(r'/mnt/d/paper6/runs/srcnn-source-cache-(engineering|main)-001/cr(10|30|50)/(wire|received)/([0-9]{6})\.(p6sr|npz)\Z')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
     '-o','ServerAliveCountMax=3','-o','ControlPath=/private/tmp/paper6-sheng-01a10237.sock','sheng@100.94.183.27']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def consume(expected, scope, callback):
    assert scope in ('engineering','main') and expected
    for name,digest in expected.items():
        match=NATIVE_PATTERN.fullmatch(name)
        assert match and match[1]==scope and re.fullmatch('[a-f0-9]{64}',digest)
        assert (match[3],match[5]) in (('wire','p6sr'),('received','npz'))
    ordered=sorted(expected); filelist=''.join(name.lstrip('/')+'\n' for name in ordered).encode('ascii')
    # Concurrent input/output is essential: the full file list exceeds pipe
    # capacity while tar may emit a large member before reading its next name.
    remote='cd / && tar --create --file=- --no-recursion --verbatim-files-from --files-from=-'
    process=subprocess.Popen(SSH+[remote],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    producer_errors=[]; stderr=bytearray(); stderr_overflow=[False]
    def feed():
        try:
            process.stdin.write(filelist);process.stdin.flush()
        except BaseException as error:producer_errors.append(repr(error))
        finally:process.stdin.close()
    def drain():
        while True:
            chunk=process.stderr.read(4096)
            if not chunk:break
            if len(stderr)+len(chunk)>65536:stderr_overflow[0]=True
            if len(stderr)<65536:stderr.extend(chunk[:65536-len(stderr)])
    writer=threading.Thread(target=feed,daemon=True); errors=threading.Thread(target=drain,daemon=True)
    writer.start();errors.start();rows=[];seen=set()
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            for member in archive:
                name='/'+member.name
                assert member.isfile() and name in expected and name not in seen
                assert 0<member.size<=MAX_MEMBER_BYTES
                stream=archive.extractfile(member);assert stream is not None
                blob=stream.read(member.size+1);assert len(blob)==member.size
                digest=hashlib.sha256(blob).hexdigest();assert digest==expected[name],name
                callback(name,blob)
                rows.append(dict(native_path=name,bytes=member.size,sha256=digest));seen.add(name)
                del blob
        writer.join(timeout=10);assert not writer.is_alive() and not producer_errors,producer_errors
        status=process.wait(timeout=60);errors.join(timeout=10)
        assert not errors.is_alive() and not stderr_overflow[0]
        assert status==0,stderr.decode(errors='replace')
        assert seen==set(expected) and len(rows)==len(expected)
        return rows
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=10)
        process.stdout.close();writer.join(timeout=10);errors.join(timeout=10);process.stderr.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--engineering-wires',action='store_true',required=True)
    parser.parse_args();output=ROOT/'data/provenance/srcnn-source-cache-engineering-001-streamed-byte-check-001.json'
    assert not output.exists()
    spec=importlib.util.spec_from_file_location('streamed_cache_parser',ROOT/'data/provenance/verify-srcnn-native-cache-local-001.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    audit_path=ROOT/'data/provenance/srcnn-source-cache-engineering-001-wire-audit.json'
    proof_path=ROOT/'data/provenance/srcnn-source-cache-engineering-001-wire-local-verification.json'
    audit,proof=[json.loads(path.read_text()) for path in (audit_path,proof_path)]
    assert audit['state']=='passed_all6_native_SRCNN_engineering_wires_actual_encoder_terminal' and audit['actual_encoder_terminal']
    assert proof['state']=='passed_all6_actual_transferred_SRCNN_engineering_wire_files_and8_artifacts'
    assert proof['native_wire_audit_sha256']==sha(audit_path)
    expected=audit['wire_files_sha256'];assert len(expected)==6
    def check(name,blob):
        retained=ROOT/'data/engineering/srcnn-source-cache-engineering-001-wire-transfer'/name.lstrip('/')
        assert blob==retained.read_bytes()
        rate,*_=validator.literal_header(blob);assert '/cr'+str(rate)+'/wire/' in name
    rows=consume(expected,'engineering',check)
    assert sum(row['bytes'] for row in rows)==sum(item['wire_bytes'] for item in audit['summaries'].values())
    result=dict(state='passed_all6_actual_streamed_SRCNN_wire_bytes_without_retained_stream_blobs',
        checked_unix=time.time(),wire_files=6,wire_bytes=sum(row['bytes'] for row in rows),rows=rows,
        source_wire_proof_sha256=sha(proof_path),native_wire_audit_sha256=sha(audit_path),
        helper_sha256=sha(__file__),storage_prelock_sha256=sha(ROOT/'experiments/original-paper-scenarios/srcnn-main-streamed-artifact-acceptance-002.md'),
        retained_stream_blob_files=0,maximum_allowed_member_bytes=MAX_MEMBER_BYTES,
        limitation='Actual complete source wires only; no streamed NPZ, GPU reconstruction, full main acceptance or detection')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],wire_bytes=result['wire_bytes'],retained_stream_blob_files=0)))


if __name__=='__main__':main()
