"""Independent post-hoc receiver decode in the ORIGINAL CPU environment."""
import hashlib,io,json,os,platform,subprocess,time,zlib
from pathlib import Path
import numpy as np
import PIL
from PIL import Image,features

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output=ROOT/'data/engineering/artemis-digital-framing-pixel-supplement-001.json';folder=output.with_suffix('')
    assert not output.exists() and not folder.exists() and os.environ.get('SLURM_JOB_ID')
    path=ROOT/'data/engineering/artemis-digital-framing-CPU-001.json';d=json.loads(path.read_text())
    assert d['state']=='passed_actual_receiver_framing_audit_pending'
    freeze=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True)
    assert freeze==d['pip_freeze_after'];folder.mkdir()
    r=dict(state='running',scope='posthoc_exploratory_independent_received_pixel_decode',job_id=os.environ['SLURM_JOB_ID'],
           source_manifest_sha256=sha(path),code_sha256=sha(Path(__file__)),started_unix=time.time(),
           protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/framing-pixel-supplement-001.md'),
           Pillow=PIL.__version__,OpenJPEG=features.version_codec('jpg_2000'),machine=platform.machine(),views=[])
    try:
        for item in d['packets']:
            if item['reception']['state']!='received':continue
            fixture=ROOT/item['arrays_path'];assert sha(fixture)==d['artifact_sha256'][item['arrays_path']]
            with np.load(fixture,allow_pickle=False) as a:bits=a['decoded'].ravel()
            assert np.isin(bits,[0,1]).all()
            header=np.packbits(bits[:160],bitorder='big').tobytes()
            assert header[:8]==b'P6SB\x01\x02\x00\x00'
            leftsize=int.from_bytes(header[8:12],'big');rightsize=int.from_bytes(header[12:16],'big')
            size=20+leftsize+rightsize;k=item['diagnostic']['identity']['k']
            assert leftsize>0 and rightsize>0 and (size*8+k-1)//k==len(bits)//k and size*8<=len(bits)
            assert not bits[size*8:].any()
            wire=np.packbits(bits[:size*8],bitorder='big').tobytes()
            assert zlib.crc32(wire[20:])==int.from_bytes(header[16:20],'big')
            originalwire=ROOT/'data/engineering/artemis-digital-framing-CPU-001'/(item['name']+'.p6sb')
            assert wire==originalwire.read_bytes()
            images=[]
            for stream in (wire[20:20+leftsize],wire[20+leftsize:]):
                with Image.open(io.BytesIO(stream)) as im:
                    assert im.format=='JPEG2000' and im.mode=='RGB';images.append(np.array(im,dtype=np.uint8))
            assert images[0].shape==images[1].shape
            hashes=[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in images]
            assert hashes==item['reception']['received_image_sha256']
            target=folder/(item['name']+'.npz');np.savez_compressed(target,left=images[0],right=images[1])
            r['views'].append(dict(packet=item['name'],fixture_sha256=sha(fixture),wire_sha256=sha(originalwire),
                                  image_sha256=hashes,shape=list(images[0].shape),pixel_arrays_path=str(target.relative_to(ROOT)),
                                  pixel_arrays_sha256=sha(target)))
        assert len(r['views'])==6 and sha(path)==r['source_manifest_sha256'] and sha(Path(__file__))==r['code_sha256']
        assert subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True)==freeze
        r['state']='passed_all6_independent_on_platform_pixel_redecodes'
    except Exception as exc:r.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['finished_unix']=time.time()
        with output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({'state':r['state'],'packets':len(r['views'])}))


if __name__=='__main__':main()
