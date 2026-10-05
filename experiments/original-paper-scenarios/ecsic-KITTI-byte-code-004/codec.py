"""Model-bound P6EKv1; reuse immutable previously audited entropy arithmetic."""
import hashlib,importlib.util,re,struct,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
LEGACY=ROOT/'experiments/original-paper-scenarios/ecsic-entropy-code/codec.py'
assert hashlib.sha256(LEGACY.read_bytes()).hexdigest()=='a22fb7717cd1e2a13e4539a74a541958e93a7f6b215ae960b9c2736ad53ec157'
spec=importlib.util.spec_from_file_location('frozen_ecsic_entropy_arithmetic004',LEGACY);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
BASE=struct.Struct('>4sBB4H32s32s32s');DESC=struct.Struct('>BIII');ORDER=old.ORDER;LIMIT=old.LIMIT;OVERHEAD=BASE.size+4*DESC.size+4
CONFIG_SHA=old.CONFIG_SHA;CDF_SHA='507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
require=old.require;dimensions=old.dimensions;shape_of=old.shape_of;encode_stream=old.encode_stream;decode_stream=old.decode_stream;scale_ids=old.scale_ids

def tables():
 rows,digest=old.tables();require(digest==CDF_SHA,'frozen CDF identity');return rows,digest

def identity(value):
 require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'canonical model SHA256');return bytes.fromhex(value)

def pack(original,padded,streams,model_sha):
 dimensions(original,padded);model=identity(model_sha);require(list(streams)==list(ORDER),'four-stream order')
 header=bytearray(BASE.pack(b'P6EK',1,4,*original,*padded,model,bytes.fromhex(CONFIG_SHA),bytes.fromhex(CDF_SHA)));body=bytearray()
 for index,name in enumerate(ORDER):
  rans,escapes=streams[name];require(type(rans) is bytes and type(escapes) is bytes,'immutable stream bytes')
  count=int(np.prod(shape_of(index,padded)));require(len(rans)>=4 and len(escapes)<=4*count and len(rans)+len(escapes)<=LIMIT,'stream length bound')
  header.extend(DESC.pack(index,count,len(rans),len(escapes)));body.extend(rans);body.extend(escapes)
 out=bytes(header+body);require(len(out)+4<=LIMIT,'container length bound');return out+struct.pack('>I',zlib.crc32(out))

def unpack(blob,expected_model_sha):
 expected=identity(expected_model_sha);require(type(blob) is bytes and OVERHEAD+16<=len(blob)<=LIMIT,'container length bound')
 require(zlib.crc32(blob[:-4])==struct.unpack('>I',blob[-4:])[0],'container CRC')
 magic,version,count,h,w,ph,pw,model,config,cdf=BASE.unpack_from(blob)
 require((magic,version,count)==(b'P6EK',1,4),'model-bound container format')
 require((model,config.hex(),cdf.hex())==(expected,CONFIG_SHA,CDF_SHA),'received model/config/CDF identity')
 dimensions([h,w],[ph,pw]);offset=BASE.size+4*DESC.size;streams={}
 for index,name in enumerate(ORDER):
  sid,count,nr,ne=DESC.unpack_from(blob,BASE.size+index*DESC.size);wanted=int(np.prod(shape_of(index,[ph,pw])))
  require(sid==index and count==wanted,'stream id/count');require(nr>=4 and ne<=4*wanted and offset+nr+ne<=len(blob)-4,'stream lengths')
  streams[name]=(blob[offset:offset+nr],blob[offset+nr:offset+nr+ne]);offset+=nr+ne
 require(offset==len(blob)-4,'unused container bytes')
 return dict(original_hw=[h,w],padded_hw=[ph,pw],model_sha256=model.hex(),config_sha256=config.hex(),CDF_sha256=cdf.hex(),streams=streams,total_bytes=len(blob),overhead_bytes=OVERHEAD)
