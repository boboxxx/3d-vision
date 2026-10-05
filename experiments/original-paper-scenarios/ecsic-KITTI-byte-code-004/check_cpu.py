"""Real sealed wire fixtures; independent header layout and foreign-weight tests."""
import argparse,hashlib,json,struct,time,zlib
from pathlib import Path
import codec as c
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PUBLIC='e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();rows=[];rejected=[]
 before={str(v.relative_to(ROOT)):sha(v) for v in [c.LEGACY,HERE/'codec.py',HERE/'check_cpu.py',ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-byte-protocol-004.md']};assert c.OVERHEAD==166
 def reject(name,call):
  try:call()
  except ValueError:rejected.append(name)
  else:raise AssertionError('accepted '+name)
 for case in ['synthetic32x64','training000000']:
  path=ROOT/f'data/engineering/ecsic-entropy-stageB-002/{case}/source.p6ec';legacy=path.read_bytes();old=c.old.unpack(legacy,c.CDF_SHA);blob=c.pack(old['original_hw'],old['padded_hw'],old['streams'],PUBLIC);parsed=c.unpack(blob,PUBLIC)
  assert len(blob)==len(legacy) and blob[162:-4]==legacy[162:-4] and parsed['streams']==old['streams'] and parsed['original_hw']==old['original_hw'] and parsed['padded_hw']==old['padded_hw']
  assert blob[:6]==b'P6EK\x01\x04' and blob[14:46].hex()==PUBLIC and blob[46:78].hex()==c.CONFIG_SHA and blob[78:110].hex()==c.CDF_SHA
  assert list(struct.unpack('>4H',blob[6:14]))==old['original_hw']+old['padded_hw'];offset=162;descriptors=[]
  for i,name in enumerate(c.ORDER):
   desc=struct.unpack('>BIII',blob[110+13*i:123+13*i]);h,w=old['padded_hw'];divisor=32 if i<2 else 8;n=48*(h//divisor)*(w//divisor);rb,eb=old['streams'][name]
   assert desc==(i,n,len(rb),len(eb)) and blob[offset:offset+len(rb)]==rb and blob[offset+len(rb):offset+len(rb)+len(eb)]==eb;offset+=len(rb)+len(eb);descriptors.append(list(desc))
  assert offset==len(blob)-4 and struct.unpack('>I',blob[-4:])[0]==zlib.crc32(blob[:-4])
  foreign=hashlib.sha256(b'format-fixture-only-not-an-actual-trained-model').hexdigest();different=c.pack(old['original_hw'],old['padded_hw'],old['streams'],foreign);assert c.unpack(different,foreign)['streams']==old['streams'] and different[162:-4]==blob[162:-4]
  reject(case+'_cross_expected_model',lambda:c.unpack(blob,foreign));reject(case+'_foreign_wire_model',lambda:c.unpack(different,PUBLIC));reject(case+'_legacy_magic',lambda:c.unpack(legacy,PUBLIC))
  for field,position,value in [('magic',0,b'FAIL'),('version',4,b'\x02'),('count',5,b'\x03'),('weight',14,b'\0'*32),('config',46,b'\0'*32),('CDF',78,b'\0'*32),('height',6,b'\0\0'),('stream_id',110,b'\x03'),('symbol_count',111,b'\0'*4),('rans_length',115,b'\0'*4),('escape_length',119,b'\xff'*4)]:
   b=bytearray(blob[:-4]);b[position:position+len(value)]=value;bad=bytes(b)+struct.pack('>I',zlib.crc32(b));reject(case+'_valid_CRC_'+field,lambda bad=bad:c.unpack(bad,PUBLIC))
  for size in [0,1,165,181,len(blob)-1]:reject(case+'_truncated_'+str(size),lambda size=size:c.unpack(blob[:size],PUBLIC))
  reject(case+'_trailing_byte',lambda:c.unpack(blob+b'\0',PUBLIC))
  for pos in [0,14,45,78,109,110,161,162,len(blob)-5,len(blob)-1]:
   b=bytearray(blob);b[pos]^=1;reject(case+'_bitflip_'+str(pos),lambda b=b:c.unpack(bytes(b),PUBLIC))
  rows.append(dict(case=case,original_container_sha256=sha(path),new_container_sha256=hashlib.sha256(blob).hexdigest(),container_bytes=len(blob),header_CRC_bytes=166,unchanged_stream_body_sha256=hashlib.sha256(blob[162:-4]).hexdigest(),descriptors=descriptors,alternate_model_binding_sha256=foreign))
 for bad in ['',PUBLIC.upper(),PUBLIC[:-1],'g'*64,None]:reject('noncanonical_model_'+repr(bad),lambda bad=bad:c.identity(bad))
 after={k:sha(ROOT/k) for k in before};assert before==after
 result=dict(state='passed_two_actual_container_model_binding_and_independent_header_gates',checked_unix=time.time(),sources=before,CDF_sha256=c.tables()[1],fixtures=rows,rejected_cases=rejected,actual_candidate_final_weight_or_calibration=False,neural_reception_executed=False,limitation='Pure framing/header/identity gate on two already decoded public-model fixtures; no adapted-model neural decode, calibration, PHY or AP.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state=result['state'],fixtures=rows,rejections=len(rejected))))
if __name__=='__main__':main()
