"""Postflight checks against saved bytes, tensors and independent decoder receipts."""
from base import *
import math
import struct
import shutil

def validate():
 import numpy as np
 records=[];errors=[]
 for path in sorted((ROOT/'manifests/receipts').glob('*.json')):
  r=load(path)
  if r['status']!='PASS':continue
  try:
   j=r['job'];x=r['result'];rate=r['rate'];p=Path(x['bitstream_path']);raw=p.read_bytes()
   assert sha(p)==x['bitstream_sha256']
   count=struct.unpack('>H',raw[8:10])[0];header=10+11*count
   assert raw[:8]==b'GROV1\0\0\0' and count==len(j['plan'])
   native=p.with_name('native.bin')
   assert native.read_bytes()==raw[header:] and sha(native)==x['native_bitstream_sha256']
   assert len(raw)*8==rate['actual_total_bits']==sum(rate[k] for k in ('RANS_payload_bits','native_SPS_IP_header_bits','control_header_bits'))
   assert len(raw)==x['real_bytes'] and rate['control_header_bits']==header*8
   assert x['frames']==16 and x['num_transitions']==15 and len(x['actual_qps'])==16
   assert math.isclose(x['bpp'],len(raw)*8/(16*1920*1080),rel_tol=1e-14)
   assert x['bytes_consumed']==len(raw)
   assert x['independent_decode_pass'] and x['state_sync_pass'] and r['reference_state_exact_all_P_frames']
   assert len(r['decoder_symbol_checks'])==4 and all(g['exact'] for f in r['decoder_symbol_checks'] for g in f['groups'])
   changes=[]
   for frame,info in r['tensor_artifacts'].items():
    assert sha(info['path'])==info['sha256']
    with np.load(info['path']) as z:
     assert z['y'].shape==z['symbols'].shape==z['original_q'].shape==(1,128,68,120)
     assert z['packed_0'].dtype==z['packed_1'].dtype==np.int8
     assert np.isfinite(z['y']).all() and np.isfinite(z['effective_q']).all()
     assert (z['effective_q']>=.5).all()
     for a in j['plan']:
      if a['frame']!=int(frame):continue
      roi=(slice(None),slice(None),slice(a['y'],a['y']+a['h']),slice(a['x'],a['x']+a['w']))
      changes.append(dict(frame=int(frame),factor=a['factor'],effective_step_changed_fraction=float((z['original_q'][roi]!=z['effective_q'][roi]).mean())))
   transitions=Path(x['feature_path']).with_suffix('.transitions.csv')
   with transitions.open() as f:ts=list(csv.DictReader(f))
   assert len(ts)==15 and math.isclose(sum(float(t['FloLPIPS']) for t in ts)/15,x['FloLPIPS'],rel_tol=1e-12)
   target=ROOT/'metrics/per_transition'/f'{j["id"]}.csv';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(transitions,target)
   records.append(dict(id=j['id'],status='PASS',actual_bits=len(raw)*8,native_bits=native.stat().st_size*8,effective_step_changes=changes))
  except Exception:
   import traceback
   errors.append(dict(id=path.stem,traceback=traceback.format_exc()))
 dependencies=load(ROOT/'manifests/dependency_sources.json')['records']
 source_changed=[d['original_path'] for d in dependencies if sha(d['original_path'])!=d['sha256'] or sha(ROOT/d['snapshot'])!=d['sha256']]
 weights=load(ROOT/'manifests/metric_weights.json')
 metric_weights_changed=[p for p,v in weights.items() if sha(p)!=v['sha256']]
 report=dict(status='PASS' if not errors and not source_changed and not metric_weights_changed else 'FAILED',validated_receipts=len(records),records=records,errors=errors,dependency_source_changes=source_changed,metric_weight_changes=metric_weights_changed,scope='Actual persisted stream bytes, source and tensor checksums, tensor domains, full causal independent decode receipts, temporal metric aggregation')
 dump('manifests/artifact_validation.json',report)
 assert report['status']=='PASS',report
 return report

if __name__=='__main__':print(validate()['status'])
