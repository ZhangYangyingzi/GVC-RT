"""Recreate fixed baseline crops from persisted streams when v1 previews are absent."""
import os,sys,json,io,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='7'
sys.path.insert(0,str(P))
import train
train.setup_path()
import torch,evaluate
from core import unit
from src.utils import stream_helper as stream

def main():
    torch.set_num_threads(2)
    provenance=json.loads((P/'preview_provenance.json').read_text()) if (P/'preview_provenance.json').exists() else []
    for branch in ['A','H00']:
        for source_id in ['UVal0','UVal1','ParkScene','Jockey']:
            key=f'{branch}_s300_{source_id}_q2';dest=P/'previews'/f'{key}_frame15.png'
            if dest.exists():continue
            metadata=json.loads((P/'server_artifacts/evaluation'/f'{key}.json').read_text())
            assert train.sha(metadata['stream_path'])==metadata['stream_sha256']
            source=next(x for x in json.loads((P/'evaluation_sources.json').read_text()) if x['id']==source_id)
            im,pm,digest=evaluate.load_checkpoint(train,branch,300)
            assert digest==metadata['checkpoint_sha256']
            buffer=io.BytesIO(Path(metadata['stream_path']).read_bytes());helper=stream.SPSHelper()
            pm.clear_dpb();pm.set_curr_poc(0)
            with torch.inference_mode():
                for i in range(16):
                    head=stream.read_header(buffer)
                    while head['nal_type']==stream.NalType.NAL_SPS:
                        helper.add_sps_by_id(stream.read_sps_remaining(buffer,head['sps_id']));head=stream.read_header(buffer)
                    sps=helper.get_sps_by_id(head['sps_id']);q,payload=stream.read_ip_remaining(buffer)
                    if i==0:
                        im.set_use_two_entropy_coders(bool(sps['ec_part']));pm.set_use_two_entropy_coders(bool(sps['ec_part']))
                        decoded=im.decompress(payload,sps,q);pm.add_ref_frame(None,decoded['x_hat'])
                    else:decoded=pm.decompress(payload,sps,q)
                output=unit(decoded['x_hat'],source['height'],source['width'])[:,:,200:456,400:656]
                evaluate.save_preview(dest,output)
            provenance.append({'branch':branch,'sequence':source_id,'frame':15,'crop':[200,456,400,656],'stream_sha256':metadata['stream_sha256'],'checkpoint_sha256':digest,'preview_sha256':train.sha(dest),'measurement':'new_preview_decode_from_existing_measured_stream'})
            del im,pm;torch.cuda.empty_cache()
    (P/'preview_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
