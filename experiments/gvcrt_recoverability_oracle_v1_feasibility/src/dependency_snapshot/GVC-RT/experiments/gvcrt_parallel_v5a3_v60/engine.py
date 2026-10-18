"""Evaluation-only dynamic-resolution codec. Frozen public model and metric modules."""
import io,math,sys
import numpy as np
import torch
from parallel_utils import *
sys.path.insert(0,str(V41))
import metric_runtime as metrics
core,eco=metrics.core,metrics.eco
from src.layers.cuda_inference import replicate_pad

class Runtime:
    def __init__(self,config,method,device):
        self.config,self.method,self.device=config,method,device
        sender,receiver=config['methods'][method]
        self.sender,self.receiver=sender,receiver
        self.wrapper=None;self.payload=None
        if sender:
            _,self.wrapper=eco.load_joint(config['checkpoints'][sender]['path'],device)
            assert core.module_hash(self.wrapper)==config['checkpoints'][sender]['module_hashes']['wrapper']
        if receiver:
            self.payload=torch.load(config['checkpoints'][receiver]['path'],map_location='cpu',weights_only=True)
        self.quality=core.quality_models(device)
        self.metric_models=metrics.load_metrics(device)
    def models(self,h,w):
        im,pm=core.load_models(self.device,force_zero_thres=self.config['force_zero_thres'])
        if self.payload is not None:eco.apply_joint(pm,self.payload)
        for model in (im,pm):
            model.set_use_two_entropy_coders(h*w>1280*720)
            assert all(not p.requires_grad for p in model.parameters())
        assert im.get_qp_num()==pm.get_qp_num()==10
        self.loaded_receiver_hashes={'bridge':core.module_hash(pm.recon_generation_net.mlp),'generator':core.module_hash(pm.recon_generation_net.decoder)}
        if self.payload is not None:
            for key,component in (('bridge',pm.recon_generation_net.mlp),('generator',pm.recon_generation_net.decoder)):
                expected={k:v.to(dtype=component.state_dict()[k].dtype) for k,v in self.payload[key].items()}
                assert self.loaded_receiver_hashes[key]==tensor_dict_hash(expected)
        return im,pm
    @torch.inference_mode()
    def run(self,frames,qp,stream_path,feature_path,recon_dir=None):
        h,w=frames[0].shape[-2:];ch,cw=(h+63)//64*64,(w+63)//64*64
        n=len(frames);sps={'sps_id':0,'height':ch,'width':cw,'ec_part':int(ch*cw>1280*720),'use_ada_i':0}
        ie,pe=self.models(ch,cw);idc,pdc=self.models(ch,cw)
        before=core.compression_hash(ie,pe)
        pe.clear_dpb();pdc.clear_dpb();pe.set_curr_poc(0);pdc.set_curr_poc(0)
        encoded_hashes=[];states=[];actual=[];proxy_max=0.0
        buffer=io.BytesIO();eco.write_sps(buffer,sps)
        for i,cpu in enumerate(frames):
            target=cpu.to(self.device);proxy=target if self.wrapper is None else self.wrapper(target)
            difference=float((proxy-target).abs().max());proxy_max=max(proxy_max,difference)
            if self.wrapper is None:assert difference==0.0
            # Exactly the public evaluator's input ordering: half -> replicate pad -> 2*x-1.
            x=replicate_pad(proxy.half(),ch-h,cw-w)*2.0-1.0
            assert x.shape[-2:]==(ch,cw)
            q=qp if i==0 else pe.shift_qp(qp,eco.INDEX_MAP[i%8]);actual.append(q)
            if i==0:
                encoded=ie.compress(x,q);pe.clear_dpb();pe.add_ref_frame(None,encoded['x_hat'])
                decoded=idc.decompress(encoded['bit_stream'],sps,q);pdc.clear_dpb();pdc.add_ref_frame(None,decoded['x_hat'])
            else:
                encoded=pe.compress(x,q);decoded=pdc.decompress(encoded['bit_stream'],sps,q)
                states.append(float((pe.dpb[0].feature-pdc.dpb[0].feature).abs().max()))
            eco.write_ip(buffer,i==0,0,q,encoded['bit_stream'])
            encoded_hashes.append(eco.tensor_sha(decoded['x_hat']))
        assert all(x==0 for x in states)
        assert core.compression_hash(ie,pe)==before
        data=buffer.getvalue();stream_path.parent.mkdir(parents=True,exist_ok=True)
        tmp=stream_path.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(stream_path)
        del ie,pe,idc,pdc
        torch.cuda.empty_cache()
        # Fresh independent decoder, using only the persisted RANS stream plus frozen receiver weights.
        ia,pa=self.models(ch,cw);pa.clear_dpb();pa.set_curr_poc(0)
        assert core.compression_hash(ia,pa)==before
        stream=io.BytesIO(stream_path.read_bytes());helper=eco.SPSHelper()
        flow,perceptual,inception=self.metric_models
        rows=[];transitions=[];refs=[];outs=[];hashes=[];previous_reference=previous_output=None
        for i,cpu in enumerate(frames):
            position=stream.tell();header=eco.read_header(stream)
            while header['nal_type']==eco.NalType.NAL_SPS:
                helper.add_sps_by_id(eco.read_sps_remaining(stream,header['sps_id']));header=eco.read_header(stream)
            current=helper.get_sps_by_id(header['sps_id']);q,bits=eco.read_ip_remaining(stream)
            assert current==sps and q==actual[i]
            assert (header['nal_type']==eco.NalType.NAL_I)==(i==0)
            if i==0:
                decoded=ia.decompress(bits,current,q);pa.clear_dpb();pa.add_ref_frame(None,decoded['x_hat'])
            else:decoded=pa.decompress(bits,current,q)
            hashes.append(eco.tensor_sha(decoded['x_hat']));assert hashes[-1]==encoded_hashes[i]
            output=core.unit(decoded['x_hat'],h,w);target=cpu.to(self.device)
            assert output.shape==target.shape==(1,3,h,w)
            spatial=eco.frame_metrics(output,target,self.quality)
            assert all(math.isfinite(float(v)) for v in spatial.values())
            rows.append(dict(frame=i,external_qp=qp,actual_i_qp=qp,actual_p_qp='' if i==0 else q,actual_qp=q,
                             real_bits=(stream.tell()-position)*8,payload_bits=len(bits)*8,**spatial))
            refs.append(inception(target).cpu().numpy());outs.append(inception(output).cpu().numpy())
            if previous_reference is not None:
                flow_diff=flow(previous_reference,target)-flow(previous_output,output)
                magnitude=flow_diff.square().sum(1,keepdim=True).sqrt().sum()
                assert torch.isfinite(magnitude) and magnitude>0,'FloLPIPS undefined motion weights'
                value=float(perceptual(previous_reference,previous_output,flow_diff,normalize=True).item())
                assert math.isfinite(value)
                transitions.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value))
            previous_reference,previous_output=target,output
            if recon_dir is not None:eco.save_png(output,recon_dir/f'frame_{i:06d}.png')
        assert stream.tell()==len(data) and len(hashes)==n and len(transitions)==n-1
        assert core.compression_hash(ia,pa)==before
        assert sum(r['real_bits'] for r in rows)==len(data)*8
        real,reconstruction=np.concatenate(refs),np.concatenate(outs)
        assert real.shape==reconstruction.shape==(n,2048)
        assert np.isfinite(real).all() and np.isfinite(reconstruction).all()
        feature_path.parent.mkdir(parents=True,exist_ok=True)
        with feature_path.with_suffix('.tmp').open('wb') as f:np.savez(f,real=real,reconstruction=reconstruction)
        feature_path.with_suffix('.tmp').replace(feature_path)
        write(feature_path.with_suffix('.transitions.csv'),transitions)
        fps=20.0 if self.config['experiment']=='A' else 30.0
        mse=sum(r['pixel_SSE'] for r in rows)/(n*h*w*3)
        result=dict(real_bytes=len(data),real_RANS_bytes=len(data),bits_per_frame=len(data)*8/n,bpp=len(data)*8/(n*h*w),
             kbps=len(data)*8*fps/n/1000,rate_accounting_fps=fps,frames=n,num_frames=n,num_transitions=n-1,
             **{k:sum(r[k] for r in rows)/n for k in SPATIAL},FloLPIPS=sum(r['FloLPIPS'] for r in transitions)/(n-1),
             PSNR_from_aggregate_MSE=-10*math.log10(max(mse,1e-15)),
             processing_canvas=[cw,ch],metric_crop=[w,h],ec_part=sps['ec_part'],
             identity_P=self.wrapper is None,identity_P_exact=proxy_max==0 if self.wrapper is None else None,
             proxy_max_abs=proxy_max,compression_hash_before=before,compression_hash_after=before,
             reconstruction_sha256=eco.sha256_bytes(''.join(hashes).encode()),
             independent_decode_pass=True,state_sync_pass=True,real_RANS=True,bytes_consumed=stream.tell(),
             bitstream_path=str(stream_path),bitstream_sha256=sha(stream_path),
             feature_path=str(feature_path),feature_sha256=sha(feature_path))
        if self.config['experiment']=='A':result['kbps_20fps']=result['kbps']
        else:result['PSNR']=result['PSNR_from_aggregate_MSE']
        result['loaded_receiver_module_hashes']=self.loaded_receiver_hashes
        del ia,pa
        torch.cuda.empty_cache()
        return result,rows
