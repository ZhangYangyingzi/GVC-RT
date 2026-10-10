"""Independent periodic-native-I RANS adapter; immutable historical implementation."""
import inspect,textwrap,types
from io21 import *
def frames_for(v):
    old=module('v621_readonly_frames',V20/'adapter.py')
    return old.frames_for(v)
def cache_key(v,q,m,ip):
    cfg=evalcfg()
    key=dict(method=m,IP=ip,QP_phase='distance_from_most_recent_I',I_mode='native_RGB',checkpoint_sha256='' if m=='original' else cfg['checkpoints'][m]['sha256'],modules=cfg['original_receiver_hashes'] if m=='original' else cfg['checkpoints'][m]['module_hashes'],RGB=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],external_qp=q,protocol_sha256=sha(ROOT/'protocol.json'),codec_dependencies_sha256=sha(ROOT/'audits/dependencies.json'),metric_module_hashes=cfg['metric_module_hashes'])
    key['adapter_sha256']=sha(ROOT/'adapter.py')
    key['QP_audit_sha256']=sha(ROOT/'qp_semantics_audit.json')
    return hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
def engine():
    sys.path.insert(0,str(ENGINE));base=module('gop21_engine',ENGINE/'engine.py')
    source=textwrap.dedent(inspect.getsource(base.Runtime.run))
    def change(a,b):
        nonlocal source
        assert source.count(a)==1,(a,source.count(a))
        source=source.replace(a,b)
    change("target=cpu.to(self.device);proxy=target if self.wrapper is None else self.wrapper(target)","is_i=i==0 or (self.ip>0 and i%self.ip==0)\n        if is_i:last_i=i\n        target=cpu.to(self.device);proxy=target if self.wrapper is None or is_i else self.wrapper(target)")
    change("q=qp if i==0 else pe.shift_qp(qp,eco.INDEX_MAP[i%8]);actual.append(q)","q=qp if is_i else pe.shift_qp(qp,eco.INDEX_MAP[(i-last_i)%8]);actual.append(q)")
    change("encoded=ie.compress(x,q);pe.clear_dpb();pe.add_ref_frame(None,encoded['x_hat'])\n            decoded=idc.decompress(encoded['bit_stream'],sps,q);pdc.clear_dpb();pdc.add_ref_frame(None,decoded['x_hat'])","encoded=ie.compress(x,q);decoded=idc.decompress(encoded['bit_stream'],sps,q)\n            assert eco.tensor_sha(encoded['x_hat'])==eco.tensor_sha(decoded['x_hat'])\n            pe.clear_dpb();pdc.clear_dpb();pe.set_curr_poc(0);pdc.set_curr_poc(0)\n            pe.add_ref_frame(None,decoded['x_hat']);pdc.add_ref_frame(None,decoded['x_hat'])\n            refreshes.append(dict(frame=i,payload_sha256=eco.sha256_bytes(encoded['bit_stream']),native_I_sha256=eco.tensor_sha(decoded['x_hat']),encoder_reference_sha256=eco.tensor_sha(pe.dpb[0].frame),decoder_reference_sha256=eco.tensor_sha(pdc.dpb[0].frame),POC_reset=0))")
    change("encoded_hashes=[];states=[];actual=[];proxy_max=0.0","encoded_hashes=[];states=[];actual=[];proxy_max=0.0;refreshes=[]")
    change("eco.write_ip(buffer,i==0,0,q,encoded['bit_stream'])","eco.write_ip(buffer,is_i,0,q,encoded['bit_stream'])")
    change("position=stream.tell();header=eco.read_header(stream)","is_i=i==0 or (self.ip>0 and i%self.ip==0)\n        if is_i:last_i=i\n        position=stream.tell();header=eco.read_header(stream)")
    change("assert (header['nal_type']==eco.NalType.NAL_I)==(i==0)","assert (header['nal_type']==eco.NalType.NAL_I)==is_i")
    change("decoded=ia.decompress(bits,current,q);pa.clear_dpb();pa.add_ref_frame(None,decoded['x_hat'])","decoded=ia.decompress(bits,current,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,decoded['x_hat'])\n            item=next(r for r in refreshes if r['frame']==i)\n            item['independent_reference_sha256']=eco.tensor_sha(pa.dpb[0].frame)\n            assert len(set(item[k] for k in ('native_I_sha256','encoder_reference_sha256','decoder_reference_sha256','independent_reference_sha256')))==1")
    source=source.replace("if i==0:","if is_i:").replace("'' if i==0 else q","'' if is_i else q")
    change("real_bits=(stream.tell()-position)*8,payload_bits=len(bits)*8,**spatial)","real_bits=(stream.tell()-position)*8,payload_bits=len(bits)*8,frame_type='I' if is_i else 'P',reference_age=i-last_i,IP=self.ip,NAL_type=int(header['nal_type']),**spatial)")
    change("transitions.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value))","transitions.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value,IP=self.ip,cross_I_refresh=is_i))")
    change("del ie,pe,idc,pdc","self.assert_frozen(ie,pe);self.assert_frozen(idc,pdc)\n    del ie,pe,idc,pdc")
    change("del ia,pa","self.assert_frozen(ia,pa)\n    del ia,pa")
    change("result['loaded_receiver_module_hashes']=self.loaded_receiver_hashes","result['loaded_receiver_module_hashes']=self.loaded_receiver_hashes\n    result.update(IP=self.ip,I_refresh_audits=refreshes,I_frames=sum(r['frame_type']=='I' for r in rows),P_frames=sum(r['frame_type']=='P' for r in rows),I_bits_including_associated_headers=sum(r['real_bits'] for r in rows if r['frame_type']=='I'),P_bits_including_headers=sum(r['real_bits'] for r in rows if r['frame_type']=='P'),I_payload_bits=sum(r['payload_bits'] for r in rows if r['frame_type']=='I'),P_payload_bits=sum(r['payload_bits'] for r in rows if r['frame_type']=='P'),header_and_SPS_bits=len(data)*8-sum(r['payload_bits'] for r in rows))")
    ns=dict(base.__dict__);exec(compile(source,str(ROOT/'adapter.py'),'exec'),ns);run=ns['run']
    class Runtime(base.Runtime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.ip=self.config.get('IP',-1);assert self.ip in IPS
            if self.wrapper is not None:self.wrapper.eval().requires_grad_(False)
            self.metric_hashes=dict(quality=[base.core.module_hash(x) for x in self.quality],flow_perceptual_inception=[base.core.module_hash(x) for x in self.metric_models]);assert self.metric_hashes==evalcfg()['metric_module_hashes']
        def models(self,h,w):
            im,pm=super().models(h,w);cfg=evalcfg()
            audit=dict(compression_hash=base.core.compression_hash(im,pm),I_hash=base.core.module_hash(im),wrapper_hash=None if self.wrapper is None else base.core.module_hash(self.wrapper),receiver_hashes=self.loaded_receiver_hashes,metric_module_hashes=self.metric_hashes)
            assert audit['compression_hash']==cfg['compression_hash'] and audit['I_hash']==cfg['I_hash']
            assert audit['receiver_hashes']==(cfg['original_receiver_hashes'] if self.method=='original' else cfg['deployment_receiver_hashes'][self.method])
            if self.method!='original':assert audit['wrapper_hash']==cfg['checkpoints'][self.method]['module_hashes']['wrapper']
            assert all(not p.requires_grad for m in (im,pm,*self.quality,*self.metric_models) for p in m.parameters())
            assert {str(p.dtype) for p in im.parameters()}=={str(p.dtype) for p in pm.parameters()}=={'torch.float16'}
            if self.wrapper is not None:assert {str(p.dtype) for p in self.wrapper.parameters()}=={'torch.float32'}
            if hasattr(self,'audit'):assert self.audit==audit
            self.audit=audit;return im,pm
        def assert_frozen(self,im,pm):
            assert base.core.compression_hash(im,pm)==self.audit['compression_hash'] and base.core.module_hash(im)==self.audit['I_hash']
            assert {k:base.core.module_hash(m) for k,m in (('bridge',pm.recon_generation_net.mlp),('generator',pm.recon_generation_net.decoder))}==self.audit['receiver_hashes']
        def run(self,*args,**kwargs):
            before=(None if self.wrapper is None else base.core.module_hash(self.wrapper),[base.core.module_hash(x) for x in (*self.quality,*self.metric_models)])
            result,rows=run(self,*args,**kwargs)
            after=(None if self.wrapper is None else base.core.module_hash(self.wrapper),[base.core.module_hash(x) for x in (*self.quality,*self.metric_models)]);assert before==after
            result.update(runtime_audit=self.audit,model_hashes_unchanged=True,protocol_sha256=sha(ROOT/'protocol.json'),i_frame_mode='native_original' if self.method=='original' else 'native_RGB_bypass')
            return result,rows
    base.Runtime=Runtime;return base
_e=None
def validate(r,v,q,m,ip=-1):
    global _e
    if _e is None:
        _e=module('gop21_validate_base',BASE/'evaluate.py');_e.ROOT=ROOT;_e.load=lambda p:evalcfg() if Path(p)==ROOT/'config.json' else load(p)
    audit=load(ROOT/'qp_semantics_audit.json')
    periodic={'actual_qps':{str(qp):[values[i if ip<0 else i%ip] for i in range(64)] for qp,values in audit['actual_qps'].items()}}
    _e.load=lambda p:evalcfg() if Path(p)==ROOT/'config.json' else periodic if Path(p).name=='qp_semantics_audit.json' else load(p)
    _e.validate(r,v,q,m);cfg=evalcfg()
    assert r['IP']==ip and r['cache_key']==cache_key(v,q,m,ip)
    assert r['source_frame_indices']==v['source_frame_indices']
    assert r['loaded_receiver_module_hashes']==(cfg['original_receiver_hashes'] if m=='original' else cfg['deployment_receiver_hashes'][m])
    assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
    assert r['rate_accounting_fps']==v['rate_accounting_fps']
    if r.get('reused'):
        assert ip==-1 and m in METHODS[:2] and sha(r['reused_from'])==r['reused_sha256'] and r['reuse_verified']
    else:
        assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['config_sha256']==sha(ROOT/'config.json') and r['model_hashes_unchanged']
        assert r['runtime_audit']['metric_module_hashes']==cfg['metric_module_hashes']
        assert [a['frame'] for a in r['I_refresh_audits']]==([0] if ip<0 else list(range(0,64,ip)))
        for a in r['I_refresh_audits']:assert len({a[k] for k in ('native_I_sha256','encoder_reference_sha256','decoder_reference_sha256','independent_reference_sha256')})==1
        assert r['I_frames']+r['P_frames']==64 and r['I_bits_including_associated_headers']+r['P_bits_including_headers']==r['real_bytes']*8
