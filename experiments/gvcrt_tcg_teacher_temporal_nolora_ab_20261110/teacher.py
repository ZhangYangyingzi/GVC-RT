"""Read-only official pretrain262144 EMA encoder; local audit writes only."""
from common_s2 import *
import torch

class Teacher(torch.nn.Module):
    def __init__(self,device):
        super().__init__();import yaml
        gate=TEACHER_GATE;assert gate['status']=='PASS' and gate['EMA'] and gate['selected_candidate']=='pretrain262144'
        loading=TEACHER_SOURCE/'teacher_checks/pretrain262144_ema/teacher_loading.json'
        assert sha(loading)==gate['records']['teacher_checks/pretrain262144_ema/teacher_loading.json']
        old=load(loading);source=TEACHER_SOURCE/'teacher_sources/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py'
        config=TEACHER_SOURCE/'teacher_sources/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml'
        assert sha(config)==old['config_sha256'] and sha(TEACHER_CHECKPOINT)==TEACHER_SHA
        expected_code=gate['official_source']['official_code_commit']
        downloads=load(TEACHER_SOURCE/'audits/official_source_downloads.json')
        registered=load(TEACHER_SOURCE/'teacher_sources/official_commit.json')['sha'];assert registered==expected_code==downloads['commit']
        for path in [source,config,TEACHER_SOURCE/'teacher_sources/official_commit.json',TEACHER_SOURCE/'teacher_sources/official_tree.json']:
            record=next(r for r in downloads['records'] if r['path']==str(path));assert sha(path)==record['sha256'],('OFFICIAL_ARCHIVE_SHA256',str(path))
        tree=load(TEACHER_SOURCE/'teacher_sources/official_tree.json')['tree']
        entry=next(x for x in tree if x['path']=='src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py')
        raw=source.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['sha']
        config_data=yaml.safe_load(config.read_text())['model']['init_args'];assert config_data['embed_dim']==config_data['ddconfig']['z_channels']==18
        official=module('teacher_official_encoder',source)
        with torch.random.fork_rng(devices=[]):self.encoder=official.Encoder(**config_data['ddconfig'])
        state=torch.load(TEACHER_CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)['state_dict'];selected={};mapping={}
        for n,t in self.encoder.state_dict().items():
            key='model_ema.'+('encoder.'+n).replace('.','');assert key in state and state[key].shape==t.shape
            selected[n]=state[key];mapping[n]=key
        assert mapping==old['encoder_keys'];self.encoder.load_state_dict(selected,strict=True)
        self.to(device).float().eval().requires_grad_(False)
        self.audit=dict(status='PASS',candidate='pretrain262144',EMA=True,checkpoint_path=str(TEACHER_CHECKPOINT),checkpoint_SHA256=TEACHER_SHA,config_SHA256=sha(config),official_encoder_source_SHA256=sha(source),official_code_commit=registered,strict_encoder_loading=True,encoder_keys=mapping,encoder_SHA256=tensor_hash(self.encoder.state_dict().items()),continuous_before_hard_sign=True,output_channels=18,input_range=[-1,1],normalization='official identity; no additional per-frame normalization',frozen=True,eval=True,old_coordinate_gate_only=True,temporal_stability_not_established=True,historical_training_checkpoint_identity_not_claimed=True,writes_to_source=False)
    @torch.no_grad()
    def forward(self,rgb):
        assert rgb.dtype==torch.float32 and rgb.shape[1]==3
        assert rgb.shape[-2]%16==rgb.shape[-1]%16==0
        out=self.encoder(rgb*2-1)
        assert out.shape==(rgb.shape[0],18,rgb.shape[-2]//16,rgb.shape[-1]//16)
        assert torch.isfinite(out).all()
        return out
