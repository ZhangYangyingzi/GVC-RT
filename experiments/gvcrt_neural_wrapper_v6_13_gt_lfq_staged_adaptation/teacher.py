"""Official frozen FP32 encoder, before LFQ sign; no model.encode shortcut."""
import importlib.util
import torch
from io_utils import *
def official_module():
    path=ROOT/'teacher_sources/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py'
    spec=importlib.util.spec_from_file_location('v613_official_magvit',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Teacher(torch.nn.Module):
    def __init__(self,device,ema=True,candidate='imagenet256'):
        super().__init__()
        import yaml
        assert candidate in ('imagenet256','pretrain262144')
        self.audit_root=ROOT/'teacher_checks'/(candidate+('_ema' if ema else '_raw'))
        config=ROOT/'teacher_sources/configs/Open-MAGVIT2/gpu'/('imagenet_lfqgan_256_L.yaml' if candidate=='imagenet256' else 'pretrain_lfqgan_256_262144.yaml')
        cfg=yaml.safe_load(config.read_text())['model']['init_args'];assert cfg['embed_dim']==cfg['ddconfig']['z_channels']==18 and cfg['n_embed']==2**18
        cp=ROOT/'teacher_weights'/('imagenet_256_L.ckpt' if candidate=='imagenet256' else 'pretrain256_262144.ckpt');provenance=load(ROOT/'audits'/('teacher_checkpoint_source.json' if candidate=='imagenet256' else 'pretrain_teacher_checkpoint_source.json'));assert sha(cp)==provenance['checkpoint']['sha256']
        state=torch.load(cp,map_location='cpu',weights_only=True)['state_dict']
        # Isolate random constructor initialization from the student's RNG.
        with torch.random.fork_rng(devices=[]):self.encoder=official_module().Encoder(**cfg['ddconfig'])
        selected={};mapping={}
        for name,t in self.encoder.state_dict().items():
            key='model_ema.'+('encoder.'+name).replace('.','') if ema else 'encoder.'+name
            assert key in state,('missing required encoder weight',key)
            assert state[key].shape==t.shape,(key,state[key].shape,t.shape)
            selected[name]=state[key];mapping[name]=key
        self.encoder.load_state_dict(selected,strict=True)
        self.to(device).float().eval().requires_grad_(False)
        dump(self.audit_root/'teacher_loading.json',dict(status='PASS',EMA=ema,candidate=candidate,strict_encoder_loading=True,missing_encoder_keys=[],checkpoint_sha256=sha(cp),config_sha256=sha(config),encoder_keys=mapping,latent_position='encoder(GT_RGB * 2 - 1), immediately before LFQ.forward hard sign',input_range=[-1,1],channel_order='official encoder.conv_out unchanged',projection='none: dim == log2(codebook_size) == 18',normalization='no additional latent normalization in official LFQ.forward',forbidden_encode_output_used=False,frozen=True,eval=True,teacher_deployed=False,source=provenance))
    @torch.no_grad()
    def forward(self,rgb):
        assert rgb.dtype==torch.float32 and rgb.shape[1]==3
        return self.encoder(rgb*2-1)
