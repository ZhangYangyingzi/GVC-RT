"""Official CCM/CSM and four-scale discriminators, explicitly pretrained Lite1."""
import sys,random
from io21 import *
sys.path.insert(0,str(ROOT/'vendor/dependencies'));sys.path.insert(0,str(ROOT/'vendor/projected-gan'))
import torch
import torch.nn.functional as F
import timm
from pg_modules import projector as official
from pg_modules.discriminator import MultiScaleD
WEIGHT=ROOT/'weights/tf_efficientnet_lite1-bde8b488.pth'
MODEL='tf_efficientnet_lite1'
def build_projector(im_res,cout,proj_type,expand=False):
    assert proj_type==2
    model=timm.create_model(MODEL,pretrained=False)
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    assert tensor_hash(model.state_dict())==tensor_hash(state)
    model.eval();pretrained=official._make_efficientnet(model).eval()
    pretrained.RESOLUTIONS=[64,32,16,8];pretrained.CHANNELS=official.calc_channels(pretrained)
    scratch=torch.nn.Module();scratch=official._make_scratch_ccm(scratch,pretrained.CHANNELS,cout,expand)
    scratch=official._make_scratch_csm(scratch,scratch.CHANNELS,cout,expand)
    pretrained.RESOLUTIONS=[128,64,32,16];pretrained.CHANNELS=scratch.CHANNELS
    return pretrained,scratch
def create(device):
    official._make_projector=build_projector
    # Isolate all initialization from shared training and codec RNG.
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(20261026);torch.cuda.manual_seed_all(20261026)
        feature=official.F_RandomProj(cout=64,expand=True,proj_type=2).to(device).eval().requires_grad_(False)
        disc=MultiScaleD(feature.CHANNELS,feature.RESOLUTIONS,num_discs=4).to(device)
    # Establish spectral normalization buffers before the first frozen G pass.
    with torch.no_grad():
        for layer in disc.modules():
            for hook in layer._forward_pre_hooks.values():
                if hasattr(hook,'compute_weight') and hasattr(hook,'n_power_iterations'):
                    for _ in range(50):layer.weight=hook.compute_weight(layer,do_power_iteration=True)
    model=Projected(feature,disc).to(device)
    p=ROOT/'audits/projected_source.json'
    if not p.exists() or load(p).get('implementation_sha256')!=sha(ROOT/'projected.py'):
        commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/projected-gan'),'rev-parse','HEAD'],text=True).strip()
        dump(p,dict(status='PASS',implementation_sha256=sha(ROOT/'projected.py'),model=MODEL,pretrained=True,strict_state_dict=True,weight_path=str(WEIGHT),weight_sha256=sha(WEIGHT),weight_bytes=WEIGHT.stat().st_size,weight_source='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',timm=timm.__version__,torch=torch.__version__,official_commit=commit,official_reference='https://github.com/autonomousvision/projected-gan',adaptations=['tf_efficientnet_lite1 replaces official Lite0 as instructed','four independent scale means averaged equally','feature/projection eval and frozen','no differential augmentation'],input_range='RGB [0,1] mapped to [-1,1], official interpolation224 bilinear',CCM_CSM='official F_RandomProj(proj_type=2,cout=64,expand=True)',num_scales=4,spectral_normalization_initial_power_iterations=50,initialization_seed=20261026,independent_RNG=True,feature_hash=tensor_hash(feature.state_dict()),dependencies={str(x):sha(x) for x in (ROOT/'vendor/projected-gan/pg_modules').glob('*.py')}))
    return model
class Projected(torch.nn.Module):
    def __init__(self,feature,disc):super().__init__();self.feature=feature;self.discriminator=disc
    def train(self,mode=True):
        super().train(mode);self.feature.eval().requires_grad_(False);self.discriminator.train(mode);return self
    def scores(self,x):
        fs=self.feature(F.interpolate(x*2-1,224,mode='bilinear',align_corners=False))
        return [d(fs[k],None) for k,d in self.discriminator.mini_discs.items()]
    def adversarial(self,x):return torch.stack([-s.mean() for s in self.scores(x)]).mean()
    def d_loss(self,gt,fake):return .5*torch.stack([F.relu(1-s).mean() for s in self.scores(gt)]).mean()+.5*torch.stack([F.relu(1+s).mean() for s in self.scores(fake.detach())]).mean()
    def g_mode(self):self.eval();self.discriminator.requires_grad_(False)
    def d_mode(self):self.train();self.discriminator.requires_grad_(True)
