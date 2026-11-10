"""LPIPS-VGG trains; Alex and VGG are separately named evaluation metrics."""
from common_s2 import *
import torch

def lpips_model(net,device):
    import lpips
    m=lpips.LPIPS(net=net,version='0.1',verbose=False).to(device).eval().requires_grad_(False)
    return m

def training_models(device,dists):return lpips_model('vgg',device),dists

def metric_audit():
    import lpips,importlib.metadata,base64
    files={}
    distribution=importlib.metadata.distribution('lpips')
    for name in ['alex','vgg']:
        p=Path(lpips.__file__).parent/'weights/v0.1'/f'{name}.pth'
        item=next(x for x in distribution.files if distribution.locate_file(x).resolve()==p.resolve())
        assert item.hash and item.hash.mode=='sha256'
        expected=base64.urlsafe_b64decode(item.hash.value+'='*((4-len(item.hash.value)%4)%4)).hex()
        assert sha(p)==expected;files[name]=dict(path=str(p),SHA256=expected,verified_against_installed_wheel_RECORD=True)
    return dict(package='lpips',version=distribution.version,learned_weights=files,input_RGB_range=[0,1],normalize=True,training_LPIPS='VGG only; Alex not added',evaluation_LPIPS=['Alex','VGG'],DISTS_require_grad_training=True)
