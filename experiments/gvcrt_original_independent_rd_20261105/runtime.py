"""Original codec only; shared scoring functions do not import experiment models."""
from common import *
import ast
import math
import sys


def initialize(device):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(ORIGINAL))
    sys.path.append(str(TCG_REPO/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
    from src.models.image_model_gvcrt import DMCI
    from src.models.video_model_gvcrt import DMC
    from src.utils.common import set_torch_env
    from src.utils import stream_helper
    from src.layers.cuda_inference import replicate_pad, CUSTOMIZED_CUDA_INFERENCE
    set_torch_env()
    torch.set_num_threads(2)
    def no_download(*args,**kwargs):
        raise RuntimeError('Required pretrained metrics must already exist locally')
    torch.hub.download_url_to_file=no_download
    namespace=dict(torch=torch,F=F,math=math)
    for path,names in [(TCG_REPO/'experiments/gvcrt_neural_wrapper_v2_joint/core.py',{'ms_ssim_rgb','quality_models'}),
                       (TCG_REPO/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py',{'frame_metrics'})]:
        tree=ast.parse(path.read_text())
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in selected}==names
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    runtime=dict(torch=torch,device=torch.device(device),DMCI=DMCI,DMC=DMC,stream=stream_helper,
                 replicate_pad=replicate_pad,frame_metrics=namespace['frame_metrics'],
                 quality_models=namespace['quality_models'],custom_cuda_kernels=CUSTOMIZED_CUDA_INFERENCE)
    verify_codec_imports()
    return runtime


def verify_codec_imports():
    imported={}
    for name,module in list(sys.modules.items()):
        if name.startswith('src.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve()
            assert path.is_relative_to(ORIGINAL),('NON_ORIGINAL_CODEC_IMPORT',name,str(path))
            imported[name]=dict(path=str(path),sha256=sha(path))
    return imported


def load_models(runtime):
    torch=runtime['torch'];device=runtime['device']
    im=runtime['DMCI'](encoder_ckpt_path=str(PROJECT/'checkpoints/GVC-RT_I.pt')).to(device).eval().half()
    pm=runtime['DMC']()
    state=torch.load(PROJECT/'checkpoints/GVC-RT_P.pt',map_location='cpu',weights_only=True)
    state=state.get('student_ema',state.get('student',state.get('state_dict',state)))
    pm.load_state_dict(state,strict=True)
    pm=pm.to(device).eval().half()
    for model in (im,pm):
        model.requires_grad_(False)
        assert not any('tcg' in n.lower() for n in model.state_dict())
        model.update(.12)
        model.set_use_two_entropy_coders(True)
    assert im.get_qp_num()==pm.get_qp_num()==10
    return im,pm


def load_quality(runtime):
    sys.path.insert(0,str(FLO))
    import flolpips_compat
    quality=runtime['quality_models'](runtime['device'])
    flow,perceptual=flolpips_compat.load_models(runtime['device'])
    return quality,flow,perceptual


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_sha(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
