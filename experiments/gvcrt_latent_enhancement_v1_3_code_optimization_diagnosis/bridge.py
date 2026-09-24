"""Read-only historical models and data. Only V1.3 code tensors may be optimized."""
import sys
import os
import json
import csv
import time
import hashlib
from pathlib import Path

sys.dont_write_bytecode=True
HOME=Path(__file__).resolve().parent
CONFIG=json.loads((HOME/"config.json").read_text())
OLD=Path(CONFIG["v12"])
sys.path.insert(0,str(OLD))
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=CONFIG["gpu_uuid"]
from support import (torch,nn,F,np,V1,V11,ROOT,FixedModels,lpips_model,base_rows,manifest,
                     source_frame,sha,tensor_hash,digest_state,save_json,save_pt,write_csv,read_csv,
                     rgb01,image,write_video,calc_psnr,calc_msssim_rgb,load_models,decode_base,dpb_hash)
from model import ResidualCodec
import bitstream as ORC

# Historical support sets its old visibility at import; override before CUDA initialization.
os.environ["CUDA_VISIBLE_DEVICES"]=CONFIG["gpu_uuid"]

MODEL=OLD/CONFIG["model"]
DELTA=CONFIG["delta"]
PIXELS=1080*1920


def setup():
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


class Bundle:
    def __init__(self,perceptual=True):
        self.fixed=FixedModels()
        ck=torch.load(MODEL,map_location="cpu",weights_only=False)
        assert ck["step"]==1500 and ck["deltas"][3]==DELTA
        self.net=ResidualCodec(ck["residual_scale"]).cuda().eval().requires_grad_(False)
        self.net.load_state_dict(ck["model"],strict=True)
        self.lp=lpips_model() if perceptual else None
        self.fingerprints=self.hashes()

    def hashes(self):
        return {"fixed":self.fixed.hashes(),"residual_codec":digest_state(self.net)}

    def check(self):
        assert self.hashes()==self.fingerprints
        self.fixed.assert_frozen()
        assert all(not p.requires_grad and p.grad is None for p in self.net.parameters())

    def context(self,data):
        ec=data["ell_c"].cuda().detach()
        q=data["q_recon"].cuda().detach()
        with torch.no_grad():
            d0=self.net.decoder(torch.zeros(CONFIG["code_shape"],device="cuda"),ec).detach()
        return ec,q,d0

    def render(self,u,context):
        ec,q,d0=context
        r=(self.net.decoder(u,ec)-d0)*self.net.residual_scale
        return rgb01(self.fixed.g(ec+r,q)),r


def entries(split):
    m=manifest()
    return m["train2"] if split=="Train2" else m["val6"]


def sample_specs():
    train=json.loads((OLD/"results_v1/fixed_samples.json").read_text())
    trained={(s["video"],s["frame"]) for s in train}
    result=[]
    for split in ["Train2","Val6"]:
        for e in entries(split):
            for frame in range(1,e["frames"]):
                video=e["video_id"]
                group="network_train6" if (video,frame) in trained else ("Train2_rest24" if split=="Train2" else "Val6_90P")
                sid=f"{video}_f{frame:04d}"
                path=OLD/"results_v1"/("teachers_train6" if group=="network_train6" else "sender_oracles")/f"{sid}.pt"
                result.append({"sample":sid,"video":video,"frame":frame,"split":split,"group":group,"teacher_file":str(path),
                               "pts_seconds":e["pts_seconds"][frame],"duration_seconds":e["duration_seconds"]})
    return result


def load_sample(spec):
    t=time.perf_counter()
    d=torch.load(spec["teacher_file"],map_location="cpu",weights_only=False)
    assert tuple(d["ell_c"].shape)==(1,18,68,120) and tuple(d["q_recon"].shape)==(1,320,1,1)
    return d,time.perf_counter()-t


def inventory():
    import subprocess
    result={}
    for root in [V1,V11,OLD]:
        for path in subprocess.check_output(["rg","--files","--hidden","--no-ignore",str(root)],text=True).splitlines():
            stat=os.stat(path)
            result[path]={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    return result
