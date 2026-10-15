"""Snapshots and paths/hashes only; never restores into historical project directories."""
from preflight import *
import shutil,importlib
def main():
 paths=[GVC/'experiments/gvcrt_neural_wrapper_v2_joint/wrapper_model.py',GVC/'experiments/gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control/config.json',GVC/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py',Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback/flolpips_compat.py')]
 rows=[]
 for p in paths:
  dest=R/'dependencies'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest);rows.append(dict(source=str(p),sha256=sha(p),snapshot=str(dest.relative_to(R))))
 coder=importlib.import_module('src.entropy_coder.MLCodec_rans');p=Path(coder.__file__);rows.append(dict(source=str(p),sha256=sha(p),bytes=p.stat().st_size,role='pre-existing local rANS binary, server dependency only'))
 save('audits/reference_dependencies.json',dict(files=rows,GLC_commit=subprocess.check_output(['git','-C',str(GLC),'rev-parse','HEAD'],text=True).strip(),GVC_commit=subprocess.check_output(['git','-C',str(GVC),'rev-parse','HEAD'],text=True).strip(),rate_extension='local non-official rANS',DISTS_weights_sha256=sha('/Huang_group/zyyz/home_dir/.conda/envs/glc/weights.pt'),GLC_PROJECT=str(GLC)))
if __name__=='__main__':main()
