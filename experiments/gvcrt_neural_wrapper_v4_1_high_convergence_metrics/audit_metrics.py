import importlib.metadata
import json
import subprocess
from experiment_utils import ROOT, REPO, dump, sha
from metric_runtime import flo,FID_SOURCE,fid,COMPAT

files={p:sha(p) for p in [COMPAT/'flolpips_compat.py',flo.OFFICIAL_ROOT/'flolpips.py',flo.OFFICIAL_ROOT/'pwcnet.py',
       flo.OFFICIAL_ROOT/'weights/v0.1/alex.pth',flo.PWC_WEIGHTS,FID_SOURCE,fid.INCEPTION_WEIGHTS]}
assert files[flo.OFFICIAL_ROOT/'flolpips.py']==flo.OFFICIAL_SOURCE_SHA256
assert files[flo.OFFICIAL_ROOT/'pwcnet.py']==flo.OFFICIAL_PWC_SOURCE_SHA256
assert files[flo.OFFICIAL_ROOT/'weights/v0.1/alex.pth']==flo.OFFICIAL_ALEX_WEIGHT_SHA256
assert files[flo.PWC_WEIGHTS]==flo.PWC_WEIGHT_SHA256
versions={}
for name in ('torch','torchvision','scipy','numpy','lpips','DISTS-pytorch','pytorch-fid','torchmetrics','clean-fid'):
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
dump('metric_implementation_audit.json',dict(
    repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    repository_evidence=['README.md','src/utils/metrics.py','expericent_generation_input/expericent_perceptual_temporal_refiner_v6/METRIC_PROTOCOL_AUDIT.md'],
    official_gvcrt_metric_pipeline='No executable published FloLPIPS/FID pipeline located in checked repository; reuse existing documented GVC-RT experiment adapters. No claim of equivalence to a private paper pipeline.',
    packages=versions,source_and_weight_sha256={str(p):h for p,h in files.items()},
    FloLPIPS=dict(implementation_source='official Danier et al. FloLPIPS source through existing local compatibility adapter',
        module_path=str(COMPAT/'flolpips_compat.py'),commit=flo.OFFICIAL_COMMIT,
        input_range='float32 RGB [0,1]; normalize=True maps to [-1,1] in LPIPS; no YUV or BGR conversion',
        flow_estimator='official PWCNet default weights, pure PyTorch 9x9 correlation compatibility implementation',
        frame_normalization='source and decoded reconstruction clamped to [0,1], no uint8 requantization',
        resize_crop='1920x1080 full frames, remove codec bottom padding only; PWC internally bilinear-resizes to multiples of 64 (1920x1088), rescales flow to original dimensions; no external resize/crop',
        temporal_order='adjacent frames t -> t+1 in original order; score on t using reference flow minus reconstruction flow; arithmetic mean of 31 validation or 63 final transitions; no transitions across videos',
        zero_flow_policy='fail explicitly, no invented zero or replacement metric'),
    FID=dict(implementation_source=str(FID_SOURCE),feature_extractor='FID InceptionV3 pool3 2048D with cached pytorch-fid TensorFlow-port weights',
        input_range='float32 RGB [0,1]',preprocessing='full 1920x1080 frame; bilinear resize to 299x299, align_corners=False; scale 2*x-1; transform_input=False; no uint8 quantization',
        aggregation='pool frames across all videos within split and method x QP, never average per-video FID',
        validation_samples=192,final_samples=512,all_qp_fid='not requested as mandatory; not computed',
        numerical_implementation='existing low_rank_fid: float64 centered features, unbiased N-1 covariance, exact low-rank SVD trace',
        sample_limitations='finite correlated video-frame samples; sample counts reported in each result'),
    use='evaluation only; no influence on training loss',status='PASS'))
print('METRIC IMPLEMENTATION AUDIT PASS')
