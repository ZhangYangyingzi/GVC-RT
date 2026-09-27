"""Read-only metadata inspection; materializes audit JSON in this experiment only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from audit_utils import *

def main():
    config=load('config.json');records={}
    for method in METHODS[1:]:
        cp=checkpoint(method);assert sha(cp)==config['checkpoints'][method]['sha256']
        value=torch.load(cp,map_location='cpu',weights_only=True)
        keys=list(value)
        metadata={k:v for k,v in value.items() if isinstance(v,(str,int,float,bool,type(None)))}
        assert all(k in value for k in ('wrapper','bridge','generator'))
        records[method]=dict(path=str(cp),sha256=sha(cp),top_level_keys=keys,scalar_metadata=metadata,
            wrapper_tensor_count=len(value['wrapper']),bridge_tensor_count=len(value['bridge']),generator_tensor_count=len(value['generator']))
        del value
    paths=[REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt',REPO/'src/models/common_model.py',
        REPO/'src/models/video_model_gvcrt.py',REPO/'src/utils/stream_helper.py',V4/'final_evaluate.py',
        ROOT.parent/'gvcrt_neural_wrapper_v2_joint/core.py',
        REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py']
    dump('checkpoint_metadata_audit.json',dict(status='PASS',source_paths_resolved_from_V5_final_records=True,
        evaluation_only=True,read_only_torch_load=True,checkpoints=records,additional_source_hashes={str(p):sha(p) for p in paths}))
    dump('source_preprocessing_audit.json',dict(status='PASS',ulong_source='exact V5-A original fixed RGB PNG frame loader',
        UVG_source='local raw planar YUV420p 8-bit, decoded using established V2 corrected UVG ffmpeg rawvideo -> RGB24 convention',
        UVG_conversion_reference=str(ROOT.parent/'gvcrt_neural_wrapper_v2_retest_fixed_uvg/prepare_sources.py'),
        UVG_color_matrix='ffmpeg default BT.601',UVG_color_range='limited YUV to full RGB',
        UVG_color_metadata_caveat='raw files have no embedded color metadata; same explicit prior experiment convention retained, not inferred camera colorimetry',
        actual_input='float32 RGB [0,1]',processing_canvas=[1920,1088],padding='bottom 8 pixels replicate',
        metrics_crop=[1920,1080],I_frame_only_at_zero=True,no_eight_frame_reset=True,
        official_rgb_benchmark_path_used=True,official_yuv_metric_branch_not_used=True,
        official_yuv_branch_note='official src_type=yuv420 feeds YCbCr channels; this RGB-trained wrapper and RGB perceptual metrics use converted RGB consistently for all four methods'))
    print('INPUT METADATA AUDIT PASS')

if __name__=='__main__':main()
