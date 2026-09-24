#!/usr/bin/env bash
set -euo pipefail
ROOT=/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_vs_dcvc_rt_matched_rate
DCVC=/Huang_group/zyyz/Projects/DCVC_RT
TAG=$1
QP=$2
GPU=$3
QPP=${4:-$QP}
SUFFIX="q${QP}"
if [[ "$QPP" != "$QP" ]]; then SUFFIX="qi${QP}_qp${QPP}"; fi
OUT="$ROOT/parts/dcvc_${TAG}_${SUFFIX}"
DONE="$OUT/PASS.json"
if [[ -f "$DONE" ]]; then exit 0; fi
mkdir -p "$OUT/streams" "$OUT/recon"
CUDA_VISIBLE_DEVICES="$GPU" /Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python "$DCVC/test_video.py" \
 --model_path_i "$DCVC/checkpoints/cvpr2025_image.pth.tar" \
 --model_path_p "$DCVC/checkpoints/cvpr2025_video.pth.tar" \
 --rate_num 1 --qp_i "$QP" --qp_p "$QPP" \
 --test_config "$ROOT/parts/${TAG}_dcvc_config.json" --cuda 1 --cuda_idx 0 -w 1 \
 --write_stream 1 --force_zero_thres 0.12 --output_path "$OUT/result.json" \
 --force_intra_period -1 --reset_interval 64 --force_frame_num 64 \
 --check_existing 0 --calc_ssim 1 --save_decoded_frame 1 \
 --stream_path "$OUT/streams" --verbose 0 >"$OUT/stdout.log" 2>"$OUT/stderr.log"
BIN=$(find "$OUT/streams" -type f -name '*.bin' -print -quit)
test -n "$BIN"
REC=$(dirname "$BIN")
test "$(find "$REC" -maxdepth 1 -name 'im*.png' | wc -l)" -eq 64
cp "$BIN" "$ROOT/bitstreams/dcvc/${TAG}_${SUFFIX}.bin"
mkdir -p "$ROOT/reconstructions/${TAG}/DCVC_${SUFFIX}"
cp "$REC"/im*.png "$ROOT/reconstructions/${TAG}/DCVC_${SUFFIX}/"
python - "$BIN" "$DONE" <<'PY'
import hashlib,json,os,sys
p,d=sys.argv[1:]
b=open(p,'rb').read()
open(d,'w').write(json.dumps({'status':'PASS','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})+'\n')
PY
