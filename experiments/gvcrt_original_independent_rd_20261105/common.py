"""Paths and atomic bookkeeping for the independent original-code evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ORIGINAL = PROJECT/'GVC-RT_worktrees/gvcrt_original_independent_20261105'
TCG_REPO = PROJECT/'GVC-RT_worktrees/tcg_stage1_data_expansion_ab_groupisolated_v2'
TCG = TCG_REPO/'experiments/gvcrt_tcg_stage1_data_expansion_ab_groupisolated_v2'
PARENT = Path('/Huang_group/zyyz/home_dir/tmp/gvcrt_tcg_stage1_expanded_20261103_worktree/experiments/gvcrt_tcg_stage1_expanded_5000')
PYTHON = '/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
COMMIT = 'b2a0895c0cf31cf2be3ab1c2571ca07c05866c03'
FLO = Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
METRICS = ('PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC')
QP_PATTERN = (0,1,0,2,0,2,0,2)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_protocol():
    p = load(ROOT/'protocol.json')
    assert p['original_commit'] == COMMIT
    for relative, expected in p['original_source_sha256'].items():
        assert sha(ORIGINAL/relative) == expected, relative
    for filename, expected in p['checkpoint_sha256'].items():
        assert sha(PROJECT/'checkpoints'/filename) == expected, filename
    for path, expected in p['metric_sources_sha256'].items():
        assert sha(path) == expected, path
    assert sha(TCG/'metrics_v3.csv') == p['previous_metrics_sha256']
    assert sha(ROOT/'videos.json') == p['videos_sha256']
    return p
