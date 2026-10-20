"""Freeze v6.39 read-only task and dependency manifests before inference."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = HERE.parent / "gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500"
QPS = (0,5,9)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = HERE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def main():
    manifest = json.loads((OLD/"eval_manifest.json").read_text())
    methods = {
        "M00": dict(full="Y_originalBridge_fullG", spatial="z0", stats="z0", origin="v6.37 endpoint"),
        "M10": dict(full="C_zF_spatial_z0_stats", spatial="zF", stats="z0", origin="new inference"),
        "M01": dict(full="S_z0_spatial_zF_stats", spatial="z0", stats="zF", origin="new inference"),
        "M11": dict(full="F_fullLoss_noP_500", spatial="zF", stats="zF", origin="v6.37 endpoint"),
    }
    tasks=[]
    for dataset, videos in manifest.items():
        for v in videos:
            for qp in QPS:
                key=f"{dataset}_{v['video_index']}_q{qp}"
                receipt=OLD/"tasks"/f"replay_{key}.json"
                x=json.loads(receipt.read_text())
                assert x["status"]=="PASS"
                archive=Path(x["latent_archive"]["path"])
                assert archive.exists() and sha(archive)==x["latent_archive"]["sha256"]
                tasks.append(dict(task=dict(dataset=dataset,video=v["video_index"],qp=qp),
                                  source_receipt=str(receipt),source_receipt_sha256=sha(receipt),
                                  latent_archive=str(archive),latent_archive_sha256=sha(archive),
                                  source_rgb_sha256=v["rgb_sha256"],source_sha256=v["source_sha256"]))
    assert len(tasks)==39
    source_files=[OLD/name for name in ("receiver37.py","replay37.py","hook_verify37.py","adapter35.py","io32.py","training35.py","evaluation_config.json","eval_manifest.json","protocol.json","audits/metric_models.json")]
    source_files += [REPO/"src/models/improved_model_gvcrt.py", REPO/"checkpoints/GVC-RT_P.pt"]
    dependencies={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in source_files}
    cp=HERE.parent/"gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500/evaluation/checkpoints/N_noP_BG_FD_500.pt"
    assert sha(cp)=="ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc"
    dependencies[str(cp)]=dict(sha256=sha(cp),bytes=cp.stat().st_size)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    status=subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True)
    config=dict(experiment="v6.39", mode="read_only_cross_inference", git_head_at_start=head,
                worktree_porcelain_sha256=hashlib.sha256(status.encode()).hexdigest(),
                methods=methods, physical_gpu_allowed=[4,5,6,7], QP=list(QPS),
                datasets={k:len(v) for k,v in manifest.items()}, videos=13, frames=64,
                P_frames=63, training_updates=0, new_real_encodes=0, streams_reused=39,
                RD_planned=156, FID_planned=36, endpoint_raw_rgb_maxabs=1e-5,
                endpoint_relative_L2=1e-6, endpoint_spatial_metric_abs=1e-5,
                endpoint_pooled_FID_abs=1e-3, description="Frozen G_F; swap only z_spatial and AdaGN second input.")
    write("effective_config.json",config)
    write("evaluation/task_manifest.json",tasks)
    write("source_manifest.json",dict(dependencies=dependencies, v637_eval_manifest_sha256=sha(OLD/"eval_manifest.json"),
                                      model_checkpoint=str(cp), historical_archive_receipts=39,
                                      historical_training_exposure="U-Long+UVG",official_pretraining_exposure="UNKNOWN"))
    write("checkpoint_reuse_manifest.json",dict(original_bridge_checkpoint=str(REPO/"checkpoints/GVC-RT_P.pt"),
                                                full_bridge_and_generator_checkpoint=str(cp),
                                                full_checkpoint_sha256=sha(cp),
                                                expected_deployed_generator_hash="f1c4d2c3e2aa424f1989a9cfe5abce381330435e05fd04a8e1367bbcf512ef8c"))
    write("audits/var_dtype_and_padding_semantics.json",dict(source=str(REPO/"src/models/improved_model_gvcrt.py"),
          source_sha256=sha(REPO/"src/models/improved_model_gvcrt.py"),
          code="rearrange(z,'b c h w -> b c (h w)'); sqrt(var(dim=-1)+eps); mean(dim=-1)",
          torch_var_default_correction=1, empirical_float32_var_of_1_and_2=0.5,
          empirical_torch_version="2.6.0+cu124", configured_eps=1e-6,
          actual_stats_scope="full 68x120 padded 18-channel generator latent; no crop or mask", dtype="source archive FP16; decoder deployment FP16"))
    print("PREPARE PASS",len(tasks),flush=True)


if __name__=="__main__": main()
