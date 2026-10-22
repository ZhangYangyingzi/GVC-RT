"""Mechanical aggregation of v6.41 A fixed 2x2 read-only intervention."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500"
sys.path.insert(0,str(OLD))
from aggregate37 import pool  # exact historical low-rank FID implementation
from intervene41 import METHODS, atomic_csv, atomic_json, sha

SPATIAL=("LPIPS","DISTS","FloLPIPS")
ALL_METRICS=(*SPATIAL,"FID","D_mu","D_cov")
EFFECTS=(("body_given_old_mod","A10","A00"),
         ("body_given_new_mod","A11","A01"),
         ("mod_given_old_body","A01","A00"),
         ("mod_given_new_body","A11","A10"),
         ("joint_generator","A11","A00"))


def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))


def interp(rows, metric, grid):
    xy=sorted((float(r["actual_bpp"]),float(r[metric])) for r in rows)
    assert len(xy)==3 and all(xy[i][0]<xy[i+1][0] for i in range(2))
    return np.interp(grid,np.log([x for x,y in xy]),[y for x,y in xy])


def effect_row(level,ds,video,qp,metric,label,method,reference,a,b,lo=None,hi=None):
    return dict(level=level,dataset=ds,video=video,QP=qp,metric=metric,effect=label,
                method=method,method_full=METHODS[method][0],reference=reference,
                reference_full=METHODS[reference][0],method_value=float(a),reference_value=float(b),
                delta=float(a-b),relative_percent=float(100*(a-b)/b) if b else None,
                lower_is_better=True,common_bpp_min=lo,common_bpp_max=hi)


def main():
    manifest=json.loads((OLD/"eval_manifest.json").read_text())
    tasks=json.loads((HERE/"evaluation/task_manifest.json").read_text())
    receipts=[];missing=[]
    for t in tasks:
        q=t["task"];path=HERE/"tasks"/f"{q['dataset']}_{q['video']}_q{q['qp']}.json"
        if not path.exists():missing.append(str(path));continue
        x=json.loads(path.read_text())
        if x.get("status")!="PASS":missing.append(str(path));continue
        assert x["task"]==q and x["new_optimizer_updates"]==x["new_encodes"]==0
        assert x["source_replay_sha256"]==t["source_receipt_sha256"]
        assert x["latent_archive_sha256"]==t["latent_archive_sha256"]
        assert x["adagn_path_checks"]==1260
        receipts.append(x)
    points=[];endpoint=[];inputs=[];streams=[];frozen=[];paths=[];gammas=[];perframe=[];reuse=[]
    for x in receipts:
        t=x["task"];key=f"{t['dataset']}_{t['video']}_q{t['qp']}"
        assert sha(x["source_stream"])==x["stream_sha256"]
        assert len(x["endpoint_equivalence"])==2
        assert all(e["raw_output_bitwise"] and e["feature_bitwise"] and max(e["spatial_metric_errors"].values())<=1e-5 for e in x["endpoint_equivalence"])
        endpoint.append(dict(task=t,source_receipt=x["source_replay_receipt"],endpoint_checks=x["endpoint_equivalence"]))
        inputs.append(dict(task=t,latent_archive=x["latent_archive"],latent_archive_sha256=x["latent_archive_sha256"],
                           q=x["actual_qp_and_q_recon"],generator_hash=x["generator_module_hash_before"],
                           P_frame_checks=x["input_checks"]))
        streams.append(dict(task=t,path=x["source_stream"],sha256=x["stream_sha256"],bytes=Path(x["source_stream"]).stat().st_size,
                            I0_hash=x["i0_hash"],source_replay_receipt=x["source_replay_receipt"],receiver_hash=x["generator_module_hash_before"]))
        frozen.append(dict(task=t,generator=x["generator_module_hash_before"]==x["generator_module_hash_after"],
                           codec=x["codec_module_hashes_before"]==x["codec_module_hashes_after"],
                           metric=x["metric_module_hashes"],new_optimizer_updates=0))
        for method,point in x["results"].items():
            assert sha(point["feature_path"])==point["feature_sha256"]
            assert point["bitstream_sha256"]==x["stream_sha256"]
            points.append(dict(point,total_bits=point["real_bytes"]*8,
                               kbps_20fps=point["real_bytes"]*8/64*20/1000,source_fps_metadata=next(v for v in manifest[t["dataset"]] if v["video_index"]==t["video"]).get("fps"),
                               new_optimizer_updates=0,new_real_encodes=0))
            fr=read_csv(HERE/"raw"/f"{key}_{method}_frames.csv")
            tr=read_csv(HERE/"raw"/f"{key}_{method}_transitions.csv")
            assert len(fr)==64 and len(tr)==63
            perframe.extend(dict(dataset=t["dataset"],video=t["video"],QP=t["qp"],method=method,
                                 frame=int(a["frame"]),actual_qp=int(a["actual_qp"]),
                                 LPIPS=float(a["LPIPS"]),DISTS=float(a["DISTS"]),
                                 FloLPIPS=float(tr[i-1]["FloLPIPS"]) if i else None,
                                 actual_bpp=point["actual_bpp"]) for i,a in enumerate(fr))
        paths.append(dict(task=t,checks=x["adagn_path_checks"],source=HERE/"raw"/f"{key}_adagn_paths.csv"))
        gammas.extend(read_csv(HERE/"raw"/f"{key}_gamma_beta.csv"))
        reuse.append(dict(task=t,stream_SHA=x["stream_sha256"],source_receipt=x["source_replay_receipt"],
                          source_receipt_SHA=x["source_replay_sha256"],endpoint_bitwise=True))
    atomic_csv(HERE/"results/A_all_rd_points.csv",points)
    atomic_csv(HERE/"results/A_per_frame_quality.csv",perframe)
    oldpool=read_csv(OLD/"results/pooled_fid.csv")
    pooled=[];fid_checks=[]
    for ds,videos in manifest.items():
        for method in METHODS:
            for qp in (0,5,9):
                rows=[p for p in points if (p["dataset"],p["method"],p["QP"])==(ds,method,qp)]
                if len(rows)!=len(videos):continue
                val=pool(rows)
                pooled.append(dict(dataset=ds,method=method,method_full=METHODS[method][0],QP=qp,**val))
                if method in ("A00","A11"):
                    historical_method="O_original" if method=="A00" else "Y_originalBridge_fullG"
                    ref=next(r for r in oldpool if (r["dataset"],r["method"],int(r["QP"]))==(ds,historical_method,qp))
                    e={m:abs(val[m]-float(ref[m])) for m in ("FID","D_mu","D_cov")}
                    fid_checks.append(dict(dataset=ds,method=method,QP=qp,errors=e,
                                           GT_pool_exact=val["GT_pool_sha256"]==ref["GT_pool_sha256"]))
                    assert e["FID"]<=1e-3 and fid_checks[-1]["GT_pool_exact"]
    atomic_csv(HERE/"results/A_pooled_fid.csv",pooled)
    assert all(len({p["GT_pool_sha256"] for p in pooled if p["dataset"]==ds})<=1 for ds in manifest)

    idx={(r["dataset"],r["video"],r["QP"],r["method"]):r for r in points}
    pair=[];effects=[];interactions=[]
    for ds,videos in manifest.items():
        for v in videos:
            for qp in (0,5,9):
                key=(ds,v["video_index"],qp)
                rr={m:idx.get((*key,m)) for m in METHODS}
                if any(z is None for z in rr.values()):continue
                assert len({r["bitstream_sha256"] for r in rr.values()})==1
                for metric in SPATIAL:
                    for label,m,ref in EFFECTS:
                        a,b=rr[m][metric],rr[ref][metric]
                        row=effect_row("video_QP",ds,v["video_index"],qp,metric,label,m,ref,a,b)
                        pair.append(row);effects.append(row)
                    v00,v10,v01,v11=(rr[m][metric] for m in ("A00","A10","A01","A11"))
                    interactions.append(dict(level="video_QP",dataset=ds,video=v["video_index"],QP=qp,metric=metric,
                                             formula="Q11-Q10-Q01+Q00",interaction=v11-v10-v01+v00))
        for qp in (0,5,9):
            rr={m:next((r for r in pooled if (r["dataset"],r["QP"],r["method"])==(ds,qp,m)),None) for m in METHODS}
            if any(v is None for v in rr.values()):continue
            for metric in ("FID","D_mu","D_cov"):
                for label,m,ref in EFFECTS:
                    effects.append(effect_row("dataset_QP_pooled",ds,"ALL",qp,metric,label,m,ref,rr[m][metric],rr[ref][metric]))
                a,b,c,d=(rr[m][metric] for m in ("A00","A10","A01","A11"))
                interactions.append(dict(level="dataset_QP_pooled",dataset=ds,video="ALL",QP=qp,metric=metric,
                                         formula="Q11-Q10-Q01+Q00",interaction=d-b-c+a))
    atomic_csv(HERE/"results/A_same_stream_pairs.csv",pair)

    per_video=[];summary=[]
    for ds,videos in manifest.items():
        byvideo=[]
        for v in videos:
            curves={m:[p for p in points if (p["dataset"],p["video"],p["method"])==(ds,v["video_index"],m)] for m in METHODS}
            if any(len(c)!=3 for c in curves.values()):continue
            lo=max(min(p["actual_bpp"] for p in c) for c in curves.values())
            hi=min(max(p["actual_bpp"] for p in c) for c in curves.values())
            if hi<=lo:continue
            grid=np.linspace(math.log(lo),math.log(hi),20)
            for metric in SPATIAL:
                ys={m:interp(c,metric,grid) for m,c in curves.items()}
                for label,m,ref in EFFECTS:
                    a,b=float(ys[m].mean()),float(ys[ref].mean())
                    row=effect_row("video_equal_rate",ds,v["video_index"],"ALL",metric,label,m,ref,a,b,lo,hi)
                    per_video.append(row);byvideo.append(row)
                interactions.append(dict(level="video_equal_rate",dataset=ds,video=v["video_index"],QP="ALL",metric=metric,
                                         formula="Q11-Q10-Q01+Q00",interaction=float((ys["A11"]-ys["A10"]-ys["A01"]+ys["A00"]).mean())))
        for metric in SPATIAL:
            for label,m,ref in EFFECTS:
                rr=[r for r in byvideo if (r["metric"],r["effect"])==(metric,label)]
                if not rr:continue
                a,b=float(np.mean([r["method_value"] for r in rr])),float(np.mean([r["reference_value"] for r in rr]))
                summary.append(dict(effect_row("dataset_video_equal_weight",ds,"ALL","ALL",metric,label,m,ref,a,b),videos=len(rr),grid_points_per_video=20))
        curves={m:[p for p in pooled if (p["dataset"],p["method"])==(ds,m)] for m in METHODS}
        if all(len(c)==3 for c in curves.values()):
            lo=max(min(p["mean_actual_bpp"] for p in c) for c in curves.values())
            hi=min(max(p["mean_actual_bpp"] for p in c) for c in curves.values())
            if hi>lo:
                grid=np.linspace(math.log(lo),math.log(hi),20)
                for metric in ("FID","D_mu","D_cov"):
                    ys={m:np.interp(grid,np.log([p["mean_actual_bpp"] for p in sorted(c,key=lambda p:p["mean_actual_bpp"])]),
                                   [p[metric] for p in sorted(c,key=lambda p:p["mean_actual_bpp"])]) for m,c in curves.items()}
                    for label,m,ref in EFFECTS:
                        a,b=float(ys[m].mean()),float(ys[ref].mean())
                        summary.append(dict(effect_row("dataset_pooled_equal_rate",ds,"ALL","ALL",metric,label,m,ref,a,b,lo,hi),
                                            videos=len(videos),samples=64*len(videos),grid_points=20))
                    interactions.append(dict(level="dataset_pooled_equal_rate",dataset=ds,video="ALL",QP="ALL",metric=metric,
                                             formula="Q11-Q10-Q01+Q00",interaction=float((ys["A11"]-ys["A10"]-ys["A01"]+ys["A00"]).mean())))
    atomic_csv(HERE/"results/A_equal_rate_per_video.csv",per_video)
    atomic_csv(HERE/"results/A_equal_rate_summary.csv",summary)
    atomic_csv(HERE/"results/A_generator_parameter_effects.csv",effects+summary)
    atomic_csv(HERE/"results/A_interaction_terms.csv",interactions)
    # Aggregate measured linear input and scale/bias values without replacing original AdaGN computation.
    groups=defaultdict(list)
    for g in gammas:
        groups[(g["dataset"],g["method"],g["layer"],g["kind"])].append(g)
    gamma_summary=[]
    for (ds,m,layer,kind),rr in groups.items():
        gamma_summary.append(dict(dataset=ds,method=m,method_full=METHODS[m][0],layer=layer,kind=kind,
                                  observations=len(rr),mean=float(np.mean([float(x["mean"]) for x in rr])),
                                  mean_std=float(np.mean([float(x["std"]) for x in rr])),
                                  mean_rms=float(np.mean([float(x["rms"]) for x in rr])),
                                  minimum=min(float(x["minimum"]) for x in rr),maximum=max(float(x["maximum"]) for x in rr),
                                  distinct_input_hashes=len({x["input_hash"] for x in rr})))
    atomic_csv(HERE/"results/A_gamma_beta_statistics.csv",gamma_summary)
    atomic_json(HERE/"audits/endpoint_equivalence.json",dict(status="PASS" if len(endpoint)==39 and len(fid_checks)==18 else "PARTIAL",
                                                       checks=endpoint,pooled_FID_checks=fid_checks,
                                                       endpoint_raw_and_features_bitwise=all(all(z["raw_output_bitwise"] and z["feature_bitwise"] for z in x["endpoint_checks"]) for x in endpoint)))
    expanded_inputs=[]
    for item in inputs:
        z=np.load(item["latent_archive"])
        for i in range(63):
            source_hash={name:hashlib.sha256(z[name][i].tobytes()).hexdigest() for name in ("z0","zF")}
            for method in METHODS:
                expanded_inputs.append(dict(dataset=item["task"]["dataset"],video=item["task"]["video"],
                                            QP=item["task"]["qp"],frame=i+1,method=method,
                                            spatial_source="z0",stats_source="z0",
                                            body_source=METHODS[method][1],mod_source=METHODS[method][2],
                                            spatial_tensor_SHA=source_hash["z0"],
                                            stats_tensor_SHA=source_hash["z0"],
                                            q_recon_SHA=item["q"][i]["q_hash"],
                                            actual_generator_pre_hook_asserted_exact=True,
                                            actual_AdaGN_second_arg_verified_by_linear_hooks=True))
    atomic_json(HERE/"audits/actual_generator_inputs.json",dict(status="PASS" if len(expanded_inputs)==39*63*4 else "PARTIAL",
                                                             entries=expanded_inputs,source_tasks=inputs,
                                                             note="The raw path CSV stats_tensor field records the original decoder AdaGN argument before replacement. Actual z_stats equality is asserted at the hook and checked through gamma/beta linear input hashes."))
    atomic_json(HERE/"audits/adagn_conditioning_paths.json",dict(status="PASS" if sum(r["checks"] for r in paths)==49140 else "PARTIAL",
                                                              checks=sum(r["checks"] for r in paths),planned=49140,
                                                              raw_paths=[dict(task=p["task"],path=str(p["source"]),sha256=sha(p["source"])) for p in paths]))
    atomic_json(HERE/"audits/frozen_modules.json",dict(status="PASS" if all(x["generator"] and x["codec"] for x in frozen) and len(frozen)==39 else "PARTIAL",tasks=frozen))
    atomic_json(HERE/"audits/stream_and_source_equality.json",dict(status="PASS" if len(streams)==39 else "PARTIAL",tasks=streams))
    atomic_json(HERE/"audits/reuse_pointwise_verification.json",dict(status="PASS" if len(reuse)==39 else "PARTIAL",tasks=reuse))
    full=len(points)==156 and len(pooled)==36 and len(receipts)==39 and not missing
    integrity=dict(status="PASS" if full else "PARTIAL",training_updates=0,new_real_encodes=0,
                   reused_streams=len(streams),planned_streams=39,RD_completed=len(points),RD_planned=156,
                   FID_completed=len(pooled),FID_planned=36,
                   endpoint_bitwise=all(all(e["raw_output_bitwise"] and e["feature_bitwise"] for e in r["endpoint_checks"]) for r in endpoint),
                   AdaGN_path_checks=sum(r["checks"] for r in paths),AdaGN_path_planned=49140,
                   frozen_modules_pass=all(x["generator"] and x["codec"] for x in frozen),
                   FID_samples={ds:64*len(v) for ds,v in manifest.items()},missing=missing,
                   GT_pool_consistent=all(len({p["GT_pool_sha256"] for p in pooled if p["dataset"]==ds})<=1 for ds in manifest))
    atomic_json(HERE/"A_integrity.json",integrity)
    report="# v6.41 A generator body vs AdaGN parameter intervention\n\n"
    report+=f"Execution: **{integrity['status']}**. New training updates **0**; new real encodes **0**. RD **{len(points)}/156**, pooled FID **{len(pooled)}/36**, verified common RANS streams **{len(streams)}/39**. Endpoints bitwise exact: **{integrity['endpoint_bitwise']}**.\n\n"
    report+="All four methods use the same original bridge z0, q_recon and persisted stream. A00=(G0 body,G0 AdaGN), A01=(G0 body,GF AdaGN), A10=(GF body,G0 AdaGN), A11=(GF body,GF AdaGN). Negative quality delta means improvement; it does not mean fewer bits. A11 is the historical Y endpoint, not the full BG endpoint F.\n\n"
    report+="## Fixed equal-rate contrasts\n\n|Dataset|Effect|Metric|Method|Reference|Mean delta|Relative %|\n|---|---|---|---|---|---:|---:|\n"
    for r in summary:
        if r["metric"] in ("LPIPS","DISTS","FloLPIPS","FID"):
            report+=f"|{r['dataset']}|{r['effect']}|{r['metric']}|{r['method']}|{r['reference']}|{r['delta']:.6f}|{r['relative_percent']:.3f}|\n"
    report+="\n## Interaction on the same fixed intervals\n\nThe raw-unit interaction is Q11−Q10−Q01+Q00. It is not a fraction of a contribution.\n\n|Dataset|Metric|Interaction|Aggregation|\n|---|---|---:|---|\n"
    for r in interactions:
        if r["level"] in ("dataset_pooled_equal_rate",):
            report+=f"|{r['dataset']}|{r['metric']}|{r['interaction']:.6f}|pooled FID component curve|\n"
    for ds in manifest:
        for metric in SPATIAL:
            rr=[r for r in interactions if (r["level"],r["dataset"],r["metric"])==("video_equal_rate",ds,metric)]
            if rr:report+=f"|{ds}|{metric}|{float(np.mean([r['interaction'] for r in rr])):.6f}|video equal weight|\n"
    report+="\n## Direct same-stream anchors\n\n`results/A_same_stream_pairs.csv` contains every video/QP comparison. Each four-method set has exactly the same persisted stream SHA and measured bpp. The equal-rate grid is diagnostic interpolation, not additional encoded samples.\n\n"
    report+="\nWithin each method, all five AdaGN layers receive the same complete padded z0 latent. PyTorch's actual `var(dim=-1)` default correction is retained; descriptive v6.37 ddof0 statistics are not substituted. Body and modulation parameter effects are conditional on these fixed checkpoints and original bridge latent. A poor cross-combination can reflect weight incompatibility. The 13 videos, three QPs, and one trained seed do not establish broad generalization or a causal explanation of FID components. Publication status is recorded separately from this computational result.\n"
    (HERE/"A_REPORT.md").write_text(report)
    (HERE/"A_RESULTS_SUMMARY.md").write_text(report[:report.index("Within each method")])
    print("AGGREGATE",integrity["status"],len(points),len(pooled),flush=True)


if __name__=="__main__":main()
