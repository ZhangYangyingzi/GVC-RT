"""Same-model source interventions. Only REAL is an actual transmitted operating point."""
import concurrent.futures
import json
import time
from collections import defaultdict
from common import *


def render(g,decoder,ell,q,u):
    # Normal synthesis API has no original/source or reference-cache argument.
    correction=decoder(u,ell)
    latent=ell+correction
    raw=g(latent,q)
    return correction,latent,raw


def run(out):
    configure()
    g=generator()
    enh,control,entropy,ck=models(CFG["ablation"]["checkpoint"])
    perceptual=lpips_model()
    fingerprints={"g":digest_state(g),"enh":digest_state(enh),"control":digest_state(control),"entropy":digest_state(entropy)}
    manifest=json.loads((HERE/"manifest.json").read_text())
    entries=manifest["val6"]
    donors={r["target_video"]:r["donor_video"] for r in json.loads((HERE/"donor_mapping.json").read_text())}
    packets={}
    for entry in entries:
        video=entry["video_id"]
        for level in CFG["ablation"]["levels"]:
            idx,delta=level["index"],level["delta"]
            path=V1/f"run_v1/quantized_d_val6_l{idx}"/f"{video}.gle"
            hw,streams=read_container(path,V1/"run_v1/base"/f"{video}_q0.bin")
            assert hw==tuple(CFG["valid_hw"])
            decoded=[]
            for frame,stream in enumerate(streams):
                if not stream:
                    assert frame==0
                    decoded.append(None)
                else:
                    s,d,l=entropy.decode(stream)
                    assert d==delta and l==idx and tuple(s.shape)==(1,8,17,30)
                    decoded.append(s)
            packets[(video,idx)]=decoded
    # Decode real base files; previous base codec audit is reused, no encoder rerun.
    vcfg=json.loads((V1/"config.json").read_text())
    i,p,_=load_models(vcfg)
    all_records=[]
    equalities=[]
    invariants=[]
    reused=0
    calculated=0
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=CFG["metric_workers"])
    video_dir=out/"ablation_videos"
    video_dir.mkdir()
    try:
        for entry in entries:
            video=entry["video_id"]
            rows=decode_base(i,p,V1/"run_v1/base"/f"{video}_q0.bin")
            baseline_metadata=json.loads((V1/"run_v1/base"/f"{video}_q0.json").read_text())["decode"]
            for row,old in zip(rows,baseline_metadata):
                assert row["dpb"]==old["dpb"] and row["qp"]==old["qp"]
                assert tensor_hash(row["x_base"])==old["x_base_sha256"]
            state_before=dpb_hash(p)
            saved_real={idx:torch.load(V1/f"run_v1/quantized_d_val6_l{idx}"/f"{video}_receiver.pt",map_location="cpu",weights_only=False)["frames"] for idx in range(3)}
            outputs=defaultdict(list)
            futures=[]
            for row in rows:
                frame=row["frame"]
                x=source_frame(video,frame,"cpu")
                a=x.numpy()[0]
                cache={}
                per_frame=[]
                def record(mode,delta,level,y,more):
                    nonlocal reused,calculated
                    y=y.detach().cpu()
                    key=tensor_hash(y)
                    if key in cache:
                        old_y,metric=cache[key]
                        assert torch.equal(y,old_y)
                        y=old_y
                        reused+=1
                    else:
                        b=y.numpy()[0]
                        with torch.no_grad():
                            lp=perceptual(x.cuda(),y.cuda(),normalize=True).item()
                        def cpu_metric(a=a,b=b,lp=lp):
                            return {"psnr":calc_psnr(a,b,data_range=1),"ms_ssim":float(calc_msssim_rgb(a,b,data_range=1)),"lpips":lp}
                        metric=pool.submit(cpu_metric)
                        cache[key]=(y,metric)
                        calculated+=1
                    item={"video":video,"frame":frame,"type":row["type"],"qp":row["qp"],"level":level,"delta":delta,
                          "mode":mode,"checkpoint":CFG["ablation"]["checkpoint"],
                          "diagnostic_intervention":mode in ["ZERO","WRONG_SOURCE","CONTINUOUS"],
                          "formal_enhancement_bits":(V1/f"run_v1/quantized_d_val6_l{level}"/f"{video}.gle").stat().st_size*8 if mode=="REAL" else None,
                          "rate_field_scope":"whole sequence for REAL only" if mode=="REAL" else "no diagnostic RD rate",
                          **more}
                    futures.append((item,metric))
                    per_frame.append((item,metric))
                    outputs[(level,mode)].append(y)
                with torch.no_grad():
                    original=rgb01(row["x_base"])
                    record("ORIGINAL_FP16",None,-1,original,{})
                    if row["type"]=="I":
                        for mode in ["REFERENCE_FP32","BASE_ONLY"]:
                            record(mode,None,-1,original,{})
                        for level in CFG["ablation"]["levels"]:
                            for mode in CFG["ablation"]["modes"]:
                                record(mode,level["delta"],level["index"],original,{"I_unchanged":True})
                        continue
                    ell=row["ell"].cuda().float()
                    q=row["q_recon"].cuda().float()
                    before_ell,before_q=tensor_hash(ell),tensor_hash(q)
                    record("REFERENCE_FP32",None,-1,rgb01(g(ell,q)),{})
                    base_only_latent=control(ell)
                    record("BASE_ONLY",None,-1,rgb01(g(base_only_latent,q)),stats(base_only_latent-ell,"delta_ell"))
                    continuous_u=enh.encoder(torch.nn.functional.pad(x.cuda(),(0,0,0,8),mode="replicate"),ell)
                    for level in CFG["ablation"]["levels"]:
                        idx,delta=level["index"],level["delta"]
                        real_s=packets[(video,idx)][frame]
                        assert torch.equal((continuous_u/delta).round().cpu().short(),real_s)
                        donor_s=packets[(donors[video],idx)][frame]
                        assert real_s.shape==donor_s.shape
                        zero_s=torch.zeros_like(real_s)
                        zero_u=dequantize(zero_s,delta)
                        assert torch.equal(zero_u,torch.zeros_like(zero_u))
                        # Compute ZERO first to allow exact same-model comparisons without bypass.
                        dz,lz,rz=render(g,enh.decoder,ell,q,zero_u)
                        for mode,s,u in [("REAL",real_s,dequantize(real_s,delta)),
                                         ("ZERO",zero_s,zero_u),
                                         ("WRONG_SOURCE",donor_s,dequantize(donor_s,delta)),
                                         ("CONTINUOUS",None,continuous_u)]:
                            if mode=="ZERO":
                                d,l,r=dz,lz,rz
                            else:
                                d,l,r=render(g,enh.decoder,ell,q,u)
                            if mode=="REAL":
                                assert torch.equal(rgb01(r).cpu(),saved_real[idx][frame])
                                checks={"video":video,"frame":frame,"delta":delta,
                                        **{f"u_{k}":v for k,v in difference(u,zero_u).items()},
                                        **{f"delta_ell_{k}":v for k,v in difference(d,dz).items()},
                                        **{f"generator_latent_{k}":v for k,v in difference(l,lz).items()},
                                        **{f"raw_rgb_{k}":v for k,v in difference(r,rz).items()},
                                        **{f"valid_rgb_{k}":v for k,v in difference(rgb01(r),rgb01(rz)).items()}}
                                equalities.append(checks)
                                if checks["u_equal"]:
                                    assert checks["delta_ell_equal"] and checks["generator_latent_equal"] and checks["raw_rgb_equal"]
                            more={"symbol_nonzero_fraction":float((s!=0).float().mean()) if s is not None else None,
                                  **stats(u,"u_hat"),**stats(d,"delta_ell"),
                                  "donor_video":donors[video] if mode=="WRONG_SOURCE" else None,
                                  "source_available_to_synthesis":False,"zero_uses_same_D_G":mode=="ZERO"}
                            record(mode,delta,idx,rgb01(r),more)
                    assert tensor_hash(ell)==before_ell and tensor_hash(q)==before_q
                    assert tensor_hash(row["ell"])==baseline_metadata[frame]["ell_sha256"]
            assert dpb_hash(p)==state_before
            invariants.append({"video":video,"base_state_unchanged":True,"all_base_frames_match_v1":True,
                               "all_real_outputs_match_v1":True,"ell_and_q_unchanged":True})
            # Resolve independently computed identical metric implementation; no rate assigned to interventions.
            video_rows=[]
            for item,future in futures:
                item.update(future.result())
                video_rows.append(item)
            index={(r["frame"],r["level"],r["mode"]):r for r in video_rows}
            for r in video_rows:
                if r["level"]>=0:
                    for comparator in ["ZERO","REAL","WRONG_SOURCE"]:
                        other=index[(r["frame"],r["level"],comparator)]
                        for metric in ["psnr","ms_ssim","lpips"]:
                            r[f"{metric}_minus_{comparator.lower()}"]=r[metric]-other[metric]
            all_records.extend(video_rows)
            # Each mode has a corresponding 16-frame video, with original I and native cadence.
            for (level,mode),frames in outputs.items():
                label=f"l{level}_{mode}" if level>=0 else mode
                write_video(video_dir/f"{video}_{label}.mkv",frames,entry["metadata"]["avg_frame_rate"])
            write_csv(out/f"ablation_{video}.csv",video_rows)
            print("ABLATION",video,"complete",len(video_rows),flush=True)
        write_csv(out/"source_ablation_frames.csv",all_records)
        summaries=[]
        keys=sorted(set((r["level"],r["mode"]) for r in all_records))
        numeric=["psnr","ms_ssim","lpips","symbol_nonzero_fraction"]+[f"{p}_{s}" for p in ["u_hat","delta_ell"] for s in ["mean","std","l2","rms","max_abs"]]+[f"{m}_minus_{c}" for c in ["zero","real","wrong_source"] for m in ["psnr","ms_ssim","lpips"]]
        for video in [e["video_id"] for e in entries]+["ALL_VAL6"]:
            for level,mode in keys:
                selected=[r for r in all_records if r["level"]==level and r["mode"]==mode and (video=="ALL_VAL6" or r["video"]==video)]
                for scope in ["P","IP"]:
                    subset=[r for r in selected if scope=="IP" or r["type"]=="P"]
                    item={"video":video,"scope":scope,"level":level,"delta":subset[0]["delta"],"mode":mode,"frames":len(subset),
                          "diagnostic_intervention":subset[0]["diagnostic_intervention"],"formal_rate_kbps":None,
                          "input_statistics_scope":"P frames only; missing on I frames"}
                    for key in numeric:
                        vals=[r[key] for r in subset if r.get(key) is not None]
                        item[key]=float(np.mean(vals)) if vals else None
                    if mode=="REAL":
                        es=[e for e in entries if video=="ALL_VAL6" or e["video_id"]==video]
                        item["formal_rate_kbps"]=sum((V1/f"run_v1/quantized_d_val6_l{level}"/f'{e["video_id"]}.gle').stat().st_size*8 for e in es)/sum(e["duration_seconds"] for e in es)/1000
                    summaries.append(item)
        write_csv(out/"source_ablation.csv",summaries)
        write_csv(out/"exact_equality.csv",equalities)
        after={"g":digest_state(g),"enh":digest_state(enh),"control":digest_state(control),"entropy":digest_state(entropy)}
        assert fingerprints==after
        save_json(out/"ablation_checks.json",{"weights_unchanged":True,"fingerprints":fingerprints,"sequences":invariants,
                  "UZERO_required":False,"zero_integer_dequantization_exactly_zero":True,
                  "independent_decoded_enhancement_files":18,"all_real_outputs_match_v1":True,
                  "metric_evaluations":calculated,"exact_equal_metric_reuses":reused,
                  "delta2_real_zero_exact_P_frames":sum(r["delta"]==2 and r["raw_rgb_equal"] for r in equalities)})
    finally:
        pool.shutdown(wait=True)
    del g,enh,control,perceptual,i,p
    torch.cuda.empty_cache()
