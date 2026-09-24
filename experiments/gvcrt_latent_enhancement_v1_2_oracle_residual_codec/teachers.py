"""Restore actual oracle latents, rebase on stage-D ZERO, and calibrate only six Train2 frames."""
import json
import math
import time
from support import *


def prepare(out,fixed,lp):
    folder=out/"teachers_train6"
    folder.mkdir()
    samples=json.loads((V11/"fixed_samples.json").read_text())
    previous=read_csv(V11/"results_v1/oracle_metrics.csv")
    selections=[]
    records=[]
    tensors=[]
    for sample in samples:
        t0=time.perf_counter()
        sid,video,frame=sample["sample"],sample["video"],sample["frame"]
        entry=next(e for e in manifest()["train2"] if e["video_id"]==video)
        row=base_rows(entry)[frame]
        assert tensor_hash(row["ell"])==sample["ell_sha256"] and tensor_hash(row["q_recon"])==sample["q_recon_sha256"]
        ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
        x=source_frame(video,frame)
        with torch.no_grad():
            ell_c=fixed.ell_c(ell)
            xc=rgb01(fixed.g(ell_c,q))
            control=rgb01(fixed.g(fixed.control(ell),q))
        baseline=quality(x,xc,lp)
        for mode,y in [("ORIGINAL",rgb01(row["x_base"]).cuda()),("ZERO_D",xc),("CONTROL_C",control)]:
            records.append({"sample":sid,"video":video,"frame":frame,"mode":mode,**quality(x,y,lp)})
        candidates=[r for r in previous if r["sample"]==sid and r["objective"]=="MSE_LPIPS" and r["step"] in CFG["teacher"]["candidate_steps"]]
        selected=[r for r in candidates if eligible(r,baseline)]
        if not selected:
            selections.append({"sample":sid,"eligible":False,"baseline":baseline})
            save_json(folder/f"{sid}_unavailable.json",selections[-1])
            continue
        selected.sort(key=lambda r:(-r["psnr"],r["lpips"],r["step"],r["initialization"]!="ORIGINAL_START"))
        chosen=selected[0]
        cp=V11/"results_v1/oracle"/sid/f'{chosen["initialization"]}_MSE_LPIPS'/f'step_{int(chosen["step"]):03d}_delta.pt'
        saved=torch.load(cp,map_location="cpu",weights_only=False)
        with torch.no_grad():
            start=fixed.old_start(ell,chosen["initialization"])
            assert tensor_hash(start)==saved["ell_start_sha256"]
            oracle=start+saved["delta_latent"].cuda().float()
            rstar=oracle-ell_c
            restored=ell_c+rstar
            y=rgb01(fixed.g(oracle,q))
        qm=quality(x,y,lp)
        assert abs(qm["psnr"]-chosen["psnr"])<1e-8 and abs(qm["lpips"]-chosen["lpips"])<1e-7
        assert eligible(qm,baseline)
        check=json.loads((cp.parent/"check.json").read_text())
        selection={"sample":sid,"eligible":True,"video":video,"frame":frame,"qp":row["qp"],
                   "baseline":baseline,"teacher":qm,"selected_initialization":chosen["initialization"],
                   "selected_step":chosen["step"],"candidate_count":len(candidates),"eligible_candidate_count":len(selected),
                   "source_delta_file":str(cp),"source_delta_sha256":sha(cp),
                   "all_candidates":[{k:r[k] for k in ["initialization","step","psnr","lpips"]} for r in candidates],
                   "teacher_rebase_max_abs":(restored-oracle).abs().max().item(),
                   "historic_full_optimizer_run_seconds":check["seconds"],"teacher_lookup_restore_evaluate_seconds":time.perf_counter()-t0}
        data={"sample":sample,"ell_base":ell.cpu(),"ell_c":ell_c.cpu(),"ell_oracle":oracle.cpu(),"r_star":rstar.cpu(),
              "q_recon":q.cpu(),"valid_hw":CFG["valid_hw"],"reference_state":row["dpb"],"selection":selection}
        save_pt(folder/f"{sid}.pt",data)
        image(folder/f"{sid}_ZERO_D.png",xc)
        image(folder/f"{sid}_TEACHER.png",y)
        tensors.append(data)
        selections.append(selection)
        records.append({"sample":sid,"video":video,"frame":frame,"mode":"TEACHER",**qm})
        print("TEACHER",sid,chosen["initialization"],chosen["step"],"gain",qm["psnr"]-baseline["psnr"],flush=True)
    save_json(out/"teacher_selection.json",selections)
    write_csv(out/"teacher_metrics.csv",records)
    if len(tensors)!=6:
        raise RuntimeError("At least one fixed frame has no teacher meeting stage-D ZERO quality rule")
    r=torch.cat([t["r_star"] for t in tensors])
    scale=r.double().square().mean((0,2,3),keepdim=True).sqrt().float().clamp_min(CFG["normalization"]["min_scale"])
    rms=r.double().square().mean().sqrt().item()
    direct_deltas=[float(np.float32(rms*m)) for m in CFG["direct"]["delta_multipliers"]]
    ent=FactorizedEntropy(18)
    logistic=(scale*math.sqrt(3)/math.pi).clamp_min(1e-4)
    with torch.no_grad():
        ent.log_scale.copy_(torch.log(torch.expm1((logistic-1e-4).clamp_min(1e-6))))
    save_pt(out/"direct_shared.pt",{"entropy":ent.state_dict(),"method":"direct","deltas":direct_deltas,"shape":[18,68,120],
                                      "normalization_source":"only six Train2 r_star tensors","baseline_checkpoint_sha256":sha(V1/CFG["baseline_checkpoint"])})
    init={s["selected_initialization"] for s in selections}
    evaluation_init=next(iter(init)) if len(init)==1 else "ORIGINAL_START"
    calibration={"residual_channel_rms":scale.flatten().tolist(),"global_residual_rms":rms,
                 "residual_min":r.min().item(),"residual_max":r.max().item(),"residual_shape":[1,18,68,120],
                 "direct_deltas":direct_deltas,"direct_calibrated_logistic_scale":logistic.flatten().tolist(),
                 "evaluation_oracle_initialization":evaluation_init,"oracle_zero_init_checkpoint":CFG["teacher"]["evaluation_zero_start_checkpoint"],
                 "evaluation_optimizer":"Adam","evaluation_lr":0.001,"evaluation_steps":150,"evaluation_lambda_p":0.001,
                 "training_samples":6,"no_validation_calibration":True}
    save_json(out/"calibration.json",calibration)
    # New baseline-binding check only: stage-D ZERO must reproduce its saved Val6 RGB exactly.
    v=manifest()["val6"][0]
    row=base_rows(v)[1]
    with torch.no_grad():
        ec=fixed.ell_c(row["ell"].cuda().float())
        yc=rgb01(fixed.g(ec,row["q_recon"].cuda().float())).cpu()
    expected=torch.load(V1/"run_v1/quantized_d_val6_l2"/f'{v["video_id"]}_receiver.pt',map_location="cpu",weights_only=False)["frames"][1]
    assert torch.equal(yc,expected)
    fixed.assert_frozen()
    save_json(out/"baseline_binding.json",{"stage_D_zero_exact":True,"not_control_C":True,"fixed_models":fixed.fingerprint,
              "expected_V1_Val6_IP":{"psnr":26.450357387621523,"ms_ssim":0.8103940248437373,"lpips":0.3059831166174263}})
    return tensors,calibration


def generate_evaluation(out,entries,calibration,fixed,lp,existing):
    """Each new evaluation P frame is optimized once, never fed to network training/calibration."""
    folder=out/"sender_oracles"
    folder.mkdir(exist_ok=True)
    result=dict(existing)
    timing=[]
    for entry in entries:
        video=entry["video_id"]
        for row in base_rows(entry):
            if row["type"]=="I" or (video,row["frame"]) in result:
                continue
            frame=row["frame"]
            path=folder/f"{video}_f{frame:04d}.pt"
            if path.exists():
                raise FileExistsError("Do not silently reuse a potentially different oracle setting")
            torch.cuda.synchronize()
            begin=time.perf_counter()
            x=source_frame(video,frame)
            ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
            with torch.no_grad():
                ec=fixed.ell_c(ell)
                start=fixed.old_start(ell,calibration["evaluation_oracle_initialization"]).detach()
                xc=rgb01(fixed.g(ec,q))
            baseline=quality(x,xc,lp)
            delta=nn.Parameter(torch.zeros_like(start))
            opt=torch.optim.Adam([delta],lr=calibration["evaluation_lr"])
            best=None
            curve=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(151):
                opt.zero_grad(set_to_none=True)
                y=rgb01(fixed.g(start+delta,q))
                mse=F.mse_loss(y,x)
                perceptual=lp(y,x,normalize=True).mean()
                loss=mse+0.001*perceptual
                if not torch.isfinite(loss) or not torch.isfinite(delta).all() or delta.abs().max()>10:
                    raise FloatingPointError(f"Oracle numerical failure {video}:{frame}:{step}; no config change")
                if step in CFG["teacher"]["evaluation_snapshots"]:
                    qm=quality(x,y.detach(),lp)
                    if eligible(qm,baseline) and (best is None or qm["psnr"]>best["metrics"]["psnr"]):
                        best={"latent":(start+delta).detach().clone(),"metrics":qm,"step":step}
                    curve.append({"step":step,**qm,"loss":loss.item()})
                if step<150:
                    loss.backward()
                    assert delta.grad is not None and torch.isfinite(delta.grad).all()
                    opt.step()
            torch.cuda.synchronize()
            seconds=time.perf_counter()-begin
            chosen=best["latent"] if best else ec.detach().clone()
            data={"ell_base":ell.detach().cpu(),"ell_c":ec.detach().cpu(),"ell_oracle":chosen.cpu(),"r_star":(chosen-ec).detach().cpu(),
                  "q_recon":q.detach().cpu(),"valid_hw":CFG["valid_hw"],"reference_state":row["dpb"],
                  "selection":{"video":video,"frame":frame,"qp":row["qp"],"eligible":best is not None,
                               "selected_step":best["step"] if best else None,"selected_initialization":calibration["evaluation_oracle_initialization"],
                               "baseline":baseline,"teacher":best["metrics"] if best else baseline,
                               "full_latent_optimization_seconds":seconds,"curve":curve,
                               "timing_includes":"source PNG read, GPU copies, C/D ZERO initialization, 150 Adam steps, snapshot metrics; model loading and base cache creation excluded"}}
            save_pt(path,data)
            result[(video,frame)]=data
            timing.append({"video":video,"frame":frame,"optimization_seconds":seconds,"updates":150,"eligible":best is not None,
                           "peak_allocated_bytes":torch.cuda.max_memory_allocated()})
            print("SENDER_ORACLE",video,frame,"seconds",seconds,"step",data["selection"]["selected_step"],flush=True)
            del delta,opt,y,mse,perceptual,loss
            torch.cuda.empty_cache()
    fixed.assert_frozen()
    return result,timing
