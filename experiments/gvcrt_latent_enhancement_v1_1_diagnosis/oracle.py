"""Single-image finite-budget latent optimization. No neural-network parameter updates."""
import json
import time
import traceback
from common import *


def run(out):
    configure()
    cfg=CFG["oracle"]
    samples=json.loads((HERE/"fixed_samples.json").read_text())
    g=generator()
    enh,control,entropy,ck=models(cfg["checkpoint"])
    perceptual=lpips_model()
    before={"g":digest_state(g),"enh":digest_state(enh),"control":digest_state(control),"entropy":digest_state(entropy)}
    metrics=[]
    references=[]
    failures=[]
    run_checks=[]
    images=out/"oracle"
    images.mkdir()
    cached_video=None
    cached_rows=None
    for sample in samples:
        sid,video,frame=sample["sample"],sample["video"],sample["frame"]
        folder=images/sid
        folder.mkdir()
        if video!=cached_video:
            cached_rows=torch.load(V1/"run_v1/base"/f"{video}_q0.pt",map_location="cpu",weights_only=False)
            cached_video=video
        row=cached_rows[frame]
        assert tensor_hash(row["ell"])==sample["ell_sha256"] and tensor_hash(row["q_recon"])==sample["q_recon_sha256"]
        assert row["qp"]==sample["qp"] and row["type"]=="P" and row["dpb"]==sample["reference_state"]
        x=source_frame(video,frame)
        ell=row["ell"].cuda().float().detach().clone()
        q=row["q_recon"].cuda().float().detach().clone()
        original_hash,q_hash=tensor_hash(ell),tensor_hash(q)
        with torch.no_grad():
            zero_symbols=torch.zeros((1,8,17,30),dtype=torch.int16)
            u_zero=dequantize(zero_symbols,cfg["zero_start_delta"])
            zero_latent=ell+enh.decoder(u_zero,ell)
            ref_frames={"ORIGINAL_FP16":rgb01(row["x_base"]).cuda(),"REFERENCE_FP32":rgb01(g(ell,q)),
                        "BASE_ONLY_C":rgb01(g(control(ell),q)),"ZERO_START_C_DELTA2":rgb01(g(zero_latent,q))}
        image(folder/"source.png",x)
        x1,y1,x2,y2=sample["crop_xyxy"]
        image(folder/"source_crop.png",x[:,:,y1:y2,x1:x2])
        for name,y in ref_frames.items():
            r={"sample":sid,"video":video,"frame":frame,"mode":name,"qp":row["qp"],**quality(x,y,perceptual)}
            references.append(r)
            image(folder/f"{name}.png",y)
            image(folder/f"{name}_crop.png",y[:,:,y1:y2,x1:x2])
        starts={"ORIGINAL_START":ell.detach().clone(),"ZERO_START":zero_latent.detach().clone()}
        for initialization in cfg["initializations"]:
            for objective in cfg["objectives"]:
                run_id=f"{initialization}_{objective}"
                sub=folder/run_id
                sub.mkdir()
                start_latent=starts[initialization].detach().clone()
                start_hash=tensor_hash(start_latent)
                delta=torch.nn.Parameter(torch.zeros_like(start_latent))
                optimizer=torch.optim.Adam([delta],lr=cfg["lr"])
                assert len(optimizer.param_groups)==1 and optimizer.param_groups[0]["params"]==[delta]
                weight=cfg["lambda_p"] if objective=="MSE_LPIPS" else 0.0
                torch.cuda.reset_peak_memory_stats()
                start_time=time.perf_counter()
                max_grad=0.0
                last_update=0
                try:
                    with (sub/"optimization_curve.jsonl").open("x") as logfile:
                        for step in range(cfg["steps"]+1):
                            optimizer.zero_grad(set_to_none=True)
                            # No no_grad here: only delta is optimized, through frozen G.
                            y=rgb01(g(start_latent+delta,q))
                            mse=torch.nn.functional.mse_loss(y,x)
                            lp=perceptual(y,x,normalize=True).mean() if weight else torch.zeros((),device="cuda")
                            loss=mse+weight*lp
                            if not torch.isfinite(loss) or not torch.isfinite(delta).all() or delta.abs().max()>cfg["abnormal_delta_max_abs_stop"] or mse>cfg["abnormal_mse_stop"]:
                                raise FloatingPointError("Non-finite or predeclared abnormal tensor/loss bound exceeded")
                            if step in cfg["snapshots"]:
                                m={"sample":sid,"video":video,"frame":frame,"qp":row["qp"],"initialization":initialization,
                                   "objective":objective,"step":step,"lambda_p":weight,"checkpoint":cfg["checkpoint"],
                                   "zero_start_delta":cfg["zero_start_delta"],**quality(x,y.detach(),perceptual),
                                   **stats(delta,"delta_latent"),"ell_start_l2":start_latent.double().norm().item(),
                                   "relative_delta_l2":delta.double().norm().item()/max(start_latent.double().norm().item(),1e-12),
                                   "objective_value":loss.item(),"transmittable":False,"single_frame_diagnostic":True}
                                metrics.append(m)
                                image(sub/f"step_{step:03d}.png",y)
                                image(sub/f"step_{step:03d}_crop.png",y[:,:,y1:y2,x1:x2])
                                with (sub/f"step_{step:03d}_delta.pt").open("xb") as f:
                                    torch.save({"delta_latent":delta.detach().cpu(),"sample":sample,"initialization":initialization,
                                                "objective":objective,"step":step,"ell_start_sha256":start_hash},f)
                                print("ORACLE",sid,initialization,objective,step,m["psnr"],m["lpips"],flush=True)
                            record={"step":step,"mse":mse.item(),"lpips_loss_component":lp.item(),"lambda_p":weight,
                                    "loss":loss.item(),"delta_l2":delta.detach().norm().item(),"delta_max_abs":delta.detach().abs().max().item()}
                            if step<cfg["steps"]:
                                loss.backward()
                                if delta.grad is None or not torch.isfinite(delta.grad).all():
                                    raise FloatingPointError("Invalid latent gradient")
                                grad=delta.grad.norm().item()
                                max_grad=max(max_grad,grad)
                                record["delta_grad_l2"]=grad
                                assert all(p.grad is None for p in g.parameters())
                                assert all(p.grad is None for p in enh.parameters())
                                assert all(p.grad is None for p in control.parameters())
                                optimizer.step()
                                last_update=step+1
                            logfile.write(json.dumps(record,allow_nan=False)+"\n")
                            logfile.flush()
                    assert last_update==150 and max_grad>0
                    assert tensor_hash(start_latent)==start_hash
                    assert tensor_hash(ell)==original_hash and tensor_hash(q)==q_hash
                    assert tensor_hash(row["ell"])==sample["ell_sha256"]
                    assert digest_state(g)==before["g"]
                    torch.cuda.synchronize()
                    check={"sample":sid,"initialization":initialization,"objective":objective,"updates":last_update,
                           "generator_unchanged":True,"inputs_unchanged":True,"max_gradient_l2":max_grad,
                           "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"seconds":time.perf_counter()-start_time}
                    run_checks.append(check)
                    save_json(sub/"check.json",check)
                except (FloatingPointError,torch.cuda.OutOfMemoryError) as exc:
                    failure={"sample":sid,"initialization":initialization,"objective":objective,"last_update":last_update,
                             "error":repr(exc),"traceback":traceback.format_exc(),"configuration_changed":False}
                    failures.append(failure)
                    save_json(sub/"FAILED.json",failure)
                    print("ORACLE_FAILURE",failure,flush=True)
                del delta,optimizer
                torch.cuda.empty_cache()
        assert row["dpb"]==sample["reference_state"]
    after={"g":digest_state(g),"enh":digest_state(enh),"control":digest_state(control),"entropy":digest_state(entropy)}
    assert before==after
    write_csv(out/"oracle_metrics.csv",metrics)
    write_csv(out/"oracle_references.csv",references)
    # Keep every logged point; Pareto is PSNR maximize and LPIPS minimize, per sample.
    pareto=[]
    for sample in samples:
        candidates=[dict(r,kind="ORACLE") for r in metrics if r["sample"]==sample["sample"]]
        candidates += [dict(r,kind="REFERENCE",step=0,initialization=r["mode"],objective="NONE") for r in references if r["sample"]==sample["sample"]]
        for r in candidates:
            dominating=[o for o in candidates if o["psnr"]>=r["psnr"] and o["lpips"]<=r["lpips"] and (o["psnr"]>r["psnr"] or o["lpips"]<r["lpips"])]
            r["nondominated_psnr_lpips"]=not dominating
            r["dominated_by_count"]=len(dominating)
            pareto.append(r)
    write_csv(out/"oracle_pareto.csv",pareto)
    save_json(out/"oracle_checks.json",{"model_weights_unchanged":True,"fingerprints":before,"runs":run_checks,"failures":failures,
              "network_updates":0,"total_latent_optimizer_updates":sum(r["updates"] for r in run_checks),
              "no_base_state_feedback":True,"finite_budget_not_theoretical_bound":True})
    del g,enh,control,perceptual
    torch.cuda.empty_cache()
