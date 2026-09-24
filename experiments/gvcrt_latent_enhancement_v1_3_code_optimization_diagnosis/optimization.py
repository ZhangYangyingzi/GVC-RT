"""Frozen-model optimization of 4080 code values only; actual hard-byte checkpoint selection."""
import concurrent.futures
from bridge import *


class Data:
    def __init__(self,out,bundle):
        self.out,self.b=out,bundle
        self.base={}
        self.contexts={}
        self.input_times=[]

    def get(self,spec):
        sid=spec["sample"]
        if sid in self.contexts:
            return self.contexts[sid]
        t=time.perf_counter()
        video=spec["video"]
        if video not in self.base:
            entry=next(e for e in entries(spec["split"]) if e["video_id"]==video)
            self.base[video]=base_rows(entry)
        row=self.base[video][spec["frame"]]
        with torch.no_grad():
            ec=self.b.fixed.ell_c(row["ell"].cuda().float()).cpu()
        context_data={"ell_c":ec,"q_recon":row["q_recon"].float(),"row":row,"spec":spec}
        self.input_times.append({"sample":sid,"cached_base_state_and_ZERO_seconds":time.perf_counter()-t,
                                 "note":"original base decoder cache is reused; actual independent receiver base decoding timed separately"})
        self.contexts[sid]=context_data
        return context_data

    def encoder(self,spec):
        path=self.out/"input_cache"/f'{spec["sample"]}.pt'
        if path.exists():
            return torch.load(path,map_location="cpu",weights_only=False)
        data,seconds=load_sample(spec)
        context=self.get(spec)
        assert torch.equal(context["ell_c"],data["ell_c"])
        assert torch.equal(context["q_recon"],data["q_recon"])
        assert context["row"]["dpb"]==data["reference_state"]
        torch.cuda.synchronize()
        t=time.perf_counter()
        with torch.no_grad():
            u=self.b.net.analyze(data["r_star"].cuda(),context["ell_c"].cuda())
        torch.cuda.synchronize()
        result={"encoder_u":u.cpu(),"spec":spec,"teacher_cache_read_seconds":seconds,
                "encoder_forward_seconds":time.perf_counter()-t}
        save_pt(path,result)
        return result


class Metrics:
    def __init__(self,bundle):
        self.b=bundle
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["metric_workers"])

    def submit(self,x,y):
        with torch.no_grad():
            lp=self.b.lp(x,y,normalize=True).item()
        a=x.detach().cpu().numpy()[0]
        b=y.detach().cpu().numpy()[0]
        def compute():
            mse=np.square(a.astype(np.float64)-b.astype(np.float64)).mean().item()
            return {"mse":mse,"psnr":calc_psnr(a,b,data_range=1),
                    "ms_ssim":float(calc_msssim_rgb(a,b,data_range=1)),"lpips":lp,"L_image":mse+.001*lp}
        return self.pool.submit(compute)


def reference(data,spec,bundle,metrics):
    context=data.get(spec)
    c=bundle.context(context)
    x=source_frame(spec["video"],spec["frame"])
    u=data.encoder(spec)["encoder_u"].cuda()
    teacher,_=load_sample(spec)
    with torch.no_grad():
        zero,r0=bundle.render(torch.zeros_like(u),c)
        enc,rc=bundle.render(u,c)
        quant,rq=bundle.render(u.div(DELTA).round()*DELTA,c)
        full=rgb01(bundle.fixed.g(teacher["ell_oracle"].cuda(),c[1]))
        original=rgb01(context["row"]["x_base"]).cuda()
    results={k:metrics.submit(x,y).result() for k,y in [("ZERO",zero),("ENCODER_CONTINUOUS",enc),("ENCODER_QUANTIZED",quant),("TEACHER",full),("ORIGINAL_QP0",original)]}
    assert torch.count_nonzero(r0)==0
    return results


def actual_snapshot(folder,spec,step,symbols,bundle):
    stream=folder/f"step_{step:03d}.orc"
    frames=[None]+[torch.zeros(CONFIG["code_shape"]) for _ in range(15)]
    frames[spec["frame"]]=symbols.detach().cpu()
    base=V1/"run_v1/base"/f'{spec["video"]}_q0.bin'
    stats=ORC.encode(stream,base,MODEL,1,3,DELTA,(8,17,30),frames,bundle.net.entropy)
    decoded=ORC.decode(stream,base,MODEL,bundle.net.entropy)
    assert torch.equal(decoded["frames"][spec["frame"]],symbols.detach().cpu().short())
    packet=stats["frames"][spec["frame"]]
    attributed=packet["actual_bits"]+(ORC.HEADER.size*8+8)/15
    return {"diagnostic_container_bits":stats["file_bits"],"actual_frame_packet_bits":packet["actual_bits"],
            "actual_frame_share_bits":attributed,"R_actual_bpp":attributed/PIXELS,"zero_grid_skip":packet["kind"]=="ZERO",
            "roundtrip_exact":True,"diagnostic_stream":str(stream)}


def optimize(out,stage,spec,initialization,lam,bundle,data,metrics,scales,refs,quantized=False):
    folder=out/"optimizations"/stage/spec["sample"]
    folder.mkdir(parents=True)
    context_data=data.get(spec)
    context=bundle.context(context_data)
    ec_hash,q_hash=tensor_hash(context[0]),tensor_hash(context[1])
    x=source_frame(spec["video"],spec["frame"])
    if initialization=="ENCODER_START":
        initial=data.encoder(spec)["encoder_u"].cuda()
    else:
        initial=torch.zeros(CONFIG["code_shape"],device="cuda")
    variable=nn.Parameter((initial/DELTA if quantized else initial).detach().clone())
    assert variable.numel()==4080
    opt=torch.optim.Adam([variable],lr=scales["quant_domain_z_lr"] if quantized else scales["physical_u_lr"],
                         betas=tuple(CONFIG["adam_betas"]),eps=CONFIG["adam_eps"])
    records=[]
    cache={}
    hits=0
    cache_test=None
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def differentiate():
        if quantized:
            rounded=variable.detach().round()
            ORC.checked_symbols(rounded)
            s=rounded+(variable-variable.detach())  # exact hard forward, identity STE derivative
            u=s*DELTA
        else:
            s=None
            u=variable
        y,r=bundle.render(u,context)
        mse=F.mse_loss(y,x)
        lp=bundle.lp(y,x,normalize=True).mean()
        image_loss=mse+.001*lp
        rate=bundle.net.entropy.bits(s,DELTA)/PIXELS if quantized else torch.zeros((),device="cuda")
        loss=image_loss+lam*rate
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite code objective")
        gradient=torch.autograd.grad(loss,variable)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite code gradient")
        return gradient.detach(),{"mse":mse.item(),"lpips_loss":lp.item(),"L_image_float32":image_loss.item(),
                                  "R_est_bpp":rate.item(),"objective_est":loss.item()}

    with (folder/"curve.jsonl").open("x") as logfile:
        for step in range(CONFIG["steps"]+1):
            opt.zero_grad(set_to_none=True)
            cache_key=tensor_hash(variable.detach().round()) if quantized else None
            cache_hit=quantized and cache_key in cache
            if cache_hit:
                gradient,values=cache[cache_key]
                hits+=1
                if cache_test is None:
                    fresh,newvalues=differentiate()
                    difference=(gradient-fresh).abs().max().item()
                    assert torch.allclose(gradient,fresh,atol=1e-7,rtol=1e-5)
                    assert abs(values["objective_est"]-newvalues["objective_est"])<1e-8
                    cache_test={"max_abs_gradient_difference":difference,"allclose":True,"same_hard_symbols":True}
            else:
                gradient,values=differentiate()
                if quantized:
                    cache[cache_key]=(gradient,values)
            grad_norm=gradient.double().norm().item()
            if step==0 and initialization=="ZERO_START":
                assert grad_norm>0,"ZERO_START gradient disconnected"
            curve={"step":step,"lambda":lam,**values,"gradient_l2":grad_norm,
                   "variable_l2":variable.detach().double().norm().item(),"variable_max_abs":variable.detach().abs().max().item(),
                   "hard_gradient_cache_hit":cache_hit}
            if step in CONFIG["snapshots"]:
                with torch.no_grad():
                    s=variable.round() if quantized else None
                    u=s*DELTA if quantized else variable
                    y,r=bundle.render(u,context)
                    if initialization=="ZERO_START" and step==0:
                        assert torch.count_nonzero(r)==0
                        assert torch.equal(y,rgb01(bundle.fixed.g(context[0],context[1])))
                    canonical=bundle.net.synthesize(u,context[0])
                    assert torch.equal(r,canonical)
                record={"sample":spec["sample"],"video":spec["video"],"frame":spec["frame"],"group":spec["group"],"split":spec["split"],
                        "stage":stage,"initialization":initialization,"step":step,"lambda":lam,"quantized":quantized,
                        "code_domain":"z = u/delta" if quantized else "physical prequantizer u",
                        "code_l2":u.detach().double().norm().item(),"code_rms":u.detach().double().square().mean().sqrt().item(),
                        "residual_l2":r.detach().double().norm().item(),"gradient_l2":grad_norm,"R_est_bpp":values["R_est_bpp"],
                        "estimated_symbol_bits":values["R_est_bpp"]*PIXELS,"elapsed_seconds":time.perf_counter()-start,
                        "symbol_nonzero_fraction":(s!=0).float().mean().item() if s is not None else None}
                if quantized:
                    record.update(actual_snapshot(folder,spec,step,s,bundle))
                future=metrics.submit(x,y)
                save_pt(folder/f"step_{step:03d}.pt",{"u_hat":u.detach().cpu(),"symbols":s.detach().cpu().short() if s is not None else None,
                                                    "variable":variable.detach().cpu(),"spec":spec,"domain":record["code_domain"],"step":step})
                image(folder/f"step_{step:03d}.png",y)
                records.append((record,future))
                print("OPT",stage,spec["sample"],initialization,step,"L",values["L_image_float32"],"rate",record.get("actual_frame_share_bits"),flush=True)
            logfile.write(json.dumps(curve,allow_nan=False)+"\n")
            logfile.flush()
            if step<CONFIG["steps"]:
                variable.grad=gradient.clone()
                opt.step()
    rows=[]
    for record,future in records:
        record.update(future.result())
        record["selection_objective"]=record["L_image"]+lam*record.get("R_actual_bpp",0.)
        for ref in ["ZERO","ENCODER_CONTINUOUS","ENCODER_QUANTIZED"]:
            for metric in ["psnr","ms_ssim","lpips"]:
                record[f"{metric}_minus_{ref}"]=record[metric]-refs[ref][metric]
        rows.append(record)
    chosen=min(rows,key=lambda r:(r["selection_objective"],r.get("actual_frame_share_bits",0),r["step"]))
    for r in rows:
        r["selected"]=r["step"]==chosen["step"]
        r["nondominated_psnr_lpips"]=not any(o["psnr"]>=r["psnr"] and o["lpips"]<=r["lpips"] and (o["psnr"]>r["psnr"] or o["lpips"]<r["lpips"]) for o in rows)
    selected=torch.load(folder/f'step_{chosen["step"]:03d}.pt',map_location="cpu",weights_only=False)
    save_pt(folder/"selected.pt",selected)
    write_csv(folder/"checkpoints.csv",rows)
    torch.cuda.synchronize()
    assert tensor_hash(context[0])==ec_hash and tensor_hash(context[1])==q_hash
    assert all(p.grad is None for p in bundle.net.parameters())
    bundle.check()
    save_json(folder/"checks.json",{"network_updates":0,"code_updates":150,"free_scalars":4080,"network_hashes_unchanged":True,
              "ell_c_q_unchanged":True,"optimization_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "hard_gradient_cache_hits":hits,"unique_hard_states":len(cache),"cache_equivalence_check":cache_test,
              "zero_start_no_skip_branch":True,"selected_step":chosen["step"],"teacher_required_for_coding":initialization=="ENCODER_START"})
    del variable,opt
    torch.cuda.empty_cache()
    return chosen,rows
