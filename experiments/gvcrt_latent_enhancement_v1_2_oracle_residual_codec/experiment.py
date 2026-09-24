"""Fixed-budget training, paired diagnostics, and actual files with source-free receiver validation."""
import concurrent.futures
import subprocess
import time
from collections import defaultdict
from support import *
from model import ResidualCodec
import bitstream


class Metrics:
    def __init__(self,lp):
        self.lp=lp
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=CFG["metric_workers"])
        self.source={}
        self.cache={}

    def score(self,video,frame,y):
        y=y.detach().cpu().float()
        key=(video,frame,tensor_hash(y))
        if key in self.cache:
            old,f=self.cache[key]
            assert torch.equal(old,y)
            return f
        if (video,frame) not in self.source:
            self.source[(video,frame)]=source_frame(video,frame,"cpu")
        x=self.source[(video,frame)]
        with torch.no_grad():
            lp=self.lp(x.cuda(),y.cuda(),normalize=True).item()
        a,b=x.numpy()[0],y.numpy()[0]
        def work():
            return {"psnr":calc_psnr(a,b,data_range=1),"ms_ssim":float(calc_msssim_rgb(a,b,data_range=1)),"lpips":lp}
        future=self.pool.submit(work)
        self.cache[key]=(y,future)
        return future

    def close(self):
        self.pool.shutdown(wait=True)


def train_targets(out):
    samples=json.loads((out/"fixed_samples.json").read_text())
    return {(s["video"],s["frame"]):torch.load(out/"teachers_train6"/f'{s["sample"]}.pt',map_location="cpu",weights_only=False) for s in samples}


def key_of(t):
    if "sample" in t:
        return t["sample"]["video"],t["sample"]["frame"]
    return t["selection"]["video"],t["selection"]["frame"]


def evaluate(out,name,entries,targets,fixed,metrics,kind,net=None,delta=None,level=0,model_path=None,qp=0,videos=True):
    folder=out/name
    folder.mkdir()
    quantized=kind in ["DIRECT","LEARNED"]
    diagnostic=kind in ["TEACHER","CONTINUOUS","ZERO_INPUT","WRONG_CONTINUOUS","WRONG_QUANTIZED"]
    learned=kind in ["LEARNED","CONTINUOUS","ZERO_INPUT","WRONG_CONTINUOUS","WRONG_QUANTIZED"]
    if net is not None:
        net.eval()
    if kind=="DIRECT":
        ck=torch.load(model_path,map_location="cpu",weights_only=False)
        entropy=FactorizedEntropy(18)
        entropy.load_state_dict(ck["entropy"],strict=True)
    elif net is not None:
        entropy=net.entropy
    else:
        entropy=None
    representations={}
    packed={}
    encode_timing={}
    rows_by_video={e["video_id"]:base_rows(e,qp) for e in entries}
    fixed_rows_by_video={e["video_id"]:base_rows(e,0) for e in entries} if qp else rows_by_video
    donors={e["video_id"]:entries[(i+1)%len(entries)]["video_id"] for i,e in enumerate(entries)}
    # Produce representations in each original context before any WRONG_SOURCE exchange.
    with torch.no_grad():
        if quantized or kind in ["CONTINUOUS","WRONG_CONTINUOUS","WRONG_QUANTIZED"]:
            for entry in entries:
                video=entry["video_id"]
                symbols=[]
                seconds=0.
                for row in rows_by_video[video]:
                    frame=row["frame"]
                    if row["type"]=="I":
                        symbols.append(None)
                        continue
                    torch.cuda.synchronize()
                    t=time.perf_counter()
                    data=targets.get((video,frame))
                    usable=data is not None and data["selection"].get("eligible",True)
                    ell=row["ell"].cuda().float()
                    ec=fixed.ell_c(ell)
                    r=data["r_star"].cuda() if data is not None else torch.zeros_like(ec)
                    if data is not None:
                        assert torch.equal(ec.cpu(),data["ell_c"])
                    u=net.analyze(r,ec) if learned else r
                    if not usable:
                        # Unscheduled P frames remain at ZERO, not a guessed source residual.
                        u=torch.zeros_like(u)
                    if quantized or kind=="WRONG_QUANTIZED":
                        s=bitstream.checked_symbols((u/delta).round())
                        representations[(video,frame)]=s
                        symbols.append(s)
                    else:
                        representations[(video,frame)]=u.cpu()
                    torch.cuda.synchronize()
                    seconds+=time.perf_counter()-t
                if quantized:
                    base_path=V1/"run_v1/base"/f"{video}_q{qp}.bin"
                    stream=folder/f"{video}.orc"
                    t=time.perf_counter()
                    shape=(8,17,30) if learned else (18,68,120)
                    stats=bitstream.encode(stream,base_path,model_path,1 if learned else 0,level,delta,shape,symbols,entropy)
                    entropy_seconds=time.perf_counter()-t
                    readback=bitstream.decode(stream,base_path,model_path,entropy)
                    assert len(readback["frames"])==len(symbols)
                    for a,b in zip(symbols,readback["frames"]):
                        assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                    for row,s in zip(rows_by_video[video],readback["frames"]):
                        if s is not None:
                            representations[(video,row["frame"])]=s
                    packed[video]=stats
                    encode_timing[video]={"residual_analysis_seconds":seconds,"actual_entropy_encode_seconds":entropy_seconds,
                                          "note":"base latent and r_star caches used; optimization time is recorded separately and must be added for uncached encoding"}
                elif kind=="WRONG_QUANTIZED":
                    # Independently serialize/decode each own-context representation before exchange.
                    base_path=V1/"run_v1/base"/f"{video}_q{qp}.bin"
                    stream=folder/f"{video}_source_context.orc"
                    bitstream.encode(stream,base_path,model_path,1,level,delta,(8,17,30),symbols,entropy)
                    own=bitstream.decode(stream,base_path,model_path,entropy)
                    for row,s in zip(rows_by_video[video],own["frames"]):
                        if s is not None:
                            representations[(video,row["frame"])]=s
    all_rows=[]
    summaries=[]
    for entry in entries:
        video=entry["video_id"]
        base=rows_by_video[video]
        outputs=[]
        futures=[]
        zero_exact=0
        source_hashes=[]
        with torch.no_grad():
            for row in base:
                frame=row["frame"]
                target=(video,frame) in targets
                original=rgb01(row["x_base"])
                nz=None
                estimated=0.
                rnrm=0.
                if row["type"]=="I":
                    y=original
                    xc=rgb01(fixed_rows_by_video[video][frame]["x_base"])
                    source_hashes.append(None)
                else:
                    ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
                    h=tensor_hash(row["ell"])
                    ec=fixed.ell_c(ell)
                    xc=rgb01(fixed.g(ec,q))
                    if qp:
                        br=fixed_rows_by_video[video][frame]
                        bec=fixed.ell_c(br["ell"].cuda().float())
                        xc=rgb01(fixed.g(bec,br["q_recon"].cuda().float()))
                    if kind=="ORIGINAL":
                        y=original
                    elif kind=="ORIGINAL_FP32":
                        y=rgb01(fixed.g(ell,q))
                    elif kind=="ZERO_D":
                        y=xc
                    elif kind=="CONTROL_C":
                        y=rgb01(fixed.g(fixed.control(ell),q))
                    else:
                        if kind=="TEACHER":
                            r=targets[(video,frame)]["r_star"].cuda() if target else torch.zeros_like(ec)
                        elif kind=="ZERO_INPUT":
                            u=torch.zeros((1,8,17,30),device="cuda")
                            r=net.synthesize(u,ec)
                            assert torch.count_nonzero(r)==0
                        else:
                            donor=donors[video] if kind.startswith("WRONG") else video
                            rep=representations[(donor,frame)]
                            if quantized or kind=="WRONG_QUANTIZED":
                                u=rep.cuda().float()*delta
                                nz=(rep!=0).float().mean().item()
                                estimated=entropy.bits(rep.to(next(entropy.parameters()).device).float(),delta).item() if torch.count_nonzero(rep)>0 else 0.
                            else:
                                u=rep.cuda()
                            r=net.synthesize(u,ec) if learned else u
                        rnrm=r.norm().item()
                        y=rgb01(fixed.g(ec+r,q))
                        if torch.count_nonzero(r)==0:
                            assert torch.equal(y,xc)
                            zero_exact+=1
                    assert h==tensor_hash(row["ell"])
                    s=representations.get((video,frame))
                    source_hashes.append(tensor_hash(s) if quantized else None)
                outputs.append(y.detach().cpu())
                futures.append(({"method":name,"kind":kind,"video":video,"frame":frame,"type":row["type"],"qp":row["qp"],
                                 "target_available":target,"delta":delta,"diagnostic":diagnostic,"symbol_nonzero_fraction":nz,
                                 "residual_hat_l2":rnrm,"estimated_enhancement_symbol_bits":estimated},
                                metrics.score(video,frame,y),metrics.score(video,frame,xc),metrics.score(video,frame,original)))
        if quantized:
            result_file=folder/f"{video}_receiver.pt"
            subprocess.run([sys.executable,str(HERE/"receiver.py"),"--base",str(V1/"run_v1/base"/f"{video}_q{qp}.bin"),
                            "--stream",str(folder/f"{video}.orc"),"--model",str(model_path),"--output",str(result_file)],check=True)
            received=torch.load(result_file,map_location="cpu",weights_only=False)
            assert received["source_access_guard"] and received["base_state_unchanged"]
            assert received["symbols_hash"]==source_hashes
            assert all(torch.equal(a,b) for a,b in zip(outputs,received["frames"]))
            save_json(folder/f"{video}_coding.json",{"bitstream":packed[video],"sender":encode_timing[video],
                       "receiver":{k:v for k,v in received.items() if k not in ["frames","symbols_hash"]},
                       "symbol_roundtrip_exact":True,"independent_rgb_exact":True})
        selected=[]
        for record,f,b,o in futures:
            qv,bv,ov=f.result(),b.result(),o.result()
            record.update(qv)
            for metric in ["psnr","ms_ssim","lpips"]:
                record[f"{metric}_gain_vs_ZERO_D"]=qv[metric]-bv[metric]
                record[f"{metric}_minus_ORIGINAL"]=qv[metric]-ov[metric]
            selected.append(record)
        all_rows.extend(selected)
        base_bits=(V1/"run_v1/base"/f"{video}_q{qp}.bin").stat().st_size*8
        enhancement_bits=packed[video]["file_bits"] if quantized else (None if diagnostic else 0)
        for scope in ["TARGET_P","P","IP"]:
            subset=[r for r in selected if scope=="IP" or (r["type"]=="P" and (scope=="P" or r["target_available"]))]
            if not subset:
                continue
            item={"method":name,"kind":kind,"video":video,"scope":scope,"frames":len(subset),"delta":delta,
                  "target_P_frames":sum(r["target_available"] for r in selected),"base_bits":base_bits,"enhancement_bits":enhancement_bits,
                  "duration_seconds":entry["duration_seconds"],"rate_scope":"complete I/P sequence, even when quality scope is P or TARGET_P",
                  "base_kbps":base_bits/entry["duration_seconds"]/1000,
                  "enhancement_kbps":enhancement_bits/entry["duration_seconds"]/1000 if enhancement_bits is not None else None,
                  "total_kbps":(base_bits+enhancement_bits)/entry["duration_seconds"]/1000 if enhancement_bits is not None else None,
                  "total_bpp":(base_bits+enhancement_bits)/(entry["frames"]*1080*1920) if enhancement_bits is not None else None,
                  "actual_enhancement_fraction":enhancement_bits/base_bits if enhancement_bits is not None else None,
                  "zero_increment_exact_frames":zero_exact,"diagnostic":diagnostic}
            for key in ["psnr","ms_ssim","lpips"]+[f"{k}_gain_vs_ZERO_D" for k in ["psnr","ms_ssim","lpips"]]+[f"{k}_minus_ORIGINAL" for k in ["psnr","ms_ssim","lpips"]]:
                item[key]=float(np.mean([r[key] for r in subset]))
            summaries.append(item)
        if videos:
            write_video(folder/f"{video}.mkv",outputs,entry["metadata"]["avg_frame_rate"])
        print("EVAL",name,video,"P gain",next(r["psnr_gain_vs_ZERO_D"] for r in summaries if r["video"]==video and r["scope"]=="P"),"bits",enhancement_bits,flush=True)
    for scope in ["TARGET_P","P","IP"]:
        rows=[r for r in summaries if r["scope"]==scope]
        if not rows:
            continue
        total_frames=sum(r["frames"] for r in rows)
        total_duration=sum(r["duration_seconds"] for r in rows)
        b=sum(r["base_bits"] for r in rows)
        e=sum(r["enhancement_bits"] for r in rows) if not diagnostic else None
        summary={"method":name,"kind":kind,"video":"ALL","scope":scope,"frames":total_frames,"delta":delta,
                 "target_P_frames":sum(r["target_P_frames"] for r in rows),"duration_seconds":total_duration,
                 "base_bits":b,"enhancement_bits":e,"base_kbps":b/total_duration/1000,
                 "enhancement_kbps":e/total_duration/1000 if e is not None else None,
                 "total_kbps":(b+e)/total_duration/1000 if e is not None else None,
                 "total_bpp":(b+e)/(sum(x["frames"] for x in entries)*1080*1920) if e is not None else None,
                 "actual_enhancement_fraction":e/b if e is not None else None,"diagnostic":diagnostic,
                 "rate_scope":"complete I/P sequence"}
        for key in ["psnr","ms_ssim","lpips"]+[f"{k}_gain_vs_ZERO_D" for k in ["psnr","ms_ssim","lpips"]]+[f"{k}_minus_ORIGINAL" for k in ["psnr","ms_ssim","lpips"]]:
            summary[key]=sum(r[key]*r["frames"] for r in rows)/total_frames
        summaries.append(summary)
    write_csv(folder/"frames.csv",all_rows)
    write_csv(folder/"summary.csv",summaries)
    save_json(folder/"summary.json",summaries)
    return summaries


def all_scope(rows,scope="TARGET_P"):
    return next(r for r in rows if r["video"]=="ALL" and r["scope"]==scope)


def direct_phase(out,fixed,lp):
    targets=train_targets(out)
    cal=json.loads((out/"calibration.json").read_text())
    entries=manifest()["train2"]
    metrics=Metrics(lp)
    try:
        evaluate(out,"train6_ZERO_D",entries,targets,fixed,metrics,"ZERO_D")
        evaluate(out,"train6_CONTROL_C",entries,targets,fixed,metrics,"CONTROL_C")
        evaluate(out,"train6_TEACHER",entries,targets,fixed,metrics,"TEACHER")
        for qp in range(4):
            evaluate(out,f"train2_ORIGINAL_q{qp}",entries,targets,fixed,metrics,"ORIGINAL",qp=qp,videos=qp==0)
            evaluate(out,f"train2_ORIGINAL_FP32_q{qp}",entries,targets,fixed,metrics,"ORIGINAL_FP32",qp=qp,videos=False)
        for level,delta in enumerate(cal["direct_deltas"]):
            evaluate(out,f"train6_DIRECT_l{level}",entries,targets,fixed,metrics,"DIRECT",delta=delta,level=level,model_path=out/"direct_shared.pt")
    finally:
        metrics.close()


def checkpoint(path,net,stage,step,deltas=None):
    save_pt(path,{"method":"learned","model":net.state_dict(),"residual_scale":net.residual_scale.detach().cpu(),
                  "stage":stage,"step":step,"deltas":deltas,"config":CFG,
                  "baseline_checkpoint_sha256":sha(V1/CFG["baseline_checkpoint"]),"base_qp":0})


def loss_step(net,t,fixed,lp,delta,beta,lam):
    video,frame=key_of(t)
    x=source_frame(video,frame)
    ec,q=t["ell_c"].cuda(),t["q_recon"].cuda()
    target=t["r_star"].cuda()
    r,pred_norm,u,symbols=net(target,ec,delta)
    y=rgb01(fixed.g(ec+r,q))  # frozen parameters, gradient to r retained
    mse=F.mse_loss(y,x)
    perceptual=lp(y,x,normalize=True).mean()
    residual=F.mse_loss(pred_norm,target/net.residual_scale)
    rate=net.entropy.bits(symbols,delta)/(1080*1920) if symbols is not None else torch.zeros((),device="cuda")
    loss=mse+0.001*perceptual+beta*residual+lam*rate
    return loss,{"mse":mse.item(),"lpips":perceptual.item(),"residual_normalized_mse":residual.item(),
                 "beta":beta,"weighted_residual":beta*residual.item(),"lambda_R":lam,"R_enh_bpp":rate.item(),
                 "weighted_rate":lam*rate.item(),"loss":loss.item(),"u_rms":u.detach().square().mean().sqrt().item(),
                 "symbol_nonzero_fraction":(symbols.detach()!=0).float().mean().item() if symbols is not None else None},symbols


def gate(real,wrong,scope="TARGET_P",minimum=.2):
    r,w=all_scope(real,scope),all_scope(wrong,scope)
    details={"psnr_gain_vs_ZERO_D":r["psnr_gain_vs_ZERO_D"],"lpips_gain_vs_ZERO_D":r["lpips_gain_vs_ZERO_D"],
             "REAL_minus_WRONG_psnr":r["psnr"]-w["psnr"],"REAL_minus_WRONG_lpips":r["lpips"]-w["lpips"],"scope":scope}
    details["pass"]=details["psnr_gain_vs_ZERO_D"]>=minimum and details["lpips_gain_vs_ZERO_D"]<=CFG["training"]["gate_LPIPS_max_increase"] and details["REAL_minus_WRONG_psnr"]>=CFG["training"]["gate_REAL_minus_WRONG_PSNR_db"] and details["REAL_minus_WRONG_lpips"]<=CFG["training"]["gate_LPIPS_max_increase"]
    return details


def train_phase(out,stage,net,targets,cal,fixed,lp,metrics,deltas=None):
    cfg=CFG["training"]
    folder=out/stage
    folder.mkdir()
    steps=cfg["A_steps"] if stage=="stage_A" else cfg["B_steps"]
    params=list(net.encoder.parameters())+list(net.decoder.parameters())+(list(net.entropy.parameters()) if deltas else [])
    opt=torch.optim.Adam(params,lr=cfg["lr"])
    dataset=list(targets.values())
    rng=np.random.default_rng(CFG["seed"]+(1 if deltas else 0))
    order=[]
    history=[]
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder/"training.jsonl").open("x") as log:
        for step in range(steps):
            if step%len(dataset)==0:
                order=rng.permutation(len(dataset)).tolist()
            t=dataset[order[step%len(dataset)]]
            level=(step+step//len(dataset))%len(deltas) if deltas else None
            delta=deltas[level] if deltas else None
            lam=next(x["lambda_R"] for x in cfg["B_rate_schedule"] if x["start"]<=step<x["end"]) if deltas else 0.
            net.train()
            opt.zero_grad(set_to_none=True)
            loss,record,symbols=loss_step(net,t,fixed,lp,delta,cfg["beta_residual"],lam)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no configuration change")
            loss.backward()
            for name,mod in [("encoder",net.encoder),("decoder",net.decoder)]:
                grads=[p.grad for p in mod.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                record[name+"_grad_l2"]=sum(g.double().square().sum().item() for g in grads)**.5
            assert all(p.grad is None for p in fixed.g.parameters())
            opt.step()
            record.update(step=step+1,video=key_of(t)[0],frame=key_of(t)[1],delta=delta)
            if deltas and (step%100==0 or step==steps-1):
                s=bitstream.checked_symbols(symbols.detach().round())
                if torch.count_nonzero(s)==0:
                    record["actual_frame_packet_bits"]=8
                    record["estimated_symbol_bits"]=0.
                else:
                    cdfs,_=net.entropy.coder(delta)
                    payload=arithmetic_encode((s.flatten()+128).tolist(),cdfs,510)
                    decoded=torch.tensor(arithmetic_decode(payload,cdfs,510,s.numel()),dtype=torch.int16).reshape_as(s)-128
                    assert torch.equal(s,decoded)
                    record["actual_frame_packet_bits"]=8*(9+len(payload))
                    record["estimated_symbol_bits"]=net.entropy.bits(s.cuda().float(),delta).item()
                record["rate_note"]="actual frame packet only; full file header charged in formal evaluation"
            history.append(record)
            log.write(json.dumps(record,allow_nan=False)+"\n")
            log.flush()
            if step%50==0 or step==steps-1:
                print("TRAIN",stage,record,flush=True)
            if (step+1)%cfg["checkpoint_interval"]==0 or step==steps-1:
                checkpoint(folder/f"checkpoint_{step+1:04d}.pt",net,stage,step+1,deltas)
            if deltas and step+1==cfg["B_adaptation_check_step"]:
                cp=folder/f"checkpoint_{step+1:04d}.pt"
                checkpoint(cp,net,stage,step+1,deltas)
                checks=[]
                for j,d in enumerate(deltas):
                    real=evaluate(out,f"train6_Badapt_l{j}",manifest()["train2"],targets,fixed,metrics,"LEARNED",net,d,j,cp,videos=False)
                    wrong=evaluate(out,f"train6_Badapt_WRONG_l{j}",manifest()["train2"],targets,fixed,metrics,"WRONG_QUANTIZED",net,d,j,cp,videos=False)
                    checks.append(gate(real,wrong,minimum=cfg["quantized_gate_min_PSNR_gain_db"]))
                save_json(folder/"adaptation_gate.json",checks)
                if sum(c["pass"] for c in checks)<cfg["quantized_gate_min_passing_levels"]:
                    fixed.assert_frozen()
                    return {"completed_steps":step+1,"stopped":"quantized adaptation gate failed","checkpoint":str(cp)}
    fixed.assert_frozen()
    assert history[-1]["encoder_grad_l2"]>0 and history[-1]["decoder_grad_l2"]>0
    with torch.no_grad():
        ec=dataset[0]["ell_c"].cuda()
        zero=net.synthesize(torch.zeros((1,8,17,30),device="cuda"),ec)
        assert torch.count_nonzero(zero)==0
        assert torch.equal(fixed.g(ec+zero,dataset[0]["q_recon"].cuda()),fixed.g(ec,dataset[0]["q_recon"].cuda()))
    result={"completed_steps":steps,"seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "frozen_models_unchanged":True,"zero_increment_exact":True,"checkpoint":str(folder/f"checkpoint_{steps:04d}.pt"),
            "optimized_parameters":sum(p.numel() for p in params),"gradient_recheck":history[-1]}
    save_json(folder/"training_summary.json",result)
    return result


def train_phases(out,fixed,lp):
    cfg=CFG["training"]
    targets=train_targets(out)
    cal=json.loads((out/"calibration.json").read_text())
    net=ResidualCodec(torch.tensor(cal["residual_channel_rms"]).reshape(1,18,1,1)).cuda()
    metrics=Metrics(lp)
    try:
        first=next(iter(targets.values()))
        with torch.no_grad():
            u=net.analyze(first["r_star"].cuda(),first["ell_c"].cuda())
            zero=net.synthesize(torch.zeros_like(u),first["ell_c"].cuda())
            assert torch.count_nonzero(zero)==0
        save_json(out/"model_interface.json",{"r_star_shape":list(first["r_star"].shape),"u_shape":list(u.shape),"spatial_downsample":4,
                  "symbols_per_P_frame":u.numel(),"r_star_values":first["r_star"].numel(),"value_ratio":first["r_star"].numel()/u.numel(),
                  "zero_normalized_increment_exact":True,"encoder_parameters":sum(p.numel() for p in net.encoder.parameters()),
                  "decoder_parameters":sum(p.numel() for p in net.decoder.parameters()),"entropy_parameters":sum(p.numel() for p in net.entropy.parameters())})
        a=train_phase(out,"stage_A",net,targets,cal,fixed,lp,metrics)
        real=evaluate(out,"train6_A_CONTINUOUS",manifest()["train2"],targets,fixed,metrics,"CONTINUOUS",net)
        zero=evaluate(out,"train6_A_ZERO_INPUT",manifest()["train2"],targets,fixed,metrics,"ZERO_INPUT",net)
        wrong=evaluate(out,"train6_A_WRONG",manifest()["train2"],targets,fixed,metrics,"WRONG_CONTINUOUS",net)
        check=gate(real,wrong,minimum=cfg["A_gate_selected_PSNR_gain_db"])
        save_json(out/"stage_A_gate.json",check)
        if not check["pass"]:
            save_json(out/"stage_A_recheck.json",{"gradients":a["gradient_recheck"],"normalization":"fixed channel RMS from six exact rebased targets",
                      "zero_increment_exact":True,"frozen_weights_unchanged":True,"r_star_shape":[1,18,68,120],"u_shape":[1,8,17,30],
                      "action":"stop; no architecture/step/loss search","stage_A_gate":check})
            return {"outcome":"STOP_A","source_information":check,"actual_rate_efficiency":"direct reference measured; learned quantized codec not reached","generalization":"not evaluated"}
        with torch.no_grad():
            us=torch.cat([net.analyze(t["r_star"].cuda(),t["ell_c"].cuda()).cpu() for t in targets.values()])
        rms=us.double().square().mean().sqrt().item()
        deltas=[float(np.float32(rms*m)) for m in cfg["quant_delta_multipliers"]]
        scales=us.square().mean((0,2,3),keepdim=True).sqrt()*np.sqrt(3)/np.pi
        with torch.no_grad():
            net.entropy.log_scale.copy_(torch.log(torch.expm1((scales-1e-4).clamp_min(1e-6))).cuda())
        save_json(out/"learned_quantization_calibration.json",{"global_u_rms_Train6":rms,"deltas":deltas,"initial_channel_logistic_scale":scales.flatten().tolist(),"validation_used":False})
        b=train_phase(out,"stage_B",net,targets,cal,fixed,lp,metrics,deltas)
        if "stopped" in b:
            return {"outcome":"STOP_B_ADAPTATION","details":b,"source_information":"continuous gate passed, quantized adaptation failed","actual_rate_efficiency":"not established","generalization":"not evaluated"}
        cp=Path(b["checkpoint"])
        evaluate(out,"train6_B_CONTINUOUS",manifest()["train2"],targets,fixed,metrics,"CONTINUOUS",net)
        selected_gates=[]
        for level,d in enumerate(deltas):
            real=evaluate(out,f"train6_CODED_l{level}",manifest()["train2"],targets,fixed,metrics,"LEARNED",net,d,level,cp)
            wrong=evaluate(out,f"train6_CODED_WRONG_l{level}",manifest()["train2"],targets,fixed,metrics,"WRONG_QUANTIZED",net,d,level,cp,videos=False)
            selected_gates.append(gate(real,wrong,minimum=cfg["quantized_gate_min_PSNR_gain_db"]))
        save_json(out/"train6_coded_gate.json",selected_gates)
        if not any(c["pass"] for c in selected_gates):
            return {"outcome":"STOP_TRAIN6_CODED","source_information":selected_gates,"actual_rate_efficiency":"no positive coded point","generalization":"not evaluated"}
        import teachers
        targets,train_timing=teachers.generate_evaluation(out,manifest()["train2"],cal,fixed,lp,targets)
        write_csv(out/"train2_sender_optimization_timing.csv",train_timing)
        formal=evaluate_cohort(out,"train2_full",manifest()["train2"],targets,cal,fixed,metrics,net,deltas,cp)
        save_json(out/"train2_full_gate.json",formal)
        if not any(c["pass"] for c in formal):
            return {"outcome":"STOP_TRAIN2_FULL","source_information":formal,"actual_rate_efficiency":"not established on full Train2","generalization":"not evaluated"}
        # Network and calibration are now locked; only per-frame sender tensors may change.
        checkpoint_hash=sha(cp)
        targets,val_timing=teachers.generate_evaluation(out,manifest()["val6"],cal,fixed,lp,targets)
        write_csv(out/"val6_sender_optimization_timing.csv",val_timing)
        val=evaluate_cohort(out,"val6",manifest()["val6"],targets,cal,fixed,metrics,net,deltas,cp)
        assert sha(cp)==checkpoint_hash
        return {"outcome":"COMPLETE_VAL6","source_information_per_level":val,"actual_rate_efficiency":"see actual budget and per-video overlap tables","generalization":"fixed Val6 evaluated; no shared updates"}
    finally:
        metrics.close()


def evaluate_cohort(out,prefix,entries,targets,cal,fixed,metrics,net,deltas,cp):
    evaluate(out,prefix+"_ZERO_D",entries,targets,fixed,metrics,"ZERO_D")
    evaluate(out,prefix+"_CONTROL_C",entries,targets,fixed,metrics,"CONTROL_C")
    evaluate(out,prefix+"_TEACHER",entries,targets,fixed,metrics,"TEACHER")
    for qp in range(4):
        evaluate(out,prefix+f"_ORIGINAL_q{qp}",entries,targets,fixed,metrics,"ORIGINAL",qp=qp,videos=qp==0)
        evaluate(out,prefix+f"_ORIGINAL_FP32_q{qp}",entries,targets,fixed,metrics,"ORIGINAL_FP32",qp=qp,videos=False)
    for level,d in enumerate(cal["direct_deltas"]):
        evaluate(out,prefix+f"_DIRECT_l{level}",entries,targets,fixed,metrics,"DIRECT",delta=d,level=level,model_path=out/"direct_shared.pt",videos=level in [0,2,4])
    evaluate(out,prefix+"_CONTINUOUS",entries,targets,fixed,metrics,"CONTINUOUS",net)
    evaluate(out,prefix+"_ZERO_INPUT",entries,targets,fixed,metrics,"ZERO_INPUT",net)
    checks=[]
    for level,d in enumerate(deltas):
        real=evaluate(out,prefix+f"_CODED_l{level}",entries,targets,fixed,metrics,"LEARNED",net,d,level,cp)
        wrong=evaluate(out,prefix+f"_WRONG_l{level}",entries,targets,fixed,metrics,"WRONG_QUANTIZED",net,d,level,cp)
        checks.append(gate(real,wrong,scope="P",minimum=CFG["training"]["quantized_gate_min_PSNR_gain_db"]))
    return checks
