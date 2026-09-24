"""Whole-sequence coding and budget selection, never a mixture of lambda candidates."""
import subprocess
from bridge import *
from optimization import optimize


def chosen_row(folder):
    rows=read_csv(folder/"checkpoints.csv")
    return next(r for r in rows if r["selected"])


def specs_for(all_specs,split):
    return [s for s in all_specs if s["split"]==split]


def render_sequences(out,name,split,specs,codes,bundle,data,metrics,quantized=False,lam=None,
                     original_codes=None,wrong=False,zero=False,no_stream=False,steps=None):
    from stages import get_refs
    folder=out/"sequences"/name
    folder.mkdir(parents=True)
    ee=entries(split)
    donor={e["video_id"]:ee[(i+1)%len(ee)]["video_id"] for i,e in enumerate(ee)}
    by_key={(s["video"],s["frame"]):s for s in specs}
    frame_stats={}
    entropy_times={}
    # Encode and legally decode each representation in its own context before interventions.
    if quantized and not wrong and not zero and not no_stream:
        for e in ee:
            video=e["video_id"]
            base=V1/"run_v1/base"/f"{video}_q0.bin"
            stream=folder/f"{video}.orc"
            frames=[None]+[codes[(video,f)]["symbols"] for f in range(1,16)]
            t=time.perf_counter()
            st=ORC.encode(stream,base,MODEL,1,3,DELTA,(8,17,30),frames,bundle.net.entropy)
            entropy_times[video]=time.perf_counter()-t
            dec=ORC.decode(stream,base,MODEL,bundle.net.entropy)
            for a,b in zip(frames,dec["frames"]):
                assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a.short(),b))
            for f in range(1,16):
                codes[(video,f)]={"symbols":dec["frames"][f],"u_hat":dec["frames"][f].float()*DELTA}
            frame_stats[video]=st
    allframes=[]
    sequence=[]
    for e in ee:
        video=e["video_id"]
        data.get(by_key[(video,1)])
        base_rows_=data.base[video]
        outputs=[]
        pending=[]
        symbol_hashes=[]
        with torch.no_grad():
            for f in range(16):
                row=base_rows_[f]
                if f==0:
                    y=rgb01(row["x_base"]).cuda()
                    x=source_frame(video,0)
                    metric=metrics.submit(x,y)
                    refs=None
                    spec={"group":"I","sample":video+"_f0000"}
                    s=None
                    rn=0.
                    u=torch.zeros(CONFIG["code_shape"],device="cuda")
                    symbol_hashes.append(None)
                else:
                    spec=by_key[(video,f)]
                    context=bundle.context(data.get(spec))
                    source_video=donor[video] if wrong else video
                    code=codes[(source_video,f)]
                    if zero or no_stream:
                        s=torch.zeros(CONFIG["code_shape"],dtype=torch.int16)
                        u=s.cuda().float()
                    else:
                        s=code.get("symbols")
                        u=code["u_hat"].cuda()
                    y,r=bundle.render(u,context)
                    rn=r.norm().item()
                    if torch.count_nonzero(u)==0:
                        assert torch.count_nonzero(r)==0 and torch.equal(y,rgb01(bundle.fixed.g(context[0],context[1])))
                    x=source_frame(video,f)
                    metric=metrics.submit(x,y)
                    refs=get_refs(out,spec,bundle,data,metrics)
                    symbol_hashes.append(tensor_hash(s) if s is not None else None)
                outputs.append(y.cpu())
                packet=frame_stats[video]["frames"][f]["actual_bits"] if video in frame_stats else None
                record={"method":name,"split":split,"video":video,"frame":f,"group":spec["group"],"type":"I" if f==0 else "P",
                        "lambda":lam,"diagnostic":(not quantized and not no_stream) or wrong or zero,"zero_no_stream_fallback":no_stream,
                        "symbol_nonzero_fraction":(s!=0).float().mean().item() if s is not None else None,
                        "nonzero_enhancement_frame":bool(f>0 and torch.count_nonzero(u)>0),
                        "u_l2":u.double().norm().item(),"residual_l2":rn,"actual_packet_bits":packet,
                        "selected_step":steps.get((video,f)) if steps else None}
                pending.append((record,metric,refs))
        if video in frame_stats or no_stream:
            args=[sys.executable,str(HOME/"receiver.py"),"--base",str(V1/"run_v1/base"/f"{video}_q0.bin"),"--output",str(folder/f"{video}_receiver.pt")]
            if not no_stream:
                args += ["--stream",str(folder/f"{video}.orc")]
            subprocess.run(args,check=True)
            received=torch.load(folder/f"{video}_receiver.pt",map_location="cpu",weights_only=False)
            assert received["source_access_guard"] and received["base_state_unchanged"]
            assert all(torch.equal(a,b) for a,b in zip(outputs,received["frames"]))
            if not no_stream:
                assert received["symbols_hash"]==symbol_hashes
            save_json(folder/f"{video}_checks.json",{"roundtrip_exact":True,"independent_RGB_exact":True,
                      "entropy_encode_seconds":entropy_times.get(video,0.),
                      "receiver":{k:v for k,v in received.items() if k not in ["frames","symbols_hash"]},
                      "bitstream":frame_stats.get(video),"no_enhancement_file_sent":no_stream})
        frame_rows=[]
        for record,future,refs in pending:
            record.update(future.result())
            if refs is None:
                refs={k:{m:record[m] for m in ["psnr","ms_ssim","lpips","L_image"]} for k in ["ZERO","ENCODER_CONTINUOUS","ENCODER_QUANTIZED","TEACHER","ORIGINAL_QP0"]}
            for method in refs:
                for m in ["psnr","ms_ssim","lpips"]:
                    record[f"{m}_minus_{method}"]=record[m]-refs[method][m]
            frame_rows.append(record)
        allframes.extend(frame_rows)
        base_bits=(V1/"run_v1/base"/f"{video}_q0.bin").stat().st_size*8
        enh_bits=frame_stats[video]["file_bits"] if video in frame_stats else (0 if no_stream else None)
        for scope in ["P","IP"]:
            subset=[r for r in frame_rows if scope=="IP" or r["type"]=="P"]
            item={"method":name,"split":split,"video":video,"scope":scope,"frames":len(subset),"lambda":lam,
                  "base_bits":base_bits,"enhancement_bits":enh_bits,"base_kbps":base_bits/e["duration_seconds"]/1000,
                  "enhancement_kbps":enh_bits/e["duration_seconds"]/1000 if enh_bits is not None else None,
                  "total_kbps":(base_bits+enh_bits)/e["duration_seconds"]/1000 if enh_bits is not None else None,
                  "total_bpp":(base_bits+enh_bits)/(16*PIXELS) if enh_bits is not None else None,
                  "enhancement_fraction":enh_bits/base_bits if enh_bits is not None else None,
                  "duration_seconds":e["duration_seconds"],"diagnostic":enh_bits is None,"zero_no_stream_fallback":no_stream,
                  "nonzero_enhancement_frame_fraction_P":np.mean([r["nonzero_enhancement_frame"] for r in frame_rows[1:]]).item()}
            fields=["psnr","ms_ssim","lpips","L_image"]+[f"{m}_minus_{r}" for r in ["ZERO","ENCODER_CONTINUOUS","ENCODER_QUANTIZED","ORIGINAL_QP0"] for m in ["psnr","ms_ssim","lpips"]]
            item.update({k:float(np.mean([r[k] for r in subset])) for k in fields})
            sequence.append(item)
        write_video(folder/f"{video}.mkv",outputs,e["metadata"]["avg_frame_rate"])
        print("SEQ",name,video,"bits",enh_bits,flush=True)
    group_rows=[]
    for group in sorted({r["group"] for r in allframes if r["type"]=="P"}):
        subset=[r for r in allframes if r["group"]==group]
        group_rows.append({"method":name,"group":group,"frames":len(subset),"lambda":lam,
                           "actual_packet_bits":sum(r["actual_packet_bits"] for r in subset) if subset[0]["actual_packet_bits"] is not None else None,
                           "rate_note":"group packet cost only, shared headers charged in complete sequence; not a group RD point",
                           **{k:float(np.mean([r[k] for r in subset if r[k] is not None])) if any(r[k] is not None for r in subset) else None for k in ["psnr","ms_ssim","lpips","symbol_nonzero_fraction","nonzero_enhancement_frame"]+[f"{m}_minus_{ref}" for ref in ["ZERO","ENCODER_CONTINUOUS","ENCODER_QUANTIZED"] for m in ["psnr","ms_ssim","lpips"]]}})
    write_csv(folder/"frames.csv",allframes)
    write_csv(folder/"summary.csv",sequence)
    write_csv(folder/"groups.csv",group_rows)
    save_json(folder/"summary.json",sequence)
    return sequence


def continuous_extension(out,bundle,data,metrics,scales,all_specs,A):
    from stages import get_refs
    init=A["initialization"]
    for split in ["Train2","Val6"]:
        specs=specs_for(all_specs,split)
        codes={}
        final_codes={}
        steps={}
        for s in specs:
            stage="A6_"+init if s["group"]=="network_train6" else "Aext_"+init
            folder=out/"optimizations"/stage/s["sample"]
            if s["group"]!="network_train6":
                refs=get_refs(out,s,bundle,data,metrics)
                optimize(out,stage,s,init,0.,bundle,data,metrics,scales,refs)
            row=chosen_row(folder)
            codes[(s["video"],s["frame"])]=torch.load(folder/"selected.pt",map_location="cpu",weights_only=False)
            final_codes[(s["video"],s["frame"])]=torch.load(folder/"step_150.pt",map_location="cpu",weights_only=False)
            steps[(s["video"],s["frame"])]=row["step"]
        render_sequences(out,f"A_{split}_selected",split,specs,codes,bundle,data,metrics,steps=steps)
        render_sequences(out,f"A_{split}_step150",split,specs,final_codes,bundle,data,metrics)


def quantized_cohort(out,split,bundle,data,metrics,scales,all_specs,B):
    from stages import get_refs
    specs=specs_for(all_specs,split)
    results=[]
    for index,lam in zip(B["selected_indices"],B["selected_lambdas"]):
        codes={}
        steps={}
        for s in specs:
            stage=f"B6q_lambda{index}" if s["group"]=="network_train6" else f"Bfull_lambda{index}"
            folder=out/"optimizations"/stage/s["sample"]
            if s["group"]!="network_train6":
                refs=get_refs(out,s,bundle,data,metrics)
                optimize(out,stage,s,"ZERO_START",lam,bundle,data,metrics,scales,refs,quantized=True)
            selected=chosen_row(folder)
            codes[(s["video"],s["frame"])]=torch.load(folder/"selected.pt",map_location="cpu",weights_only=False)
            steps[(s["video"],s["frame"])]=selected["step"]
        name=f"B_{split}_lambda{index}"
        rows=render_sequences(out,name,split,specs,codes,bundle,data,metrics,True,lam,steps=steps)
        # codes now contain legally decoded original-context representations.
        render_sequences(out,name+"_WRONG",split,specs,codes,bundle,data,metrics,False,lam,wrong=True)
        render_sequences(out,name+"_ZERO",split,specs,codes,bundle,data,metrics,False,lam,zero=True)
        results.extend(rows)
    zero_codes={(s["video"],s["frame"]):{"symbols":torch.zeros(CONFIG["code_shape"],dtype=torch.int16),"u_hat":torch.zeros(CONFIG["code_shape"])} for s in specs}
    baseline=render_sequences(out,f"B_{split}_ZERO_NO_STREAM",split,specs,zero_codes,bundle,data,metrics,no_stream=True)
    results.extend(baseline)
    return results


def select_budgets(out,split,results):
    selected=[]
    encoder=[]
    original_prefix="train2_full" if split=="Train2" else "val6"
    for e in entries(split):
        video=e["video_id"]
        baseline=next(r for r in results if r["video"]==video and r["scope"]=="IP" and r["zero_no_stream_fallback"])
        old_candidates=[dict(baseline,method="EXISTING_ZERO_NO_STREAM")]
        for level in range(4):
            path=OLD/"results_v1"/f"{original_prefix}_CODED_l{level}"
            r=next(r for r in json.loads((path/"summary.json").read_text()) if r["video"]==video and r["scope"]=="IP")
            frames=[r for r in read_csv(path/"frames.csv") if r["video"]==video]
            assert all(f["psnr"]<99 for f in frames)
            li=float(np.mean([10**(-f["psnr"]/10)+.001*f["lpips"] for f in frames]))
            old_candidates.append({"method":f"EXISTING_ENCODER_l{level}","video":video,"base_bits":r["base_bits"],"enhancement_bits":r["enhancement_bits"],
                                   "L_image":li,"psnr":r["psnr"],"ms_ssim":r["ms_ssim"],"lpips":r["lpips"],
                                   "psnr_minus_ZERO":r["psnr_gain_vs_ZERO_D"],"lpips_minus_ZERO":r["lpips_gain_vs_ZERO_D"],"zero_no_stream_fallback":False})
        candidates=[r for r in results if r["video"]==video and r["scope"]=="IP"]
        for alpha in CONFIG["B"]["budgets"]:
            feasible=[r for r in candidates if r["enhancement_bits"]<=alpha*r["base_bits"]]
            chosen=min(feasible,key=lambda r:(r["L_image"],r["enhancement_bits"],r["method"]))
            selected.append({"budget":alpha,**chosen,"budget_satisfied":True,"selection":"whole candidate, minimum L_image; no cross-lambda frame mixing"})
            feasible_old=[r for r in old_candidates if r["enhancement_bits"]<=alpha*r["base_bits"]]
            old=min(feasible_old,key=lambda r:(r["L_image"],r["enhancement_bits"],r["method"]))
            encoder.append({"budget":alpha,**old,"budget_satisfied":True})
    write_csv(out/f"B_{split}_budget_selections.csv",selected)
    write_csv(out/f"B_{split}_existing_encoder_budget_selections.csv",encoder)
    gates=[]
    for alpha in CONFIG["B"]["budgets"]:
        rows=[r for r in selected if r["budget"]==alpha]
        gain=float(np.mean([r["psnr_minus_ZERO"] for r in rows]))
        lp=float(np.mean([r["lpips_minus_ZERO"] for r in rows]))
        gates.append({"budget":alpha,"all_sequences_feasible":all(r["enhancement_bits"]<=alpha*r["base_bits"] for r in rows),
                      "mean_IP_PSNR_gain":gain,"mean_IP_LPIPS_change":lp,"fallback_sequences":sum(r["zero_no_stream_fallback"] for r in rows),
                      "pass":gain>=.1 and lp<=1e-4})
    return {"split":split,"budgets":gates,"expand_Val6":any(r["pass"] for r in gates),
            "network_updates":0,"whole_sequence_selection_only":True}
