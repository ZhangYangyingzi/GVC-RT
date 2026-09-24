"""Independent Q1/Q2 calibration and gated full-sequence diagnosis."""
from bridge import *
from optimization import Data,Metrics,reference,optimize


def get_refs(out,spec,bundle,data,metrics):
    folder=out/"references"
    folder.mkdir(exist_ok=True)
    path=folder/f'{spec["sample"]}.json'
    if path.exists():
        return json.loads(path.read_text())
    refs=reference(data,spec,bundle,metrics)
    save_json(path,refs)
    return refs


def calibrate(out,bundle,resume_B=False):
    specs=[s for s in json.loads((out/"samples.json").read_text()) if s["group"]=="network_train6"]
    scales=json.loads((out/"optimization_scale.json").read_text())
    data=Data(out,bundle)
    metrics=Metrics(bundle)
    try:
        chosen_by_init={}
        for initialization in ([] if resume_B else CONFIG["A"]["initializations"]):
            chosen=[]
            all_records=[]
            for s in specs:
                refs=get_refs(out,s,bundle,data,metrics)
                selected,records=optimize(out,"A6_"+initialization,s,initialization,0.,bundle,data,metrics,scales,refs)
                chosen.append(selected)
                all_records.extend(records)
            chosen_by_init[initialization]=chosen
            write_csv(out/f"A6_{initialization}_checkpoints.csv",all_records)
        if not resume_B:
            initialization=min(chosen_by_init,key=lambda k:(np.mean([r["L_image"] for r in chosen_by_init[k]]),k!="ZERO_START"))
            chosen=chosen_by_init[initialization]
            improvement=float(np.mean([r["psnr_minus_ENCODER_CONTINUOUS"] for r in chosen]))
            A={"initialization":initialization,"selected_mean_PSNR_gain_vs_encoder":improvement,
               "selected_mean_LPIPS_change_vs_encoder":float(np.mean([r["lpips_minus_ENCODER_CONTINUOUS"] for r in chosen])),
               "expand":improvement>=CONFIG["A"]["expansion_gate_psnr_db"],"criterion":"six-frame selected mean PSNR improvement >=0.2 dB",
               "initialization_mean_selected_objective":{k:float(np.mean([r["L_image"] for r in v])) for k,v in chosen_by_init.items()}}
            save_json(out/"A_frozen_rule.json",A)
            print("Q1_GATE",A,flush=True)
        # Q2 calibration is unconditional on Q1. Scale lambda against gradients in quantization units.
        estimates=[]
        for s in ([] if resume_B else specs):
            ctx=bundle.context(data.get(s))
            x=source_frame(s["video"],s["frame"])
            z=(data.encoder(s)["encoder_u"].cuda()/DELTA).detach().requires_grad_(True)
            hard=z.detach().round()+(z-z.detach())
            y,_=bundle.render(hard*DELTA,ctx)
            image=F.mse_loss(y,x)+.001*bundle.lp(y,x,normalize=True).mean()
            rate=bundle.net.entropy.bits(hard,DELTA)/PIXELS
            gi=torch.autograd.grad(image,z,retain_graph=True)[0]
            gr=torch.autograd.grad(rate,z)[0]
            ratio=gi.double().norm().item()/max(gr.double().norm().item(),1e-12)
            estimates.append({"sample":s["sample"],"L_image":image.item(),"R_est_bpp":rate.item(),
                              "image_gradient_l2_quant_domain":gi.norm().item(),"rate_gradient_l2_quant_domain":gr.norm().item(),"ratio":ratio})
            del y,image,rate,z,hard,gi,gr
        if resume_B:
            lambdas=json.loads((out/"B_lambda_calibration.json").read_text())["lambdas"]
        else:
            lambda_ref=float(np.median([r["ratio"] for r in estimates]))
            assert np.isfinite(lambda_ref) and lambda_ref>0
            lambdas=[lambda_ref*v for v in CONFIG["B"]["lambda_multipliers"]]
            save_json(out/"B_lambda_calibration.json",{"lambda_reference":lambda_ref,"lambdas":lambdas,"measurements":estimates,
                      "units":"gradient balance in z=u/delta domain against bits/original valid pixel, not an unconverted V1.2 lambda",
                      "initialization":"ZERO_START"})
        B=[]
        for index,lam in enumerate(lambdas):
            chosen=[]
            rows=[]
            for s in specs:
                refs=get_refs(out,s,bundle,data,metrics)
                selected,records=optimize(out,f"B6q_lambda{index}",s,"ZERO_START",lam,bundle,data,metrics,scales,refs,quantized=True)
                chosen.append(selected)
                rows.extend(records)
            write_csv(out/f"B6q_lambda{index}_checkpoints.csv",rows)
            per_video=[]
            for e in entries("Train2"):
                p=[r for r in chosen if r["video"]==e["video_id"]]
                projected=ORC.HEADER.size*8+8+5*sum(r["actual_frame_packet_bits"] for r in p)
                base_bits=(V1/"run_v1/base"/f'{e["video_id"]}_q0.bin').stat().st_size*8
                per_video.append({"video":e["video_id"],"calibration_only_projected_fraction":projected/base_bits})
            B.append({"index":index,"lambda":lam,"mean_L_image":float(np.mean([r["L_image"] for r in chosen])),
                      "mean_psnr_gain_vs_ZERO":float(np.mean([r["psnr_minus_ZERO"] for r in chosen])),
                      "mean_lpips_change_vs_ZERO":float(np.mean([r["lpips_minus_ZERO"] for r in chosen])),
                      "projected_fraction":float(np.mean([v["calibration_only_projected_fraction"] for v in per_video])),
                      "projection_not_a_formal_RD_point":True,"per_video":per_video})
            print("Q2_CAL",B[-1],flush=True)
        middle=min(B[2:5],key=lambda r:(abs(np.log(r["projected_fraction"]/.25)),r["lambda"]))
        selected=[B[1],middle,B[5]]
        rule={"initialization":"ZERO_START","selected_indices":[r["index"] for r in selected],"selected_lambdas":[r["lambda"] for r in selected],
              "all_candidates":B,"checkpoint_selection":CONFIG["B"]["checkpoint_selection"],
              "sequence_budget_selection":CONFIG["B"]["sequence_selection"],"frozen_before_Train2_rest24_or_Val6":True}
        save_json(out/"B_frozen_rule.json",rule)
        save_json(out/"calibration_input_timing.json",data.input_times)
        bundle.check()
    finally:
        metrics.pool.shutdown(wait=True)


def evaluate(out,bundle):
    from sequences import continuous_extension,quantized_cohort,select_budgets
    all_specs=json.loads((out/"samples.json").read_text())
    scales=json.loads((out/"optimization_scale.json").read_text())
    A=json.loads((out/"A_frozen_rule.json").read_text())
    B=json.loads((out/"B_frozen_rule.json").read_text())
    data=Data(out,bundle)
    metrics=Metrics(bundle)
    outcome={"Q1_six_frame_gate":A,"Q2_Val6_expanded":False}
    try:
        if A["expand"]:
            continuous_extension(out,bundle,data,metrics,scales,all_specs,A)
            outcome["Q1_extension_completed"]=True
        else:
            outcome["Q1_extension_completed"]=False
        train_results=quantized_cohort(out,"Train2",bundle,data,metrics,scales,all_specs,B)
        gate=select_budgets(out,"Train2",train_results)
        save_json(out/"B_Train2_budget_gate.json",gate)
        outcome["Q2_Train2_gate"]=gate
        if gate["expand_Val6"]:
            val_results=quantized_cohort(out,"Val6",bundle,data,metrics,scales,all_specs,B)
            selection=select_budgets(out,"Val6",val_results)
            save_json(out/"B_Val6_budget_results.json",selection)
            outcome["Q2_Val6_expanded"]=True
        bundle.check()
        save_json(out/"evaluation_input_timing.json",data.input_times)
        return outcome
    finally:
        metrics.pool.shutdown(wait=True)
