"""Bounded A -> B -> C -> D gates. Existing results are never overwritten."""
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/"config.json").read_text())
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["gpu_uuid"]
sys.path.insert(0,str(HERE.parents[1]))

import lpips
import numpy as np
import torch
from torch.nn import functional as F
from codec import (ROOT, load_models, encode_base, decode_base, jsonable_rows,
                   source_frame, digest_state, strict_load)
from enhancement import Enhancement, BaseOnly, FactorizedEntropy
from evaluation import quality, rgb01, write_video, write_container, temporal
from src.models.video_model_gvcrt import DMC

RUN = HERE/os.environ.get("GVCRT_RUN_NAME","run_v1")


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2))


def fresh_generator():
    model = DMC()
    strict_load(model,ROOT/CFG["checkpoint_p"],"P")
    return model.recon_generation_net.decoder.cuda().eval().requires_grad_(False)


def prepare_base(entries, qps):
    i,p,_ = load_models(CFG)
    for entry in entries:
        for qp in qps:
            stem = RUN/"base"/f'{entry["video_id"]}_q{qp}'
            if stem.with_suffix(".pt").exists():
                continue
            encoded = encode_base(i,p,CFG,entry,qp,stem.with_suffix(".bin"))
            decoded = decode_base(i,p,stem.with_suffix(".bin"))
            torch.save(decoded,stem.with_suffix(".pt"))
            save_json(stem.with_suffix(".json"),{"encode":encoded,"decode":jsonable_rows(decoded)})
            print("BASE",entry["video_id"],qp,"bits",sum(r["base_bits"] for r in decoded),flush=True)
    del i,p
    torch.cuda.empty_cache()


def load_rows(entry,qp):
    return torch.load(RUN/"base"/f'{entry["video_id"]}_q{qp}.pt',map_location="cpu",weights_only=False)


def sources(entry):
    return [source_frame(entry["video_id"],f,"cpu") for f in range(entry["frames"])]


def evaluate(name,entries,qp,generator,perceptual,enh=None,base_only=None,delta=None,checkpoint=None,video=True):
    folder = RUN/name
    folder.mkdir(exist_ok=False)
    all_rows,summary = [],[]
    for entry in entries:
        base_rows = load_rows(entry,qp)
        src = sources(entry)
        outputs,streams,estimates,send_times,hashes = [],[],[],[],[]
        with torch.no_grad():
            for row,x in zip(base_rows,src):
                if row["type"]=="I" or (enh is None and base_only is None and name.startswith("original")):
                    y = rgb01(row["x_base"])
                    streams.append(b"")
                    hashes.append(None)
                    estimates.append(0.)
                    send_times.append(0.)
                else:
                    ell,q = row["ell"].cuda().float(),row["q_recon"].cuda().float()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if base_only is not None:
                        corrected = base_only(ell)
                    elif enh is not None:
                        corrected,symbols = enh(F.pad(x.cuda(),(0,0,0,8),mode="replicate"),ell,delta)
                    else:
                        corrected = ell
                    if delta is not None:
                        stream = ENTROPY.encode(symbols,delta,CFG["training"]["deltas"].index(delta))
                        # Decode using a separately created entropy model immediately, AND later a separate process.
                        symbols_dec,d,_ = ENTROPY.decode(stream)
                        assert torch.equal(symbols.cpu().short(),symbols_dec)
                        streams.append(stream)
                        hashes.append(hashlib.sha256(symbols_dec.numpy().tobytes()).hexdigest())
                        estimates.append(ENTROPY.bits(symbols,delta).item())
                    else:
                        streams.append(b"")
                        hashes.append(None)
                        estimates.append(0.)
                    torch.cuda.synchronize()
                    send_times.append(time.perf_counter()-start)
                    y = rgb01(generator(corrected,q)).cpu()
                outputs.append(y)
        actual_enh = 0
        if delta is not None:
            base_path = RUN/"base"/f'{entry["video_id"]}_q{qp}.bin'
            enh_path = folder/f'{entry["video_id"]}.gle'
            actual_enh = write_container(enh_path,base_path,entry["valid_hw"],streams)
            output_path = folder/f'{entry["video_id"]}_receiver.pt'
            # This receiver never reads originals or the training cache.
            subprocess.run([sys.executable,str(HERE/"receiver.py"),"--base",str(base_path),
                            "--enhancement",str(enh_path),"--checkpoint",str(checkpoint),"--output",str(output_path)],check=True)
            rec = torch.load(output_path,map_location="cpu",weights_only=False)
            assert rec["symbol_hashes"]==hashes
            assert all(torch.equal(a,b) for a,b in zip(outputs,rec["frames"]))
            outputs = rec["frames"]
            save_json(folder/f'{entry["video_id"]}_timing.json',
                      {"sender_analysis_entropy_seconds":send_times,"receiver":{k:v for k,v in rec.items() if k!="frames"},
                       "sender_base_latent_acquisition_seconds":sum(r["decode_seconds"] for r in base_rows),
                       "note":"sender training cache used in evaluation; base-latent acquisition separately measured by real full stream decode, not included in cached analysis timer; a separate timed end-to-end check is required for deployment latency"})
        frame_metrics = []
        for row,x,y in zip(base_rows,src,outputs):
            metrics = quality(x,y,perceptual)
            frame_metrics.append(metrics)
            all_rows.append({"method":name,"video":entry["video_id"],"frame":row["frame"],"type":row["type"],
                             "qp":row["qp"],"base_bits":row["base_bits"],"estimated_enh_bits":estimates[row["frame"]],**metrics})
        bits = sum(r["base_bits"] for r in base_rows)
        duration = entry["duration_seconds"]
        item = {"method":name,"video":entry["video_id"],"qp":qp,
                "base_bits":bits,"enhancement_bits":actual_enh,"total_bits":bits+actual_enh,
                "estimated_enhancement_bits":sum(estimates),"actual_minus_estimated_bits":actual_enh-sum(estimates),
                "base_kbps":bits/duration/1000,"enhancement_kbps":actual_enh/duration/1000,"total_kbps":(bits+actual_enh)/duration/1000,
                "bpp":(bits+actual_enh)/(entry["frames"]*1080*1920),
                **{m:float(np.mean([r[m] for r in frame_metrics])) for m in ["psnr","ms_ssim","lpips"]},
                "temporal":temporal(src,outputs),"transmittable":delta is not None or enh is None}
        summary.append(item)
        print("EVAL",name,entry["video_id"],"PSNR",item["psnr"],"bpp",item["bpp"],flush=True)
        if video:
            fps = entry["metadata"]["avg_frame_rate"]
            write_video(folder/f'{entry["video_id"]}.mkv',outputs,fps)
            original = RUN/"source_videos"/f'{entry["video_id"]}.mkv'
            if not original.exists():
                write_video(original,src,fps)
    with (folder/"metrics.csv").open("x",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    save_json(folder/"summary.json",summary)
    return summary


def mean_psnr(results):
    return float(np.mean([r["psnr"] for r in results]))


def train(stage,entries,qp,enh,base_only,entropy,generator,perceptual,steps,quantized):
    folder = RUN/stage
    folder.mkdir(exist_ok=False)
    # Only extension parameters in optimizers. Base-only is an independent control.
    enh.train()
    base_only.train()
    params = list(enh.parameters()) + (list(entropy.parameters()) if quantized else [])
    opt = torch.optim.Adam(params,lr=CFG["training"]["lr"])
    control_opt = torch.optim.Adam(base_only.parameters(),lr=CFG["training"]["lr"])
    samples = []
    for entry in entries:
        for row in load_rows(entry,qp):
            if row["type"]=="P":
                samples.append((entry["video_id"],row))
    before = digest_state(generator)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder/"training.jsonl").open("x") as logfile:
        for step in range(steps):
            video,row = samples[step % len(samples)]
            x = source_frame(video,row["frame"])
            ell,q = row["ell"].cuda().float(),row["q_recon"].cuda().float()
            delta = CFG["training"]["deltas"][step%3] if quantized else None
            opt.zero_grad(set_to_none=True)
            corrected,symbols = enh(F.pad(x,(0,0,0,8),mode="replicate"),ell,delta)
            image = rgb01(generator(corrected,q))
            mse = F.mse_loss(image,x)
            lp = perceptual(image,x,normalize=True).mean() if quantized else torch.zeros((),device="cuda")
            rate = entropy.bits(symbols,delta)/(1080*1920) if quantized else torch.zeros((),device="cuda")
            loss = mse + CFG["training"]["lambda_p"]*lp + CFG["training"]["lambda_r"]*rate
            loss.backward()
            genc = sum(p.grad.float().square().sum().item() for p in enh.encoder.parameters() if p.grad is not None)**.5
            gdec = sum(p.grad.float().square().sum().item() for p in enh.decoder.parameters() if p.grad is not None)**.5
            if not torch.isfinite(loss) or not np.isfinite(genc+gdec):
                raise RuntimeError("Non-finite loss or gradient")
            assert all(p.grad is None for p in generator.parameters())
            opt.step()
            # Same source, steps, image objective and decoder precision for base-only control.
            control_opt.zero_grad(set_to_none=True)
            y_control = rgb01(generator(base_only(ell),q))
            control_mse = F.mse_loss(y_control,x)
            control_lp = perceptual(y_control,x,normalize=True).mean() if quantized else torch.zeros((),device="cuda")
            (control_mse+CFG["training"]["lambda_p"]*control_lp).backward()
            control_opt.step()
            record = {"step":step+1,"video":video,"frame":row["frame"],"qp":row["qp"],"delta":delta,
                      "mse":mse.item(),"lpips":lp.item(),"rate_bpp":rate.item(),"loss":loss.item(),
                      "encoder_grad_l2":genc,"decoder_grad_l2":gdec,"control_mse":control_mse.item()}
            if quantized and (step%100==0 or step==steps-1):
                stream = entropy.encode(symbols.detach().round(),delta,step%3)
                decoded,_,_ = entropy.decode(stream)
                assert torch.equal(decoded,symbols.detach().round().cpu().short())
                record["estimated_bits"] = entropy.bits(symbols.detach(),delta).item()
                record["actual_bits"] = len(stream)*8
                record["actual_minus_estimated_bits"] = record["actual_bits"]-record["estimated_bits"]
            logfile.write(json.dumps(record)+"\n")
            logfile.flush()
            if step%50==0 or step==steps-1:
                print("TRAIN",stage,record,flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    unchanged = before==digest_state(generator)
    assert unchanged and genc>0 and gdec>0
    checkpoint = folder/"checkpoint.pt"
    torch.save({"enhancement":enh.state_dict(),"base_only":base_only.state_dict(),"entropy":entropy.state_dict(),
                "steps":steps,"stage":stage,"base_qp":qp,"config":CFG,"seed":CFG["seed"],
                "manifest_sha256":hashlib.sha256((HERE/"manifest.json").read_bytes()).hexdigest()},checkpoint)
    save_json(folder/"training_summary.json",{"steps":steps,"seconds":elapsed,"generator_unchanged":unchanged,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"optimizer_parameters":sum(p.numel() for p in params),
              "control_parameters":sum(p.numel() for p in base_only.parameters()),
              "timing_note":"includes source PNG reads, two branch forward/backwards, LPIPS and periodic entropy checks; excludes base cache construction"})
    enh.eval()
    base_only.eval()
    return checkpoint


def main():
    global ENTROPY
    RUN.mkdir(exist_ok=False)
    for name in ["base","source_videos"]:
        (RUN/name).mkdir()
    status = {"status":"RUNNING","stage":"B"}
    save_json(RUN/"fixed_config.json",CFG)
    try:
        assert json.loads((HERE/"stage_a_v2/audit.json").read_text())["status"]=="INTERFACE-PASS"
        torch.set_num_threads(CFG["execution"]["torch_threads"])
        torch.manual_seed(CFG["seed"])
        random.seed(CFG["seed"])
        np.random.seed(CFG["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        # Fail offline rather than allowing torchvision to fetch model weights.
        if not (Path(torch.hub.get_dir())/"checkpoints/alexnet-owt-7be5be79.pth").exists():
            raise RuntimeError("Local LPIPS AlexNet weights absent; downloads disabled")
        perceptual = lpips.LPIPS(net="alex",version="0.1").cuda().eval().requires_grad_(False)
        manifest = json.loads((HERE/"manifest.json").read_text())
        train2 = [e for e in manifest["train"] if e["video_id"] in manifest["train2"]]
        prepare_base(train2,CFG["qp_candidates"])
        totals = {qp:sum(sum(r["base_bits"] for r in load_rows(e,qp)) for e in train2) for qp in CFG["qp_candidates"]}
        qp = min(totals,key=totals.get)
        save_json(RUN/"qp_selection.json",{"measured_train2_bits":totals,"selected_low_rate_qp":qp})
        print("QP_SELECTED",qp,totals,flush=True)
        generator = fresh_generator()
        enh = Enhancement(CFG["architecture"]).cuda()
        base_only = BaseOnly(CFG["architecture"]).cuda()
        ENTROPY = FactorizedEntropy(CFG["architecture"]["enhancement_channels"]).cuda()
        original = evaluate("original_train2",train2,qp,generator,perceptual)
        reference = evaluate("reference_fp32_train2",train2,qp,generator,perceptual)
        train("stage_b",train2,qp,enh,base_only,ENTROPY,generator,perceptual,CFG["training"]["continuous_steps"],False)
        continuous = evaluate("continuous_b_train2",train2,qp,generator,perceptual,enh=enh)
        control = evaluate("base_only_b_train2",train2,qp,generator,perceptual,base_only=base_only)
        gain = mean_psnr(continuous)-mean_psnr(reference)
        status["stage_b_gain_db"] = gain
        if gain < CFG["training"]["continuous_gate_db"]:
            status.update(status="NO-GO",reason="Current Train2 interface/config did not reach fixed 0.2 dB gate after 500 steps; gradients/weights checked")
            return
        status["stage"] = "C"
        save_json(RUN/"status.json",status)
        checkpoint = train("stage_c",train2,qp,enh,base_only,ENTROPY,generator,perceptual,CFG["training"]["quantized_steps"],True)
        continuous_c = evaluate("continuous_c_train2",train2,qp,generator,perceptual,enh=enh)
        evaluate("base_only_c_train2",train2,qp,generator,perceptual,base_only=base_only)
        quant = []
        for j,delta in enumerate(CFG["training"]["deltas"]):
            quant.append(evaluate(f"quantized_c_train2_l{j}",train2,qp,generator,perceptual,enh=enh,delta=delta,checkpoint=checkpoint))
        gains = [mean_psnr(v)-mean_psnr(reference) for v in quant]
        stable = sum(all(a["psnr"]>b["psnr"] for a,b in zip(v,reference)) for v in quant)>=2
        rates = [sum(r["enhancement_bits"] for r in v) for v in quant]
        status.update(stage_c_gains_db=gains,stage_c_actual_bits=rates)
        if not stable or len(set(rates))<3:
            status.update(status="PARTIAL-GO",reason="Quantized Train2 positivity/multirate gate failed; no expansion")
            return
        status["stage"] = "D"
        save_json(RUN/"status.json",status)
        prepare_base(manifest["train"], [qp])
        prepare_base(manifest["val"], CFG["qp_candidates"])
        checkpoint = train("stage_d",manifest["train"],qp,enh,base_only,ENTROPY,generator,perceptual,CFG["training"]["generalization_steps"],True)
        for point in CFG["qp_candidates"]:
            evaluate(f"original_val6_q{point}",manifest["val"],point,generator,perceptual)
        val_reference = evaluate("reference_fp32_val6",manifest["val"],qp,generator,perceptual)
        evaluate("continuous_d_val6",manifest["val"],qp,generator,perceptual,enh=enh)
        val_control = evaluate("base_only_d_val6",manifest["val"],qp,generator,perceptual,base_only=base_only)
        val_quant = []
        for j,delta in enumerate(CFG["training"]["deltas"]):
            val_quant.append(evaluate(f"quantized_d_val6_l{j}",manifest["val"],qp,generator,perceptual,enh=enh,delta=delta,checkpoint=checkpoint))
        status.update(status="PARTIAL-GO",reason="Bounded stages completed; final report evaluates overlap, source information and temporal evidence",stage="D-complete",
                      val_gains_db=[mean_psnr(v)-mean_psnr(val_reference) for v in val_quant],
                      val_vs_control_db=[mean_psnr(v)-mean_psnr(val_control) for v in val_quant])
    except Exception:
        status.update(status="BLOCKED/INCONCLUSIVE",exception=traceback.format_exc())
        raise
    finally:
        save_json(RUN/"status.json",status)
        print("STATUS",status,flush=True)


if __name__ == "__main__":
    main()
