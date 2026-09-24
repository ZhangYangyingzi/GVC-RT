#!/usr/bin/env python3
import argparse
import faulthandler
import csv
import gc
import hashlib
import io
import json
import math
import random
import struct
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V13 = ROOT.parent / "gvcrt_v13_generator_aware_mixed_precision"
V124 = ROOT.parent / "gvcrt_v12_4_sequence_discrete_beam"
for path in (REPO, V13, ROOT): sys.path.insert(0, str(path))

import v13_codec as codec
import run_v13 as base
from src.utils.stream_helper import write_sps

CFG = json.loads((ROOT / "config.json").read_text())
MAN = json.loads((ROOT / "manifest.json").read_text())
SPS = base.SPS


class Tee:
    def __init__(self, stream, path): self.stream, self.file = stream, open(path, "a", buffering=1)
    def write(self, value): self.stream.write(value); self.file.write(value); return len(value)
    def flush(self): self.stream.flush(); self.file.flush()


def write_csv(path, rows, fields=None):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open("w", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f: return list(csv.DictReader(f))


def tag(video, qp): return f"{video['dataset']}_{int(video['video_id']):02d}_qp{qp}"


def gran_tag(video, qp, gran): return f"{tag(video, qp)}_{gran}"


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average(rows): return base.average_metrics(rows)


def evaluate(enc, dec, prepared, modes, qp, gt, dec_state, tile, original=False, global_mode=None):
    base.restore_device(dec, dec_state, prepared["w1"].device)
    cand = prepared if modes == prepared["modes"] else codec.requantize_frame(enc, prepared, modes, tile, tile)
    if original: payload, comp = codec.encode_original(enc, cand, qp)
    else: payload, comp = codec.encode_mixed(enc, cand, qp, tile, tile, global_mode)
    out, cap = codec.decode_mixed(dec, payload, SPS, qp, tile, tile)
    if not torch.equal(cap["z"], cand["z"]) or not torch.equal(cap["w0"], cand["w0"]) or not torch.equal(cap["w1"], cand["w1"]):
        raise RuntimeError("entropy symbol mismatch")
    if cap["mixed"] and cap["modes"] != list(modes): raise RuntimeError("precision map mismatch")
    if not torch.equal(cap["latent"], cand["latent"]): raise RuntimeError("latent mismatch")
    if not torch.equal(cap["feature"], cand["feature"]): raise RuntimeError("feature/state mismatch")
    if not torch.equal(out, cand["rgb"]): raise RuntimeError("RGB mismatch")
    mse, psnr, _ = base.basic_metrics(out, gt)
    est = codec.gaussian_estimated_bits(enc, cand["w1"], cand["s1"] / cand["delta"], cand["s1"])
    return cand, payload, comp, out, {"MSE": mse, "PSNR": psnr}, base.snap_cpu(dec), cap, est


def baseline(video, frames, qp, device, metrics):
    t = tag(video, qp); path = ROOT / "bitstreams" / f"{t}_baseline.bin"
    iframe, enc, dec = base.load_triplet(device); before = base.model_hash(iframe, enc, dec)
    stream, rgb0, ip = base.init_sequence(iframe, enc, dec, frames[0], qp)
    rows = [dict(frame=0, actual_qp=qp, actual_bytes=len(base.packet(ip, qp, True)),
                 **base.frame_metric(rgb0, frames[0], metrics), DPB_state_hash=base.state_hash(base.snap_cpu(dec)))]
    totals = {"I_bytes": len(ip), "z_bytes": 0, "w0_bytes": 0, "w1_bytes": 0, "mode_map_bits": 0, "mode_map_bytes": 0}
    symbols_ok = rgb_ok = state_ok = True
    ref_path = V124 / "bitstreams" / f"{t}_BASELINE.bin"; ref_payloads, _ = base.parse_stream(ref_path)
    for fi in range(1, 64):
        aq = enc.shift_qp(qp, base.INDEX_MAP[fi % 8]); prep = codec.prepare_frame(enc, frames[fi], aq, None, 4, 4)
        payload, comp = codec.encode_original(enc, prep, aq)
        if payload != ref_payloads[fi][2]: raise RuntimeError(f"baseline payload mismatch frame {fi}")
        out, cap = codec.decode_mixed(dec, payload, SPS, aq, 4, 4)
        symbols_ok &= torch.equal(cap["w1"], prep["w1"]); rgb_ok &= torch.equal(out, prep["rgb"])
        enc.add_ref_frame(prep["feature"], None); state_ok &= torch.equal(enc.dpb[0].feature, dec.dpb[0].feature)
        pkt = base.packet(payload, aq); stream += pkt
        for key in ("z_bytes", "w0_bytes", "w1_bytes"): totals[key] += comp[key]
        rows.append(dict(frame=fi, actual_qp=aq, actual_bytes=len(pkt), **base.frame_metric(out, frames[fi], metrics),
                         w1_hash=codec.tensor_hash(cap["w1"]), latent_hash=codec.tensor_hash(cap["latent"]),
                         RGB_hash=codec.tensor_hash(out), DPB_state_hash=base.state_hash(base.snap_cpu(dec))))
    path.write_bytes(stream); after = base.model_hash(iframe, enc, dec)
    identity = stream == ref_path.read_bytes() and symbols_ok and rgb_ok and state_ok and before == after
    if not identity: raise RuntimeError("baseline/all-1x identity gate failed")
    totals["header_bytes"] = len(stream) - sum(totals[k] for k in ("I_bytes", "z_bytes", "w0_bytes", "w1_bytes", "mode_map_bytes"))
    avg = average(rows)
    row = {"dataset": video["dataset"], "video": video["name"], "video_id": video["video_id"], "qp": qp,
           "frames": 64, "baseline_bytes": len(stream), "baseline_bits": len(stream)*8,
           "bpp": len(stream)*8/(64*1920*1080), "kbps": len(stream)*8*video["fps"]/64/1000,
           **totals, **avg, "bitstream_path": str(path), "bitstream_sha256": sha256(path),
           "v12_4_sha256": sha256(ref_path), "identity_gate_pass": identity,
           "symbols_identical": symbols_ok, "decoded_w1_identical": symbols_ok, "decoded_RGB_identical": rgb_ok,
           "DPB_state_identical": state_ok, "model_hash_before": before, "model_hash_after": after}
    write_csv(ROOT/"parts"/f"{t}_baseline_cells.csv", [row])
    write_csv(ROOT/"parts"/f"{t}_baseline_frames.csv", [dict(dataset=video["dataset"],video=video["name"],video_id=video["video_id"],qp=qp,trajectory="baseline",**r) for r in rows])
    return row, rows


def local_bits(model, symbols, scales, activity_scales):
    threshold = model.gaussian_encoder.force_zero_thres
    active = torch.ones_like(activity_scales, dtype=torch.bool) if threshold is None else activity_scales > threshold
    q = symbols.float()[active]
    if q.numel() == 0: return 0.0
    source = activity_scales.float()[active]
    scale = scales.float()[active]
    idx = ((torch.log(source.clamp(model.gaussian_encoder.scale_min, model.gaussian_encoder.scale_max))
            - model.gaussian_encoder.log_scale_min) * model.gaussian_encoder.log_step_recip).long().clamp(0,127)
    delta = (source/scale).clamp_min(1.0); ts = model.gaussian_encoder.scale_table.to(scale.device)[idx]/delta
    normal = torch.distributions.Normal(torch.zeros_like(ts), ts)
    prob = (normal.cdf(q+.5)-normal.cdf(q-.5)).clamp_min(1e-12)
    return float((-torch.log2(prob)).sum())


def g4_scores(video, qp, frame, baseline_est):
    rows = []
    with (V13/"block_probe.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"]==video["dataset"] and int(r["video_id"])==int(video["video_id"]) and int(r["qp"])==qp and int(r["frame"])==frame and int(r["mode"])==1:
                saving = baseline_est - float(r["estimated_w1_entropy_bits"])
                if saving > 0: rows.append((float(r["delta_MSE"])/saving, int(r["block_index"]), saving, float(r["delta_MSE"])))
    return sorted(rows)


def probe_scores(enc, dec, prep, gt, tile):
    h,w=prep["shape"][-2:]; gh,gw=codec.block_grid(h,w,tile,tile); total=gh*gw
    base_est=codec.gaussian_estimated_bits(enc,prep["w1"],prep["s1"],prep["s1"])
    base_mse=base.basic_metrics(prep["rgb"],gt)[0]; scores=[]; batch=[]
    def flush():
        nonlocal batch
        if not batch:return
        latents=torch.cat([x[4] for x in batch],0); ctx=prep["ctx"].expand(latents.shape[0],*prep["ctx"].shape[1:])
        feat=enc.dec(latents,ctx,prep["qd"]); rgb=enc.recon_generation_net(feat,prep["qr"])
        target=base.unit(gt); errs=(base.unit(rgb)-target).square().flatten(1).mean(1)
        for i,(bi,saving,dmse0,modes,latent) in enumerate(batch):
            dmse=float(errs[i])-base_mse
            if saving>0:scores.append((dmse/saving,bi,saving,dmse))
        batch=[]; del latents,ctx,feat,rgb,target,errs
    for bi in range(total):
        by,bx=divmod(bi,gw); ys=slice(by*tile,min((by+1)*tile,h)); xs=slice(bx*tile,min((bx+1)*tile,w))
        modes=[0]*total;modes[bi]=1; cand=codec.requantize_frame(enc,prep,modes,tile,tile,reconstruct_rgb=False)
        b0=local_bits(enc,prep["w1"][...,ys,xs],prep["s1"][...,ys,xs],prep["s1"][...,ys,xs])
        b1=local_bits(enc,cand["w1"][...,ys,xs],cand["s1"][...,ys,xs]/2,cand["s1"][...,ys,xs])
        batch.append((bi,b0-b1,0,modes,cand["latent"]))
        if len(batch)>=CFG["probe_batch_size"]:flush()
    flush(); return sorted(scores),base_est,total


def hypothetical_map_audit(modes):
    if all(x==0 for x in modes):
        return {"raw_bitmap_bits":len(modes),"rle_bits":0,"sparse_bits":0,"inverse_sparse_bits":0,
                "selected_syntax":"ORIGINAL_NO_MAP","selected_map_bits":0,"aligned_map_bytes":0}
    info=codec.encode_mode_map_best(modes)
    return {k:info[k] for k in ("raw_bitmap_bits","rle_bits","sparse_bits","inverse_sparse_bits")}|{"selected_syntax":info["syntax"],"selected_map_bits":info["bits"],"aligned_map_bytes":len(info["data"])}


@torch.no_grad()
def build_maps(video, frames, qp, gran, tile, device):
    gt=gran_tag(video,qp,gran); out_maps=ROOT/"parts"/f"{gt}_frame_candidate_maps.csv"
    out_audit=ROOT/"parts"/f"{gt}_mode_map_audit.csv"; done=ROOT/"parts"/f"{gt}_maps_done.json"
    if done.exists() and json.loads(done.read_text()).get("status")=="PASS":
        rows=read_csv(out_maps); return {fi:{r["candidate_name"]:json.loads(r["mode_map_json"]) for r in rows if int(r["frame"])==fi} for fi in range(1,64)},json.loads(done.read_text())["coverage"]
    iframe,enc,dec=base.load_triplet(device); base.init_sequence(iframe,enc,dec,frames[0],qp)
    rows=[];audits=[];proposals={};tiles_probed=eligible=actual=0
    old_base={}
    if gran=="G4":
        with (V13/"frame_candidate_maps.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                if r["dataset"]==video["dataset"] and int(r["video_id"])==int(video["video_id"]) and int(r["qp"])==qp and r["candidate_map"]=="all_1x":old_base[int(r["frame"])]=float(r["estimated_w1_entropy_bits"])
    for fi in range(1,64):
        enc.update(.12);dec.update(.12);enc.set_use_two_entropy_coders(True);dec.set_use_two_entropy_coders(True)
        aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);ds=base.snap_cpu(dec);prep=codec.prepare_frame(enc,frames[fi],aq,None,tile,tile)
        bp,bcomp=codec.encode_original(enc,prep,aq);bout,bcap=codec.decode_mixed(dec,bp,SPS,aq,tile,tile)
        base_mse,base_psnr,_=base.basic_metrics(bout,frames[fi]);h,w=prep["shape"][-2:];gh,gw=codec.block_grid(h,w,tile,tile);n=gh*gw
        if gran=="G4": scores=g4_scores(video,qp,fi,old_base[fi]);base_est=old_base[fi]
        else:scores,base_est,n=probe_scores(enc,dec,prep,frames[fi],tile)
        tiles_probed+=n;eligible+=len(scores);proposals[fi]={}
        for fraction in CFG["candidate_fractions"]:
            name=f"F{int(round(fraction*100)):03d}";count=0 if fraction==0 else min(len(scores),math.ceil(n*fraction))
            modes=[0]*n;nominal=0.0
            for _,bi,saving,_ in scores[:count]:modes[bi]=1;nominal+=saving
            if fraction==0:cand,payload,comp,out,met,cap_est=prep,bp,bcomp,bout,{"MSE":base_mse,"PSNR":base_psnr},base_est
            else:
                cand,payload,comp,out,met,_,_,cap_est=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile)
                actual+=1
            ma=hypothetical_map_audit(modes)
            rows.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"frame":fi,
              "granularity":gran,"candidate_name":name,"candidate_fraction":fraction,"number_tiles":n,
              "number_1x":modes.count(0),"number_2x":modes.count(1),"estimated_saved_bits":nominal,
              "estimated_w1_bits":cap_est,"actual_w1_bytes":comp["w1_bytes"],"actual_map_bits":comp.get("selected_map_bits",0),
              "actual_map_bytes":comp["mode_map_bytes"],"actual_frame_total_bytes":len(base.packet(payload,aq)),
              "RGB_MSE":met["MSE"],"PSNR":met["PSNR"],"mode_map_json":json.dumps(modes,separators=(",",":"))})
            audits.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"frame":fi,
              "granularity":gran,"candidate_fraction":fraction,**ma})
            proposals[fi][name]=modes
        base.restore_device(dec,ds,device);codec.decode_mixed(dec,bp,SPS,aq,tile,tile);enc.add_ref_frame(prep["feature"],None)
        if fi%4==0:
            write_csv(str(out_maps)+".partial",rows);write_csv(str(out_audit)+".partial",audits)
        print(gt,"maps",fi,"tiles",n,"eligible",len(scores),flush=True)
    write_csv(out_maps,rows);write_csv(out_audit,audits)
    coverage={"frames":64,"tiles_probed":tiles_probed,"one_to_two_proposals":tiles_probed,"candidates_with_estimated_saving":eligible,
              "candidate_maps":len(rows),"map_actual_RANS_encodes":actual,"G4_probe_reused":gran=="G4"}
    done.write_text(json.dumps({"status":"PASS","coverage":coverage},indent=2)+"\n")
    return proposals,coverage


def dedup(states):
    out={};pruned=0
    for s in states:
        key=(hashlib.sha256(s["stream"]).hexdigest(),base.state_hash(s["dec_state"]),s["cum_bytes"],struct.pack(">d",s["cum_sse"]))
        if key in out:pruned+=1
        else:out[key]=s
    return list(out.values()),pruned


def select_beam(states,baseline_prefix):
    states,dups=dedup(states);states=[s for s in states if s["cum_bytes"]/baseline_prefix<=CFG["prefix_max_rate_ratio"]]
    pareto=[]
    for i,a in enumerate(states):
        if not any(i!=j and b["cum_bytes"]<=a["cum_bytes"] and b["cum_sse"]<=a["cum_sse"] and
          (b["cum_bytes"]<a["cum_bytes"] or b["cum_sse"]<a["cum_sse"]) for j,b in enumerate(states)):pareto.append(a)
    chosen=[]
    for lo,hi in CFG["rate_buckets"]:
        pool=[s for s in pareto if lo<=s["cum_bytes"]/baseline_prefix<hi]
        if pool:chosen.append(min(pool,key=lambda s:(s["cum_sse"],s["cum_bytes"])))
    for order in (sorted(pareto,key=lambda s:(s["cum_sse"],s["cum_bytes"])),sorted(pareto,key=lambda s:(s["cum_bytes"],s["cum_sse"])),
                  sorted(states,key=lambda s:(s["cum_sse"],s["cum_bytes"]))):
        for s in order:
            if len(chosen)>=CFG["beam_width"]:break
            if s not in chosen:chosen.append(s)
    return chosen[:CFG["beam_width"]],dups


@torch.no_grad()
def search(video,frames,qp,gran,tile,device,proposals,baseline_row,baseline_frames):
    gt=gran_tag(video,qp,gran);done=ROOT/"parts"/f"{gt}_search_done.json"
    if done.exists() and json.loads(done.read_text()).get("status")=="PASS":
        return json.loads((ROOT/"artifacts"/f"{gt}_final_states.json").read_text()),json.loads(done.read_text())["coverage"]
    iframe,enc,dec=base.load_triplet(device);initial,rgb0,_=base.init_sequence(iframe,enc,dec,frames[0],qp)
    sse0=base.basic_metrics(rgb0,frames[0])[0];root={"tid":"root","enc_state":base.snap_cpu(enc),"dec_state":base.snap_cpu(dec),
      "stream":initial,"cum_bytes":len(initial),"cum_sse":sse0,"history":[],"n1":0,"n2":0,"modified":0,"baseline":True}
    baseline_state=root;beam=[];beam_rows=[];step_rows=[];serial=expanded=actual=dups=numerical=0
    baseline_prefix=len(initial);start_fi=1
    checkpoint=ROOT/"artifacts"/f"{gt}_search_checkpoint.pt"
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        baseline_state=saved["baseline_state"];beam=saved["beam"]
        beam_rows=saved["beam_rows"];step_rows=saved["step_rows"]
        serial=saved["serial"];expanded=saved["expanded"];actual=saved["actual"]
        dups=saved["dups"];numerical=saved["numerical"]
        baseline_prefix=saved["baseline_prefix"];start_fi=saved["next_fi"]
        print(gt,"resume checkpoint frame",start_fi,flush=True)
    crash_marker=ROOT/"sanity"/f"{gt}_active_candidate.json"
    for fi in range(start_fi,64):
        baseline_prefix+=int(baseline_frames[fi]["actual_bytes"]);aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);successors=[]
        for parent in [baseline_state]+beam:
            base.restore_device(enc,parent["enc_state"],device);base.restore_device(dec,parent["dec_state"],device)
            prep=codec.prepare_frame(enc,frames[fi],aq,None,tile,tile);ds=base.snap_cpu(dec);expanded+=1
            for name,modes in proposals[fi].items():
                try:
                    if video["dataset"]=="uvg" and int(video["video_id"])==2 and fi>=40:
                        crash_marker.write_text(json.dumps({"frame":fi,"parent":parent["tid"],"candidate":name},indent=2)+"\n")
                    original=name=="F000";cand,payload,comp,out,met,new_dec,cap,est=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,original=original)
                    actual+=1;base.restore_device(enc,parent["enc_state"],device);enc.add_ref_frame(cand["feature"],None)
                    pkt=base.packet(payload,aq);serial+=1;history=parent["history"]+[name]
                    state={"tid":f"{gt}_T{serial}","parent":parent["tid"],"enc_state":base.snap_cpu(enc),"dec_state":new_dec,
                      "stream":parent["stream"]+pkt,"cum_bytes":parent["cum_bytes"]+len(pkt),"cum_sse":parent["cum_sse"]+met["MSE"],
                      "history":history,"n1":parent["n1"]+modes.count(0),"n2":parent["n2"]+modes.count(1),
                      "modified":parent["modified"]+(name!="F000"),"baseline":parent["baseline"] and original,
                      "current_w1_hash":codec.tensor_hash(cap["w1"])}
                    successors.append(state)
                    step_rows.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
                      "frame":fi,"trajectory_id":state["tid"],"parent_trajectory_id":parent["tid"],"candidate_name":name,
                      "actual_frame_bytes":len(pkt),"cumulative_actual_bytes":state["cum_bytes"],"baseline_prefix_bytes":baseline_prefix,
                      "frame_RGB_MSE":met["MSE"],"cumulative_RGB_MSE":state["cum_sse"],"number_2x_tiles":modes.count(1),
                      "bitstream_hash":hashlib.sha256(state["stream"]).hexdigest(),"DPB_state_hash":base.state_hash(new_dec)})
                except (RuntimeError,ValueError,FloatingPointError):numerical+=1
        b=[s for s in successors if s["baseline"]]
        if len(b)!=1:raise RuntimeError(f"baseline path multiplicity {fi}")
        baseline_state=b[0];beam,d=select_beam([s for s in successors if not s["baseline"]],baseline_prefix);dups+=d
        for rank,s in enumerate(beam):beam_rows.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,
          "granularity":gran,"frame":fi,"trajectory_id":s["tid"],"parent_trajectory_id":s["parent"],"cumulative_actual_bytes":s["cum_bytes"],
          "baseline_prefix_bytes":baseline_prefix,"prefix_rate_ratio":s["cum_bytes"]/baseline_prefix,"cumulative_RGB_MSE":s["cum_sse"],
          "number_1x_tiles":s["n1"],"number_2x_tiles":s["n2"],"modified_frames":s["modified"],"beam_rank":rank,
          "bitstream_hash":hashlib.sha256(s["stream"]).hexdigest(),"DPB_state_hash":base.state_hash(s["dec_state"])})
        beam_rows.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
          "frame":fi,"trajectory_id":"BASELINE_PATH","parent_trajectory_id":"BASELINE_PATH","cumulative_actual_bytes":baseline_state["cum_bytes"],
          "baseline_prefix_bytes":baseline_prefix,"prefix_rate_ratio":1.0,"cumulative_RGB_MSE":baseline_state["cum_sse"],
          "number_1x_tiles":baseline_state["n1"],"number_2x_tiles":0,"modified_frames":0,"beam_rank":-1,
          "bitstream_hash":hashlib.sha256(baseline_state["stream"]).hexdigest(),"DPB_state_hash":base.state_hash(baseline_state["dec_state"])})
        if fi%4==0:
            write_csv(ROOT/"parts"/f"{gt}_beam_states.partial.csv",beam_rows);write_csv(ROOT/"parts"/f"{gt}_trajectory_steps.partial.csv",step_rows)
            torch.save({"baseline_state":baseline_state,"beam":beam,"beam_rows":beam_rows,"step_rows":step_rows,
              "serial":serial,"expanded":expanded,"actual":actual,"dups":dups,"numerical":numerical,
              "baseline_prefix":baseline_prefix,"next_fi":fi+1},checkpoint)
        print(gt,"beam",fi,len(beam),"ratio",min((s["cum_bytes"]/baseline_prefix for s in beam),default=1),flush=True)
    write_csv(ROOT/"parts"/f"{gt}_beam_states.csv",beam_rows);write_csv(ROOT/"parts"/f"{gt}_trajectory_steps.csv",step_rows)
    final=[]
    for s in beam:
        path=ROOT/"bitstreams"/f"{s['tid']}.bin";path.write_bytes(s["stream"])
        final.append({"trajectory_id":s["tid"],"history":s["history"],"total_bytes":s["cum_bytes"],"cumulative_RGB_MSE":s["cum_sse"],
          "number_1x_tiles":s["n1"],"number_2x_tiles":s["n2"],"modified_frames":s["modified"],"bitstream_path":str(path),"bitstream_sha256":sha256(path)})
    (ROOT/"artifacts"/f"{gt}_final_states.json").write_text(json.dumps(final,separators=(",",":"))+"\n")
    coverage={"actual_RANS_encodes":actual,"beam_states_expanded":expanded,"duplicate_beam_states_pruned":dups,
              "final_unique_bitstreams":len({r["bitstream_sha256"] for r in final}),"final_unique_DPB_states":len({base.state_hash(s["dec_state"]) for s in beam}),
              "numerical_failures":numerical}
    done.write_text(json.dumps({"status":"PASS","coverage":coverage},indent=2)+"\n")
    checkpoint.unlink(missing_ok=True);crash_marker.unlink(missing_ok=True)
    return final,coverage


@torch.no_grad()
def run_control(video,frames,qp,device,metrics,name,mode=None,fraction=None):
    t=tag(video,qp);iframe,enc,dec=base.load_triplet(device);stream,rgb0,ip=base.init_sequence(iframe,enc,dec,frames[0],qp)
    rows=[dict(frame=0,actual_qp=qp,actual_bytes=len(base.packet(ip,qp,True)),**base.frame_metric(rgb0,frames[0],metrics))]
    totals={"I_bytes":len(ip),"z_bytes":0,"w0_bytes":0,"w1_bytes":0,"mode_map_bits":0,"mode_map_bytes":0}
    n1=n2=n4=0
    for fi in range(1,64):
        aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);tile=8 if fraction is not None else 4
        prep=codec.prepare_frame(enc,frames[fi],aq,None,tile,tile);n=len(prep["modes"])
        if fraction is not None:
            count=math.ceil(n*fraction);modes=[0]*n
            seed=int(hashlib.sha256(f"{CFG['random_seed']}:{video['dataset']}:{video['video_id']}:{qp}:{fraction}:{fi}".encode()).hexdigest()[:16],16)
            indices=list(range(n));random.Random(seed).shuffle(indices)
            for i in indices[:count]:modes[i]=1
            global_mode=None
        else:modes=[mode]*n;global_mode=mode
        ds=base.snap_cpu(dec);cand,payload,comp,out,met,new_dec,cap,est=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,global_mode=global_mode)
        enc.add_ref_frame(cand["feature"],None);pkt=base.packet(payload,aq);stream+=pkt
        for k in ("z_bytes","w0_bytes","w1_bytes","mode_map_bytes"):totals[k]+=comp[k]
        totals["mode_map_bits"]+=comp.get("selected_map_bits",0);n1+=modes.count(0);n2+=modes.count(1);n4+=modes.count(2)
        rows.append(dict(frame=fi,actual_qp=aq,actual_bytes=len(pkt),**base.frame_metric(out,frames[fi],metrics)))
    path=ROOT/"bitstreams"/f"{t}_{name}.bin";path.write_bytes(stream);avg=average(rows)
    totals["header_bytes"]=len(stream)-sum(totals[k] for k in ("I_bytes","z_bytes","w0_bytes","w1_bytes","mode_map_bytes"))
    result={"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"control":name,
      "fraction":"" if fraction is None else fraction,"global_delta":"" if mode is None else codec.MODE_TO_DELTA[mode],
      "total_tiles":n1+n2+n4,"number_1x_tiles":n1,"number_2x_tiles":n2,"number_4x_tiles":n4,
      "total_actual_bytes":len(stream),"actual_bits":len(stream)*8,"bpp":len(stream)*8/(64*1920*1080),
      "kbps":len(stream)*8*video["fps"]/64/1000,**totals,**avg,"bitstream_path":str(path),"bitstream_sha256":sha256(path),"decode_pass":True}
    return result,rows


@torch.no_grad()
def replay_and_audit(video,frames,qp,gran,tile,device,metrics,proposals,state,target):
    history=state["history"];source=Path(state["bitstream_path"])
    iframe,enc,dec=base.load_triplet(device);stream,rgb0,ip=base.init_sequence(iframe,enc,dec,frames[0],qp)
    expected=[];totals={"I_bytes":len(ip),"z_bytes":0,"w0_bytes":0,"w1_bytes":0,"mode_map_bits":0,"mode_map_bytes":0}
    for fi,name in enumerate(history,1):
        aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);prep=codec.prepare_frame(enc,frames[fi],aq,None,tile,tile);modes=proposals[fi][name]
        ds=base.snap_cpu(dec);cand,payload,comp,out,met,new_dec,cap,est=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,original=name=="F000")
        enc.add_ref_frame(cand["feature"],None);stream+=base.packet(payload,aq)
        for k in ("z_bytes","w0_bytes","w1_bytes","mode_map_bytes"):totals[k]+=comp[k]
        totals["mode_map_bits"]+=comp.get("selected_map_bits",0)
        expected.append({"modes":modes,"w1":codec.tensor_hash(cap["w1"]),"latent":codec.tensor_hash(cap["latent"]),
                         "feature":codec.tensor_hash(cap["feature"]),"rgb":codec.tensor_hash(out),"state":base.state_hash(new_dec)})
    if stream!=source.read_bytes():raise RuntimeError("selected trajectory replay differs from beam stream")
    path=ROOT/"bitstreams"/f"{gran_tag(video,qp,gran)}_target_{target:.3f}_{state['trajectory_id']}.bin";path.write_bytes(stream)
    payloads,cumulative=base.parse_stream(path);iframe2,_,dec2=base.load_triplet(device);dec2.clear_dpb();dec2.set_curr_poc(0)
    metric_rows=[];symbols_ok=maps_ok=latent_ok=state_ok=rgb_ok=True
    for fi,(is_i,aq,payload) in enumerate(payloads):
        if is_i:
            out=iframe2.decompress(payload,SPS,aq)["x_hat"];dec2.add_ref_frame(None,out)
        else:
            out,cap=codec.decode_mixed(dec2,payload,SPS,aq,tile,tile);exp=expected[fi-1];parsed=codec.parse_mixed(payload,tile,tile)
            decoded_modes=[0]*len(exp["modes"]) if parsed is None else parsed["modes"]
            maps_ok &= decoded_modes==exp["modes"];symbols_ok &= codec.tensor_hash(cap["w1"])==exp["w1"]
            latent_ok &= codec.tensor_hash(cap["latent"])==exp["latent"]
            state_ok &= codec.tensor_hash(cap["feature"])==exp["feature"] and base.state_hash(base.snap_cpu(dec2))==exp["state"]
            rgb_ok &= codec.tensor_hash(out)==exp["rgb"]
        metric_rows.append(dict(frame=fi,actual_qp=aq,actual_bytes=len(base.packet(payload,aq,is_i)),**base.frame_metric(out,frames[fi],metrics)))
    passed=len(payloads)==64 and cumulative[-1]==path.stat().st_size and symbols_ok and maps_ok and latent_ok and state_ok and rgb_ok
    totals["header_bytes"]=len(stream)-sum(totals[k] for k in ("I_bytes","z_bytes","w0_bytes","w1_bytes","mode_map_bytes"))
    audit={"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
      "trajectory_id":state["trajectory_id"],"target_rate_ratio":target,"frames":len(payloads),"precision_map_equal":maps_ok,
      "w1_symbol_equal":symbols_ok,"decoded_latent_equal":latent_ok,"DPB_state_equal":state_ok,"RGB_equal":rgb_ok,
      "bytes_consumed_equal":cumulative[-1]==path.stat().st_size,"decode_audit_pass":passed,"bitstream_path":str(path),"bitstream_sha256":sha256(path)}
    return audit,metric_rows,average(metric_rows),totals


def process_cell(video,qp,device):
    t=tag(video,qp);done=ROOT/"parts"/f"{t}_done.json"
    if done.exists() and json.loads(done.read_text()).get("status")=="PASS":print("reuse",t,flush=True);return
    frames=base.read_frames(video,device);metrics=base.init_metric_models(device)
    baseline_row,baseline_frames=baseline(video,frames,qp,device,metrics)
    all_map_rows=[];all_map_audit=[];all_beam=[];all_steps=[];all_traj=[];all_final=[];all_audits=[];all_coverage=[]
    mixed_frame_rows=[];mixed_seq=[]
    for gran,tile in CFG["granularities"].items():
        proposals,cov1=build_maps(video,frames,qp,gran,int(tile),device)
        states,cov2=search(video,frames,qp,gran,int(tile),device,proposals,baseline_row,baseline_frames)
        all_map_rows.extend(read_csv(ROOT/"parts"/f"{gran_tag(video,qp,gran)}_frame_candidate_maps.csv"))
        all_map_audit.extend(read_csv(ROOT/"parts"/f"{gran_tag(video,qp,gran)}_mode_map_audit.csv"))
        all_beam.extend(read_csv(ROOT/"parts"/f"{gran_tag(video,qp,gran)}_beam_states.csv"))
        all_steps.extend(read_csv(ROOT/"parts"/f"{gran_tag(video,qp,gran)}_trajectory_steps.csv"))
        selected_ids=set();cache={}
        for target in CFG["target_rate_ratios"]:
            target_bytes=math.floor(int(baseline_row["baseline_bytes"])*target)
            feasible=[s for s in states if int(s["total_bytes"])<=target_bytes]
            best=min(feasible,key=lambda s:float(s["cumulative_RGB_MSE"])) if feasible else None
            if best is None:
                all_final.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
                  "target_rate_ratio":target,"baseline_bytes":baseline_row["baseline_bytes"],"target_bytes":target_bytes,"selected_bytes":"",
                  "selected_rate_ratio":"","selection_type":"TARGET_NOT_REACHED","I_bytes":"","z_bytes":"","w0_bytes":"","w1_bytes":"",
                  "mode_map_bits":"","mode_map_bytes":"","header_bytes":"","modified_frames":"","total_tiles":"","number_1x_tiles":"",
                  "number_2x_tiles":"","PSNR":"","MS_SSIM":"","LPIPS":"","DISTS":"","decode_audit_pass":"","bitstream_path":"","bitstream_sha256":""})
                continue
            selected_ids.add(best["trajectory_id"])
            if best["trajectory_id"] not in cache:
                audit,fr,avg,totals=replay_and_audit(video,frames,qp,gran,int(tile),device,metrics,proposals,best,target)
                cache[best["trajectory_id"]]=(audit,fr,avg,totals);all_audits.append(audit)
                mixed_frame_rows.extend(dict(dataset=video["dataset"],video=video["name"],video_id=video["video_id"],qp=qp,
                  trajectory=best["trajectory_id"],granularity=gran,**r) for r in fr)
                mixed_seq.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"trajectory":best["trajectory_id"],
                  "granularity":gran,"total_actual_bytes":best["total_bytes"],"bpp":int(best["total_bytes"])*8/(64*1920*1080),
                  "kbps":int(best["total_bytes"])*8*video["fps"]/64/1000,**avg})
            audit,fr,avg,totals=cache[best["trajectory_id"]]
            path=audit["bitstream_path"]
            all_final.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
              "target_rate_ratio":target,"baseline_bytes":baseline_row["baseline_bytes"],"target_bytes":target_bytes,"selected_bytes":best["total_bytes"],
              "selected_rate_ratio":int(best["total_bytes"])/int(baseline_row["baseline_bytes"]),"selection_type":"MIXED",**totals,
              "modified_frames":best["modified_frames"],"total_tiles":int(best["number_1x_tiles"])+int(best["number_2x_tiles"]),
              "number_1x_tiles":best["number_1x_tiles"],"number_2x_tiles":best["number_2x_tiles"],"PSNR":avg["PSNR"],
              "MS_SSIM":avg["MS_SSIM"],"LPIPS":avg["LPIPS"],"DISTS":avg["DISTS"],"decode_audit_pass":audit["decode_audit_pass"],
              "bitstream_path":path,"bitstream_sha256":audit["bitstream_sha256"]})
        for s in states:
            all_traj.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
              **{k:v for k,v in s.items() if k!="history"},"mode_history_json":json.dumps(s["history"]),"is_selected":s["trajectory_id"] in selected_ids})
        all_coverage.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"granularity":gran,
          **cov1,**cov2,"decode_failures":sum(not a["decode_audit_pass"] for a in all_audits if a["granularity"]==gran)})
    uniform=[];random_rows=[];control_frames=[];control_seq=[]
    for mode in (1,2):
        r,fr=run_control(video,frames,qp,device,metrics,f"uniform_{int(codec.MODE_TO_DELTA[mode])}x",mode=mode);uniform.append(r)
        control_frames.extend(dict(dataset=video["dataset"],video=video["name"],video_id=video["video_id"],qp=qp,trajectory=r["control"],granularity="GLOBAL",**x) for x in fr)
        control_seq.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"trajectory":r["control"],"granularity":"GLOBAL",
          "total_actual_bytes":r["total_actual_bytes"],"bpp":r["bpp"],"kbps":r["kbps"],"MSE":r["MSE"],"PSNR":r["PSNR"],"MS_SSIM":r["MS_SSIM"],"LPIPS":r["LPIPS"],"DISTS":r["DISTS"]})
    for fraction in CFG["random_control_fractions"]:
        r,fr=run_control(video,frames,qp,device,metrics,f"random_G8_{int(fraction*100):02d}pct",fraction=fraction);random_rows.append(r)
        control_frames.extend(dict(dataset=video["dataset"],video=video["name"],video_id=video["video_id"],qp=qp,trajectory=r["control"],granularity="G8",**x) for x in fr)
        control_seq.append({"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"trajectory":r["control"],"granularity":"G8",
          "total_actual_bytes":r["total_actual_bytes"],"bpp":r["bpp"],"kbps":r["kbps"],"MSE":r["MSE"],"PSNR":r["PSNR"],"MS_SSIM":r["MS_SSIM"],"LPIPS":r["LPIPS"],"DISTS":r["DISTS"]})
    write_csv(ROOT/"parts"/f"{t}_frame_candidate_maps.csv",all_map_rows);write_csv(ROOT/"parts"/f"{t}_mode_map_audit.csv",all_map_audit)
    write_csv(ROOT/"parts"/f"{t}_beam_states.csv",all_beam);write_csv(ROOT/"parts"/f"{t}_trajectory_steps.csv",all_steps)
    write_csv(ROOT/"parts"/f"{t}_final_trajectories.csv",all_traj);write_csv(ROOT/"parts"/f"{t}_final_streams.csv",all_final)
    write_csv(ROOT/"parts"/f"{t}_uniform_controls.csv",uniform);write_csv(ROOT/"parts"/f"{t}_random_controls.csv",random_rows)
    write_csv(ROOT/"parts"/f"{t}_bitstream_audit.csv",all_audits);write_csv(ROOT/"parts"/f"{t}_search_coverage.csv",all_coverage)
    base_frame=[dict(dataset=video["dataset"],video=video["name"],video_id=video["video_id"],qp=qp,trajectory="baseline",granularity="BASELINE",**r) for r in baseline_frames]
    write_csv(ROOT/"parts"/f"{t}_frame_metrics.csv",base_frame+control_frames+mixed_frame_rows)
    base_seq={"dataset":video["dataset"],"video":video["name"],"video_id":video["video_id"],"qp":qp,"trajectory":"baseline","granularity":"BASELINE",
      "total_actual_bytes":baseline_row["baseline_bytes"],"bpp":baseline_row["bpp"],"kbps":baseline_row["kbps"],"MSE":baseline_row["MSE"],"PSNR":baseline_row["PSNR"],
      "MS_SSIM":baseline_row["MS_SSIM"],"LPIPS":baseline_row["LPIPS"],"DISTS":baseline_row["DISTS"]}
    write_csv(ROOT/"parts"/f"{t}_sequence_metrics.csv",[base_seq]+control_seq+mixed_seq)
    ok=baseline_row["identity_gate_pass"] and all(a["decode_audit_pass"] for a in all_audits)
    done.write_text(json.dumps({"status":"PASS" if ok else "FAIL","cell":t,"baseline_identity_pass":baseline_row["identity_gate_pass"],
      "G4_completed":True,"G8_completed":True,"G16_completed":True,"selected_streams":len(all_audits),
      "selected_decode_pass":sum(a["decode_audit_pass"] for a in all_audits),"uniform_controls":len(uniform),"random_controls":len(random_rows)},indent=2)+"\n")
    if not ok:raise RuntimeError("cell integrity failed")


def main():
    faulthandler.enable()
    ap=argparse.ArgumentParser();ap.add_argument("--gpu",type=int,required=True);ap.add_argument("--cells",required=True);ap.add_argument("--worker",default="")
    args=ap.parse_args();log=ROOT/"logs"/f"gpu{args.gpu}_{args.worker or 'run'}.log";sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    torch.manual_seed(CFG["random_seed"]);device=torch.device(f"cuda:{args.gpu}");torch.cuda.set_device(device)
    lookup={(v["dataset"],int(v["video_id"])):v for v in MAN["videos"]}
    for spec in args.cells.split(","):
        ds,vid,qp=spec.split(":");video=lookup[(ds,int(vid))]
        try:process_cell(video,int(qp),device)
        except Exception as exc:
            (ROOT/"parts"/f"{tag(video,int(qp))}_FAILED.json").write_text(json.dumps({"status":"FAIL","error":repr(exc),"traceback":traceback.format_exc()},indent=2)+"\n")
            print("FAILED",spec,repr(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":main()
