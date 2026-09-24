"""Freeze samples, identities, and code-domain numerical scales on the six training frames."""
import subprocess
from bridge import *


def run(out,bundle):
    specs=sample_specs()
    assert len([s for s in specs if s["group"]=="network_train6"])==6
    assert len([s for s in specs if s["group"]=="Train2_rest24"])==24
    assert len([s for s in specs if s["group"]=="Val6_90P"])==90
    save_json(out/"samples.json",specs)
    save_json(out/"manifest.json",manifest())
    mapping=[]
    for split in ["Train2","Val6"]:
        ee=entries(split)
        for i,e in enumerate(ee):
            mapping.append({"split":split,"target":e["video_id"],"donor":ee[(i+1)%len(ee)]["video_id"],"frame_rule":"same P-frame index"})
    save_json(out/"donor_mapping.json",mapping)
    caches=out/"input_cache"
    caches.mkdir()
    stats=[]
    for s in [s for s in specs if s["group"]=="network_train6"]:
        d,seconds=load_sample(s)
        with torch.no_grad():
            ec=bundle.fixed.ell_c(d["ell_base"].cuda()).cpu()
            assert torch.equal(ec,d["ell_c"])
            t=time.perf_counter()
            u=bundle.net.analyze(d["r_star"].cuda(),ec.cuda())
            torch.cuda.synchronize()
            forward=time.perf_counter()-t
        assert u.numel()==4080
        context=bundle.context(d)
        zero=torch.zeros_like(u,requires_grad=True)
        y,r=bundle.render(zero,context)
        with torch.no_grad():
            baseline=rgb01(bundle.fixed.g(context[0],context[1]))
            canonical=bundle.net.synthesize(torch.zeros_like(u),context[0])
        assert torch.equal(y.detach(),baseline) and torch.count_nonzero(r.detach())==0 and torch.count_nonzero(canonical)==0
        x=source_frame(s["video"],s["frame"])
        loss=F.mse_loss(y,x)+.001*bundle.lp(y,x,normalize=True).mean()
        g=torch.autograd.grad(loss,zero)[0]
        assert torch.isfinite(g).all() and g.norm()>0
        with torch.no_grad():
            direct=bundle.net.synthesize(u,context[0])
            cached=(bundle.net.decoder(u,context[0])-context[2])*bundle.net.residual_scale
            assert torch.equal(direct,cached)
            h=ORC.decode(OLD/"results_v1/train2_full_CODED_l3"/f'{s["video"]}.orc',V1/"run_v1/base"/f'{s["video"]}_q0.bin',MODEL,bundle.net.entropy)
            assert torch.equal((u/DELTA).round().cpu().short(),h["frames"][s["frame"]])
        save_pt(caches/f'{s["sample"]}.pt',{"encoder_u":u.cpu(),"spec":s,"base_cache_load_seconds":seconds,"encoder_forward_seconds":forward})
        stats.append({"sample":s["sample"],"u_rms":u.square().mean().sqrt().item(),"quant_domain_rms":(u/DELTA).square().mean().sqrt().item(),
                      "zero_start_gradient_norm":g.norm().item(),"zero_exact":True,"normalization_preserved":True,
                      "current_encoder_symbols_match_V12":True})
        del y,r,loss,g,zero
    median=float(np.median([s["u_rms"] for s in stats]))
    calibration={"physical_u_lr":.02*median,"quant_domain_z_lr":.02*median/DELTA,"median_six_encoder_u_rms":median,
                 "rule":CONFIG["lr_rule"],"scale_corrections":0,"statistics":stats}
    save_json(out/"optimization_scale.json",calibration)
    bundle.check()
    save_json(out/"interface_checks.json",{"code_shape":CONFIG["code_shape"],"code_scalars":4080,"delta":DELTA,"level":3,
              "continuous_pre_quantizer":"u=E_res(r_star/residual_scale,ell_c)",
              "quantized_decoder_input":"u_hat=delta*round(u/delta); no mean or other source scale",
              "zero_domains_agree":True,"F_zero_exact":True,"zero_start_has_gradient":True,
              "levels_change_only_delta":True,"shared_normalization_frozen":True,"models":bundle.fingerprints})
    print("CALIBRATION",json.dumps(calibration),flush=True)
