"""Stage A. Real source frames and bitstreams only. No enhancement training before gates."""
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["gpu_uuid"]
sys.path.insert(0, str(HERE.parents[1]))

import torch
from torch.nn import functional as F
from codec import (ROOT, load_models, strict_load, digest_state, encode_base, decode_base,
                   jsonable_rows, source_frame)
from enhancement import Enhancement, BaseOnly, FactorizedEntropy
from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
from src.models.video_model_gvcrt import DMC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="stage_a_v2")
    args = parser.parse_args()
    out = HERE / args.output
    out.mkdir(exist_ok=False)
    report = {"status": "RUNNING", "fused_available": CUSTOMIZED_CUDA_INFERENCE}
    def save():
        (out / "audit.json").write_text(json.dumps(report, indent=2))
    try:
        torch.set_num_threads(CFG["execution"]["torch_threads"])
        torch.manual_seed(CFG["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report["environment"] = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                                 "torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(0), "gpu_uuid": CFG["gpu_uuid"],
                                 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                 "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)}
        (out / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        (out / "nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
        i, p, weights = load_models(CFG)
        report["weights"] = weights
        save()
        manifest = json.loads((HERE / "manifest.json").read_text())
        entry = dict(manifest["train"][0], frames=8)
        rows_enc = encode_base(i,p,CFG,entry,0,out / "audit_base.bin")
        rows = decode_base(i,p,out / "audit_base.bin", bypass_check=True)
        # A second separately checkpoint-loaded decoder proves causal replay and no hidden source dependency.
        del i, p
        torch.cuda.empty_cache()
        i2, p2, _ = load_models(CFG)
        replay = decode_base(i2,p2,out / "audit_base.bin", bypass_check=False)
        report["causal_replay_exact"] = all(torch.equal(a["x_base"],b["x_base"]) and a["dpb"] == b["dpb"] for a,b in zip(rows,replay))
        report["frames"] = jsonable_rows(rows)
        report["encoding"] = rows_enc
        assert report["causal_replay_exact"]
        # Deliberately accelerated I/P switch & feature-reset unit sequence, distinct from the working-point data.
        special_cfg = dict(CFG, intra_period=8, reset_interval=4)
        special_entry = dict(manifest["train"][0], frames=12)
        encode_base(i2,p2,special_cfg,special_entry,0,out / "reset_switch.bin")
        special = decode_base(i2,p2,out / "reset_switch.bin", bypass_check=True)
        special_replay = decode_base(i2,p2,out / "reset_switch.bin", bypass_check=False)
        report["reset_switch_exact"] = all(torch.equal(a["x_base"],b["x_base"]) and a["dpb"]==b["dpb"] for a,b in zip(special,special_replay))
        report["reset_switch_frames"] = jsonable_rows(special)
        assert report["reset_switch_exact"]
        save()
        del i2, p2, replay, special, special_replay
        torch.cuda.empty_cache()
        # Fresh checkpoint instance; never switch a fused inference model back to torch.
        model = DMC()
        strict_load(model, ROOT / CFG["checkpoint_p"], "P")
        model.requires_grad_(False).eval().cuda()
        if CUSTOMIZED_CUDA_INFERENCE:
            raise RuntimeError("Fused extension appeared; training pure-torch isolation must be explicitly implemented")
        generator = model.recon_generation_net.decoder
        sample = rows[1]
        ell, q = sample["ell"].cuda().float(), sample["q_recon"].cuda().float()
        x = source_frame(entry["video_id"], 1)
        x_pad = F.pad(x, (0,0,0,8), mode="replicate")
        with torch.no_grad():
            reference = generator(ell,q)
        half_base = sample["x_base"].cuda().float()
        diff = (reference.clamp(-1,1)-half_base.clamp(-1,1))/2
        report["fp32_vs_official_fp16"] = {"max_abs_rgb01": diff.abs().max().item(), "mse_rgb01": diff.square().mean().item(),
                                            "note": "Same FP16-decoded ell/q; full FP32 generator. This difference is not enhancement gain."}
        # Same precision/path comparison uses a separate fresh generator copy.
        other = DMC()
        strict_load(other, ROOT / CFG["checkpoint_p"], "P")
        other.requires_grad_(False).eval().cuda().half()
        with torch.no_grad():
            result = other.recon_generation_net.decoder(ell.half(),q.half())
        report["torch_vs_official_same_precision_exact"] = torch.equal(result.cpu(),sample["x_base"])
        assert report["torch_vs_official_same_precision_exact"]
        del other, result
        torch.cuda.empty_cache()
        before = digest_state(model)
        enh = Enhancement(CFG["architecture"]).cuda()
        base_only = BaseOnly(CFG["architecture"])
        report["parameters"] = {"encoder": sum(p.numel() for p in enh.encoder.parameters()),
                                "decoder": sum(p.numel() for p in enh.decoder.parameters()),
                                "base_only": sum(p.numel() for p in base_only.parameters()),
                                "entropy": CFG["architecture"]["enhancement_channels"]}
        optimizer = torch.optim.Adam(enh.parameters(), lr=CFG["training"]["lr"])
        grads = []
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            corrected, _ = enh(x_pad,ell)
            # Frozen generator parameters, but NEVER no_grad on corrected latent -> RGB.
            image = (generator(corrected,q)[:,:,:1080]+1)/2
            loss = F.mse_loss(image,x)
            loss.backward()
            grad = {"step": step, "mse": loss.item()}
            for name, sub in [("encoder",enh.encoder),("decoder",enh.decoder)]:
                grad[name] = sum(v.grad.detach().float().square().sum().item() for v in sub.parameters() if v.grad is not None)**.5
            if not all(torch.isfinite(v.grad).all() for v in enh.parameters() if v.grad is not None):
                raise RuntimeError("Non-finite enhancement gradient")
            assert all(v.grad is None for v in model.parameters())
            grads.append(grad)
            optimizer.step()
            print("gradient",grad,flush=True)
        torch.cuda.synchronize()
        report["gradient_steps"] = grads
        report["gradient_seconds"] = time.perf_counter()-start
        report["gradient_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["base_weights_unchanged"] = digest_state(model) == before
        assert report["base_weights_unchanged"] and grads[-1]["encoder"] > 0 and grads[-1]["decoder"] > 0
        save()
        torch.save({"enhancement": enh.state_dict(), "diagnostic_only": True},out / "gradient_diagnostic.pt")
        # Real enhancement symbols from this source, plus native boundary checks.
        entropy = FactorizedEntropy(CFG["architecture"]["enhancement_channels"]).cuda()
        with torch.no_grad():
            symbols = (enh.encoder(x_pad,ell)/.5).round()
        stream = entropy.encode(symbols,.5,1)
        (out / "enhancement_roundtrip.bin").write_bytes(stream)
        receiver_entropy = FactorizedEntropy(CFG["architecture"]["enhancement_channels"])
        receiver_entropy.load_state_dict(entropy.cpu().state_dict(),strict=True)
        decoded, delta, level = receiver_entropy.decode((out / "enhancement_roundtrip.bin").read_bytes())
        assert torch.equal(symbols.cpu().short(),decoded)
        boundary = torch.tensor([-128,-127,-1,0,1,126,127,0],dtype=torch.float32).reshape(1,8,1,1)
        bstream = receiver_entropy.encode(boundary,.5,1)
        assert torch.equal(receiver_entropy.decode(bstream)[0],boundary.short())
        try:
            receiver_entropy.encode(boundary+256,.5,1)
        except OverflowError:
            overflow_rejected = True
        else:
            raise AssertionError("Overflow not rejected")
        report["enhancement_entropy"] = {"roundtrip_exact": True, "boundary_roundtrip_exact": True,
                                         "overflow_rejected": overflow_rejected, "actual_bits": len(stream)*8,
                                         "estimated_bits": receiver_entropy.bits(symbols.cpu(),.5).item(),
                                         "header_bytes": receiver_entropy.HEADER.size,
                                         "symbol_min": symbols.min().item(),"symbol_max":symbols.max().item()}
        report["status"] = "INTERFACE-PASS"
        save()
        print(json.dumps({k:v for k,v in report.items() if k not in ["frames","reset_switch_frames","environment"]},indent=2))
    except Exception:
        report["status"] = "BLOCKED/INCONCLUSIVE"
        report["exception"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
