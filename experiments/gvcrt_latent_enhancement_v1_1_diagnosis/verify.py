"""Validate V1.1 outputs and the added oracle step-zero autograd path, without optimization."""
import argparse
import json
from common import *
from postprocess import read_csv
from prepare import inventory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    args=parser.parse_args()
    out=HERE/args.output
    configure()
    status=json.loads((out/"status.json").read_text())
    assert status["status"]=="COMPLETE"
    initial=json.loads((HERE/"input_audit.json").read_text())
    assert initial["v1_inventory"]==inventory()
    for path,digest in initial["used_files_sha256"].items():
        assert sha(path)==digest
    assert json.loads((out/"fixed_config.json").read_text())==CFG
    af=read_csv(out/"source_ablation_frames.csv")
    ag=read_csv(out/"source_ablation.csv")
    eq=read_csv(out/"exact_equality.csv")
    om=read_csv(out/"oracle_metrics.csv")
    refs=read_csv(out/"oracle_references.csv")
    assert len(af)==1440 and len(eq)==270 and len(om)==120 and len(refs)==24
    coarse=[r for r in eq if r["delta"]==2]
    assert len(coarse)==90
    for row in coarse:
        assert all(row[k] is True for k in row if k.endswith("_equal"))
        assert all(row[k]==0 for k in row if k.endswith("_max_abs"))
    for row in af:
        assert all(np.isfinite(row[k]) for k in ["psnr","ms_ssim","lpips"])
        if row["mode"] in ["ZERO","WRONG_SOURCE","CONTINUOUS"]:
            assert row["formal_enhancement_bits"] is None and row["diagnostic_intervention"]
    old=read_csv(V1/"run_v1/aggregate_metrics.csv")
    for level in range(3):
        a=next(r for r in ag if r["video"]=="ALL_VAL6" and r["scope"]=="IP" and r["mode"]=="REAL" and r["level"]==level)
        b=next(r for r in old if r["method"]==f"quantized_d_val6_l{level}")
        assert all(abs(a[k]-b[k])<1e-12 for k in ["psnr","ms_ssim","lpips"])
    checks=json.loads((out/"oracle_checks.json").read_text())
    assert len(checks["runs"])==24 and not checks["failures"] and checks["network_updates"]==0
    assert checks["total_latent_optimizer_updates"]==3600 and checks["model_weights_unchanged"]
    for row in om:
        assert row["step"] in CFG["oracle"]["snapshots"]
        assert np.isfinite(row["relative_delta_l2"]) and row["transmittable"] is False
        if row["step"]==0:
            assert row["delta_latent_l2"]==row["delta_latent_max_abs"]==0
            mode="REFERENCE_FP32" if row["initialization"]=="ORIGINAL_START" else "ZERO_START_C_DELTA2"
            ref=next(r for r in refs if r["sample"]==row["sample"] and r["mode"]==mode)
            assert all(abs(row[k]-ref[k])<1e-12 for k in ["psnr","ms_ssim","lpips"])
    # Specific added-path check: identical start latent with/without a gradient leaf.
    g=generator()
    enh,control,entropy,ck=models(CFG["oracle"]["checkpoint"])
    before=digest_state(g)
    starts_checked=[]
    samples=json.loads((HERE/"fixed_samples.json").read_text())
    for sample in samples:
        rows=torch.load(V1/"run_v1/base"/f'{sample["video"]}_q0.pt',map_location="cpu",weights_only=False)
        row=rows[sample["frame"]]
        ell,q=row["ell"].cuda().float(),row["q_recon"].cuda().float()
        with torch.no_grad():
            zero=ell+enh.decoder(dequantize(torch.zeros((1,8,17,30),dtype=torch.int16),2.0),ell)
        for initialization,start in [("ORIGINAL_START",ell),("ZERO_START",zero)]:
            with torch.no_grad():
                ref=g(start,q)
            leaf=torch.zeros_like(start,requires_grad=True)
            value=g(start+leaf,q)
            comp=difference(value.detach(),ref)
            assert comp["equal"]
            starts_checked.append({"sample":sample["sample"],"initialization":initialization,**comp})
            del value,leaf,ref
        del rows
    assert digest_state(g)==before
    result={"all_checks_passed":True,"v1_file_inventory_unchanged":True,"input_hashes_unchanged":True,
            "frame_ablation_rows":len(af),"delta2_exact_P_frames":90,"real_metrics_match_v1":True,
            "oracle_metric_rows":120,"completed_oracle_runs":24,"latent_updates":3600,"network_updates":0,
            "oracle_step_zero_metrics_match_reference":True,"oracle_grad_enabled_step_zero_exact":starts_checked,
            "configuration_sha256":sha(HERE/"config.json"),"sample_selection_sha256":sha(HERE/"fixed_samples.json"),
            "donor_mapping_sha256":sha(HERE/"donor_mapping.json"),
            "source_code_sha256":{p.name:sha(p) for p in sorted(HERE.glob("*.py"))}}
    save_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
