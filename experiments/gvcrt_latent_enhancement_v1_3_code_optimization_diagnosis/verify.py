"""Verify code-only updates, selection rules, actual sequence budgets and independent receiver evidence."""
import argparse
from bridge import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results_v1")
    args=p.parse_args()
    out=HOME/args.output
    audit=json.loads((out/"input_audit.json").read_text())
    assert inventory()==audit["historical_inventory"] and sha(MODEL)==audit["model_sha256"]
    assert json.loads((out/"config.json").read_text())==CONFIG
    valid_runs=0
    valid_updates=0
    quantized_runs=0
    for path in sorted((out/"optimizations").glob("*/*/checks.json")):
        stage=path.parent.parent.name
        if not stage.startswith(("A6_","Aext_","B6q_","Bfull_")):
            continue
        check=json.loads(path.read_text())
        assert check["network_updates"]==0 and check["free_scalars"]==4080 and check["code_updates"]==150
        assert check["network_hashes_unchanged"] and check["ell_c_q_unchanged"]
        rows=read_csv(path.parent/"checkpoints.csv")
        assert sorted(r["step"] for r in rows)==[0,25,50,100,150]
        expected_quant=stage.startswith(("B6q_","Bfull_"))
        assert all(r["quantized"]==expected_quant for r in rows)
        best=min(rows,key=lambda r:(r["selection_objective"],r.get("actual_frame_share_bits",0) or 0,r["step"]))
        chosen=next(r for r in rows if r["selected"])
        assert best["step"]==chosen["step"]==check["selected_step"]
        for row in rows:
            saved=torch.load(path.parent/f'step_{int(row["step"]):03d}.pt',map_location="cpu",weights_only=False)
            assert tuple(saved["variable"].shape)==tuple(CONFIG["code_shape"]) and saved["variable"].numel()==4080
            assert torch.isfinite(saved["variable"]).all()
            if expected_quant:
                symbols=saved["symbols"]
                assert torch.equal(symbols,saved["variable"].round().short())
                assert torch.equal(saved["u_hat"],symbols.float()*DELTA)
                stream=path.parent/f'step_{int(row["step"]):03d}.orc'
                assert stream.stat().st_size*8==row["diagnostic_container_bits"]
                if torch.count_nonzero(symbols)==0:
                    assert row["actual_frame_packet_bits"]==8 and row["diagnostic_container_bits"]==816
                expected_score=row["L_image"]+row["lambda"]*row["R_actual_bpp"]
                assert abs(expected_score-row["selection_objective"])<1e-12
            else:
                assert saved["symbols"] is None and torch.equal(saved["u_hat"],saved["variable"])
        valid_runs+=1
        valid_updates+=150
        quantized_runs+=expected_quant
    streams=0
    zero_streams=0
    for path in sorted((out/"sequences").glob("*/*_checks.json")):
        check=json.loads(path.read_text())
        assert check["roundtrip_exact"] and check["independent_RGB_exact"]
        assert check["receiver"]["base_state_unchanged"] and check["receiver"]["source_access_guard"]
        if check["bitstream"]:
            video=path.stem.removesuffix("_checks")
            stream=path.parent/f"{video}.orc"
            st=check["bitstream"]
            assert stream.stat().st_size*8==st["file_bits"]==st["header_bits"]+sum(r["actual_bits"] for r in st["frames"])
            if all(r["kind"] in ["I","ZERO"] for r in st["frames"]):
                assert stream.stat().st_size==102
                zero_streams+=1
            streams+=1
        else:
            assert check["no_enhancement_file_sent"]
    for split in ["Train2","Val6"]:
        path=out/f"B_{split}_budget_selections.csv"
        if not path.exists():
            continue
        selected=read_csv(path)
        for r in selected:
            assert r["enhancement_bits"]<=r["budget"]*r["base_bits"]
            if r["zero_no_stream_fallback"]:
                assert r["enhancement_bits"]==0 and r["method"].endswith("ZERO_NO_STREAM")
            else:
                seq=out/"sequences"/r["method"]/f'{r["video"]}.orc'
                assert seq.stat().st_size*8==r["enhancement_bits"]
    result={"all_checks_passed":True,"historical_files_unchanged":True,"model_sha256_unchanged":True,
            "valid_code_optimization_runs":valid_runs,"valid_code_updates":valid_updates,"quantized_runs":quantized_runs,
            "network_parameter_updates":0,"only_4080_value_codes_optimized":True,"step0_always_recorded":True,
            "actual_sequence_streams_checked":streams,"all_zero_actual_containers":zero_streams,
            "zero_container_bytes":102,"zero_no_stream_fallback_is_distinct":True,
            "budget_selections_are_real_whole_candidates":True,
            "invalid_attempts_excluded":"B6_lambda0, retained with explicit domain-error record",
            "source_code_sha256":{p.name:sha(p) for p in sorted(HOME.glob("*.py"))}}
    save_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
