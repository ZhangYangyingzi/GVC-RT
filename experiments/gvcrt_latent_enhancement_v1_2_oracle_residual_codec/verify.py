"""Cross-file checks and a compact all-zero stream test, without retraining or oracle reruns."""
import argparse
import subprocess
from support import *
from run import inventory
from model import ResidualCodec
import bitstream


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results_v1")
    args=parser.parse_args()
    out=HERE/args.output
    configure()
    original=json.loads((out/"input_audit.json").read_text())
    assert inventory()==original["history_inventory"]
    for p,h in original["checkpoint_sha256"].items():
        assert sha(p)==h
    assert json.loads((out/"config.json").read_text())==CFG
    selections=json.loads((out/"teacher_selection.json").read_text())
    assert len(selections)==6 and all(s["eligible"] for s in selections)
    for s in selections:
        assert sha(s["source_delta_file"])==s["source_delta_sha256"]
        assert eligible(s["teacher"],s["baseline"])
    checked=0
    for path in sorted(out.glob("*/*_coding.json")):
        data=json.loads(path.read_text())
        video=path.stem.removesuffix("_coding")
        stream=path.parent/f"{video}.orc"
        assert data["bitstream"]["file_bits"]==stream.stat().st_size*8
        assert data["bitstream"]["header_bits"]==bitstream.HEADER.size*8
        assert data["bitstream"]["file_bits"]==data["bitstream"]["header_bits"]+sum(r["actual_bits"] for r in data["bitstream"]["frames"])
        assert data["symbol_roundtrip_exact"] and data["independent_rgb_exact"]
        assert data["receiver"]["source_access_guard"] and data["receiver"]["base_state_unchanged"]
        for f in data["bitstream"]["frames"]:
            if f["kind"]=="ZERO":
                assert f["payload_bits"]==0 and f["actual_bits"]==8
        checked+=1
    methods=0
    for path in sorted(out.glob("*/summary.json")):
        rows=json.loads(path.read_text())
        frame=read_csv(path.parent/"frames.csv")
        for r in rows:
            if r["video"]=="ALL":
                continue
            selected=[f for f in frame if f["video"]==r["video"] and (r["scope"]=="IP" or (f["type"]=="P" and (r["scope"]=="P" or f["target_available"])))]
            assert len(selected)==r["frames"]
            for key in ["psnr","ms_ssim","lpips"]:
                assert abs(np.mean([f[key] for f in selected])-r[key])<1e-12
            if r["diagnostic"]:
                assert r["enhancement_bits"] is None and r["total_kbps"] is None
            elif r["enhancement_bits"]:
                assert r["enhancement_bits"]==(path.parent/f'{r["video"]}.orc').stat().st_size*8
                assert abs(r["total_bpp"]-(r["base_bits"]+r["enhancement_bits"])/(16*1080*1920))<1e-12
        methods+=1
    # Normal receiver proves that actual all-zero flags return the fixed stage-D ZERO.
    cp=out/"stage_B/checkpoint_1500.pt"
    if not cp.exists():
        cp=out/"stage_A/checkpoint_1000.pt"
    ck=torch.load(cp,map_location="cpu",weights_only=False)
    entropy=FactorizedEntropy(8)
    entropy.load_state_dict({"log_scale":ck["model"]["entropy.log_scale"]},strict=True)
    fixture=out/"zero_compact_check"
    fixture.mkdir()
    video=manifest()["train2"][0]["video_id"]
    base=V1/"run_v1/base"/f"{video}_q0.bin"
    frames=[None]+[torch.zeros((1,8,17,30)) for _ in range(15)]
    delta=ck["deltas"][-1] if ck["deltas"] else 1.0
    level=len(ck["deltas"])-1 if ck["deltas"] else 0
    stats=bitstream.encode(fixture/"zero.orc",base,cp,1,level,delta,(8,17,30),frames,entropy)
    assert stats["file_bits"]==816 and all(r["payload_bits"]==0 for r in stats["frames"])
    restored=bitstream.decode(fixture/"zero.orc",base,cp,entropy)
    assert all(torch.count_nonzero(s)==0 for s in restored["frames"][1:])
    try:
        bitstream.checked_symbols(torch.full((1,8,17,30),128.0))
    except OverflowError:
        overflow_rejected=True
    else:
        raise AssertionError("Overflow accepted")
    subprocess.run([sys.executable,str(HERE/"receiver.py"),"--base",str(base),"--stream",str(fixture/"zero.orc"),
                    "--model",str(cp),"--output",str(fixture/"receiver.pt")],check=True)
    zero=torch.load(fixture/"receiver.pt",map_location="cpu",weights_only=False)
    assert zero["zero_exact_frames"]==list(range(1,16))
    assert zero["base_state_unchanged"] and zero["source_access_guard"]
    result={"all_checks_passed":True,"historical_files_unchanged":True,"original_checkpoints_unchanged":True,
            "six_teachers_rebased_and_eligible":True,"formal_stream_sequences_checked":checked,"metric_method_directories_checked":methods,
            "compact_all_zero_sequence_bytes":102,"zero_payload_bytes":0,"zero_increment_exact_P_frames":15,
            "overflow_rejected":overflow_rejected,"source_free_receiver":True,
            "configuration_sha256":sha(HERE/"config.json"),"calibration_sha256":sha(out/"calibration.json"),
            "source_code_sha256":{p.name:sha(p) for p in sorted(HERE.glob("*.py"))}}
    save_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
