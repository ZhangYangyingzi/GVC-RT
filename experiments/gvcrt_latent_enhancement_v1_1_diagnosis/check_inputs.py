"""Read-only source identity checks against V1's fixed manifest and checkpoint audit."""
import json
from common import HERE,V1,ROOT,sha,save_json


def main():
    manifest=json.loads((HERE/"manifest.json").read_text())
    count=0
    for entry in manifest["train2"]+manifest["val6"]:
        for frame,expected in enumerate(entry["png_sha256"]):
            path=V1/"data"/entry["video_id"]/f"{frame:04d}.png"
            assert sha(path)==expected,f"Source frame mismatch: {path}"
            count+=1
    audit=json.loads((V1/"stage_a_v2/audit.json").read_text())
    for tag in ["I","P"]:
        assert sha(audit["weights"][tag]["path"])==audit["weights"][tag]["sha256"]
    save_json(HERE/"source_identity_checks.json",{"checked_source_frames":count,"all_match_v1_manifest":True,
              "original_codec_checkpoints_match_v1_audit":True,"checks_only_no_input_writes":True})
    print("Source identity verified",count)


if __name__=="__main__":
    main()
