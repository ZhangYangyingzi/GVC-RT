"""Evaluate all independent step-zero models against V3 on GPU7."""
import json
import subprocess
import sys
from report import ROOT, BRANCHES, read, write, METRICS, check_stream

for b in BRANCHES:
    subprocess.run([sys.executable,"-B",str(ROOT/"validate.py"),"--gpu","7","--stage",b,"--steps","0"],check=True)
weights = json.loads((ROOT/"initialization_weights_audit.json").read_text())
rows = [r for p in (ROOT/"parts").glob("checkpoint_validation_*.csv") for r in read(p)]
lookup = {(r["stage"],int(r["step"]),int(r["video_index"]),int(r["qp"])):r for r in rows}
audits = []
for b in BRANCHES:
    equal = weights[b]["state_equal_to_v3"]
    for v in range(6):
        for q in range(4):
            a, ref = lookup[b,0,v,q], lookup["v3",20000,v,q]
            check_stream(a); check_stream(ref)
            bits = a["real_bytes"] == ref["real_bytes"] and a["bitstream_sha256"] == ref["bitstream_sha256"]
            recon = a["reconstruction_sha256"] == ref["reconstruction_sha256"]
            metrics = all(float(a[m]) == float(ref[m]) for m in METRICS)
            assert all(equal.values()) and bits and recon and metrics
            audits.append({"branch":b,"video_index":v,"qp":q,**{k+"_equal":x for k,x in equal.items()},
                           "bitstream_equal":bits,"reconstruction_equal":recon,"metrics_equal":metrics,"status":"PASS"})
write("initialization_audit.csv",audits)
print("Initialization state, real bitstream, and decoded reconstruction: PASS (72 points)",flush=True)
