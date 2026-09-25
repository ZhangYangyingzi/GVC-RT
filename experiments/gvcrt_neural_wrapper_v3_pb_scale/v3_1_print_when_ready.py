#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "v3_1_final_integrity.json").is_file():
    time.sleep(30)
with open(ROOT / "v3_1_stdout.log", "w") as output:
    subprocess.run([sys.executable, str(ROOT / "v3_1_print_status.py")],
                   stdout=output, stderr=subprocess.STDOUT, check=True)
