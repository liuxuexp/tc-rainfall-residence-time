#!/usr/bin/env python3
"""Run the numbered pipeline in order (01-17)."""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = sorted(p for p in HERE.glob("[0-9][0-9]_*.py")
                 if p.name[0] != "0" or p.name[1] != "0")   # skip 00 itself

def main():
    failed = []
    for s in SCRIPTS:
        t = time.time()
        print(f"[run] {s.name}", flush=True)
        r = subprocess.run([sys.executable, str(s)])
        if r.returncode != 0:
            failed.append(s.name)
        print(f"      {time.time()-t:.1f}s", flush=True)
    if failed:
        print("FAILED: " + ", ".join(failed)); sys.exit(1)
    print("pipeline done -> figures/, report/data/")

if __name__ == "__main__":
    main()
