#!/usr/bin/env python3
"""run_pipeline.py – v1.2 (2025‑06‑13)
=====================================
End‑to‑end helper that
  1. flattens a SystemVerilog design with **Yosys**; and
  2. feeds the flattened netlist to your **generate_dfg.py**
     to emit a data‑flow graph.

Fixes & Features (v1.2)
-----------------------
* **Strip attributes** (`-noattr`) so PyVerilog is not confused by `(* … *)`.
* **Detect the *real* top‑module after flatten** (handles escaped names like
  `\Trojan0` that Yosys may emit) and pass that exact name to PyVerilog.
* Clear, colour‑free `[INFO] / [ERROR]` prints and tail‑of‑log dumps.

Usage
-----
```bash
python run_pipeline.py design.v
# optional flags
#   -y PATH   custom flatten template (defaults to auto‑generated)
#   -g PATH   path to generate_dfg.py (default ./generate_dfg.py)
#   -o NAME   output prefix (default basename without .v)
#   -k        keep the temp .ys script instead of auto‑deleting
```
Outputs (relative to cwd):
  • `flattened_<prefix>.v` – flattened netlist
  • `<prefix>_dfg.txt`    – textual DFG produced by generate_dfg.py

"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

# ----------------------- helpers -----------------------

def die(msg: str, rc: int = 1):
    print(f"[ERROR] {msg}")
    sys.exit(rc)


def log(msg: str):
    print(f"[INFO] {msg}")


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def detect_module_name(verilog_path: Path) -> str | None:
    """Return the *first* module identifier in a Verilog file, including any
    leading backslash (escaped identifier)."""
    module_re = re.compile(r"^\s*module\s+([A-Za-z_\\][A-Za-z0-9_\\$]*)")
    with verilog_path.open() as f:
        for line in f:
            m = module_re.match(line)
            if m:
                return m.group(1)
    return None


def tail(text: str, n: int = 20) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join(lines[-n:])


# ----------------------- main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Flatten Verilog then build DFG")
    parser.add_argument("verilog", help="Input SystemVerilog design (*.v)")
    parser.add_argument("-y", "--yosys-script", dest="ys", help="Flatten template (.ys)")
    parser.add_argument("-g", "--generate-dfg", dest="gpy", default="generate_dfg.py",
                        help="Path to generate_dfg.py (default: ./generate_dfg.py)")
    parser.add_argument("-o", "--output-prefix", dest="prefix",
                        help="Prefix for output files (default: input basename)")
    parser.add_argument("-k", "--keep-temp", action="store_true",
                        help="Keep the temporary .ys script for debugging")

    args = parser.parse_args()

    in_path = Path(args.verilog).resolve()
    if not in_path.exists():
        die(f"Input file not found: {in_path}")

    if which("yosys") is None:
        die("Yosys not found in $PATH – please install it first")

    gpy_path = Path(args.gpy).resolve()
    if not gpy_path.exists():
        die(f"generate_dfg.py not found: {gpy_path}")

    prefix = args.prefix or in_path.stem
    flat_path = Path(f"flattened_{prefix}.v").resolve()
    dfg_path = Path(f"{prefix}_dfg.txt").resolve()

    # --------------------------------------------------
    # 1. Detect *original* top‑module (first in source)
    # --------------------------------------------------
    orig_top = detect_module_name(in_path)
    if orig_top is None:
        die("Could not detect any module in input Verilog")
    log(f"Detected top‑level module in source: {orig_top}")

    # --------------------------------------------------
    # 2. Build a temporary Yosys script
    # --------------------------------------------------
    if args.ys:
        # patch the user‑supplied template
        tpl = Path(args.ys).read_text()
        patched = tpl.replace("<FILENAME>", str(in_path)) \
                     .replace("<TOP>", orig_top) \
                     .replace("<OUT>", str(flat_path))
    else:
        # minimal auto‑generated script
        patched = dedent(f"""
        read_verilog -sv {in_path}
        hierarchy -check -top {orig_top}
        flatten
        write_verilog -noattr {flat_path}
        """)

    tmp_ys = Path(tempfile.mkstemp(prefix="tmp_flatten_", suffix=".ys")[1])
    tmp_ys.write_text(patched)

    # --------------------------------------------------
    # 3. Run Yosys
    # --------------------------------------------------
    log(f"[Yosys] flattening via script: {tmp_ys}")
    yc = subprocess.run([
        "yosys", "-s", str(tmp_ys)
    ], text=True, capture_output=True)

    if yc.returncode != 0:
        die("Yosys failed!\n" + tail(yc.stderr))

    if not args.keep_temp:
        tmp_ys.unlink(missing_ok=True)

    if not flat_path.exists():
        die("Expected output not found: " + str(flat_path))

    # --------------------------------------------------
    # 4. Detect *actual* top‑module in the flattened file
    # --------------------------------------------------
    flat_top = detect_module_name(flat_path)
    if flat_top is None:
        die("No module found in flattened netlist – aborting")
    log(f"Detected top‑level module in flattened netlist: {flat_top}")

    # --------------------------------------------------
    # 5. Run generate_dfg.py
    # --------------------------------------------------
    log(f"[PyVerilog] Generating DFG → {dfg_path}")
    gc = subprocess.run([
        sys.executable, str(gpy_path), str(flat_path), flat_top
    ], text=True, capture_output=True)

    if gc.returncode != 0:
        err_tail = tail(gc.stderr)
        die("generate_dfg.py failed!\n----- STDERR (tail) -----\n" + err_tail)

    dfg_path.write_text(gc.stdout)
    log("Pipeline finished ✔︎")
    log(f"Flattened netlist : {flat_path}")
    log(f"DFG (text)       : {dfg_path}")


if __name__ == "__main__":
    main()
