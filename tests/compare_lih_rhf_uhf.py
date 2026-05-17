#!/usr/bin/env python3
"""Compare real LiH RHF and spin-preserved UHF XCholemodel outputs.

The script runs holemodel.py in temporary directories so validation never
overwrites production XCholemodel_* files in the DFT method folders.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DFT_ROOT = Path("/pscratch/sd/l/lhou/Workdir/LiH_xchole/FCI/ccpcvtz_DFT")
DATASETS = ("LDA_X", "LDA_C", "LDA_XC", "PBE_X", "PBE_C", "PBE_XC")


def load_holemodel():
    spec = importlib.util.spec_from_file_location("holemodel_compare", REPO_ROOT / "holemodel.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_model(module, input_plot: Path, work_root: Path, label: str) -> Path:
    work_dir = work_root / label
    work_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        module.DFThxcmodel(str(input_plot))
    finally:
        os.chdir(cwd)
    output = work_dir / f"XCholemodel_{input_plot.stem}.plot"
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(f"missing model output: {output}")
    return output


def compare_pair(r_plot: Path, u_plot: Path, tol: float) -> list[tuple[str, float]]:
    diffs: list[tuple[str, float]] = []
    with h5py.File(r_plot, "r") as rhf, h5py.File(u_plot, "r") as uhf:
        for dataset in DATASETS:
            r_values = np.asarray(rhf[dataset])
            u_values = np.asarray(uhf[dataset])
            diff = float(np.max(np.abs(r_values - u_values)))
            diffs.append((dataset, diff))
            if diff > tol:
                raise AssertionError(f"{dataset} max_abs={diff:.6e} > tol={tol:.6e}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dft-root", type=Path, default=DEFAULT_DFT_ROOT)
    parser.add_argument("--stems", nargs="+", default=("LiH_1p598", "LiH_1p599", "LiH_1p6"))
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--keep-work", type=Path, default=None)
    args = parser.parse_args()

    module = load_holemodel()
    os.environ.setdefault("XCHOLEMODEL_RESTRICTED_CLOSED_SHELL", "auto")

    with tempfile.TemporaryDirectory(prefix="xcholemodel_lih_rhf_uhf_") as tmpdir:
        work_root = args.keep_work or Path(tmpdir)
        print(f"LiH RHF/UHF comparison work_root={work_root}")
        for stem in args.stems:
            r_input = args.dft_root / "r_hf" / f"{stem}.plot"
            u_input = args.dft_root / "u_hf" / f"{stem}.plot"
            if not r_input.exists() or not u_input.exists():
                raise FileNotFoundError(f"missing RHF/UHF input plot for {stem}")
            r_output = run_model(module, r_input, work_root, f"r_hf_{stem}")
            u_output = run_model(module, u_input, work_root, f"u_hf_{stem}")
            diffs = compare_pair(r_output, u_output, args.tol)
            summary = " ".join(f"{name}={diff:.3e}" for name, diff in diffs)
            print(f"PASS {stem} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
