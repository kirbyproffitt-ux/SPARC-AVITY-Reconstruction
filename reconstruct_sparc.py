#!/usr/bin/env python3
"""
recovered_pt8_sparc_runner.py

Recovered PT-8 / AVITY SPARC runner.

What this script does
---------------------
1. Reads SPARC *_rotmod.dat galaxy files from sparc_database.zip
2. Runs the recovered frozen PT-8 BASE model

       V_model(r) = Ve * S(r; rt2) + Vc * B(r; rt1)

   where

       S(r; rt) = [1 + (rt / sqrt(r^2 + (eps*rt)^2))^alpha]^(-1/alpha)
       B(r; rt1) = S(r; rt1) * exp(-r / rt1)

3. Evaluates in log10 acceleration space

       g = V^2 / r
       RMS = sqrt(mean((log10(g_obs) - log10(g_mod))^2))

4. Optionally uses a canonical/full-sweep table for seeded or fixed parameters
5. Can do:
   - canonical parameter evaluation
   - deterministic Ve/Vc solve with fixed rt1/rt2
   - simple grid search over rt1/rt2 if you do not want to depend on a table
   - PNG overlay plot output

Important honesty note
----------------------
This script reproduces the recovered BASE/PT-8 behavior and is suitable for SPARC testing.
It does NOT claim to reproduce the final 143-row canonical export bit-for-bit in every branch,
because the original export involved joined baseline/sweep outputs and branch logic.

However, it is the correct saveable script for:
- raw SPARC sample testing
- BASE/Tier-1 style checks
- NGC7331-type validation
- batch sample testing

Examples
--------
1) Run one galaxy using canonical geometry from a table and solve Ve/Vc:
   python recovered_pt8_sparc_runner.py \
       --sparc-zip sparc_database.zip \
       --galaxy NGC7331 \
       --canonical Table1_AVITY_Canonical.csv \
       --mode solve_vevc \
       --plot

2) Evaluate one galaxy using the exact canonical row parameters:
   python recovered_pt8_sparc_runner.py \
       --sparc-zip sparc_database.zip \
       --galaxy NGC7331 \
       --canonical Table1_AVITY_Canonical.csv \
       --mode canonical \
       --plot

3) Run a quick raw SPARC search without any canonical table:
   python recovered_pt8_sparc_runner.py \
       --sparc-zip sparc_database.zip \
       --galaxy NGC7331 \
       --mode grid \
       --rt1-min 0.3 --rt1-max 8.0 \
       --rt2-min 1.0 --rt2-max 20.0 \
       --n-rt1 40 --n-rt2 60 \
       --plot

4) Batch a few galaxies:
   python recovered_pt8_sparc_runner.py \
       --sparc-zip sparc_database.zip \
       --galaxies NGC7331 UGC06973 NGC4214 \
       --canonical Table1_AVITY_Canonical.csv \
       --mode solve_vevc \
       --outdir out --plot
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class GalaxyData:
    name: str
    r: np.ndarray
    vobs: np.ndarray
    errv: np.ndarray
    vgas: np.ndarray
    vdisk: np.ndarray
    vbul: np.ndarray

    @property
    def positive_mask(self) -> np.ndarray:
        return (self.r > 0) & (self.vobs > 0)

    @property
    def gobs(self) -> np.ndarray:
        return (self.vobs ** 2) / self.r

    @property
    def loggobs(self) -> np.ndarray:
        mask = self.positive_mask
        out = np.full_like(self.r, np.nan, dtype=float)
        out[mask] = np.log10((self.vobs[mask] ** 2) / self.r[mask])
        return out


# -----------------------------
# IO helpers
# -----------------------------

def read_delimited_table(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            rows.append({str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
        return rows


def canonical_index(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_delimited_table(path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = row.get("galaxy") or row.get("Galaxy")
        if key:
            out[key] = row
    return out


def read_rotmod_from_zip(zf: zipfile.ZipFile, galaxy: str) -> GalaxyData:
    # Accept either top-level or sparc_database/ prefix
    candidates = [
        f"sparc_database/{galaxy}_rotmod.dat",
        f"{galaxy}_rotmod.dat",
    ]
    raw = None
    chosen = None
    for name in candidates:
        try:
            raw = zf.read(name).decode("utf-8", errors="replace")
            chosen = name
            break
        except KeyError:
            continue
    if raw is None:
        raise FileNotFoundError(f"Could not find {galaxy}_rotmod.dat in zip")

    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"No usable data rows in {chosen}")

    arr = np.loadtxt(io.StringIO("\n".join(lines)))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # SPARC rotmod columns:
    # r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    if arr.shape[1] < 6:
        raise ValueError(f"Unexpected SPARC rotmod shape for {galaxy}: {arr.shape}")

    return GalaxyData(
        name=galaxy,
        r=arr[:, 0].astype(float),
        vobs=arr[:, 1].astype(float),
        errv=arr[:, 2].astype(float),
        vgas=arr[:, 3].astype(float),
        vdisk=arr[:, 4].astype(float),
        vbul=arr[:, 5].astype(float),
    )


# -----------------------------
# PT-8 model
# -----------------------------

def S_term(r: np.ndarray, rt: float, alpha: float = 2.0, eps: float = 0.15) -> np.ndarray:
    denom = np.sqrt(r**2 + (eps * rt) ** 2)
    return (1.0 + (rt / denom) ** alpha) ** (-1.0 / alpha)


def B_term(r: np.ndarray, rt1: float, alpha: float = 2.0, eps: float = 0.15) -> np.ndarray:
    return S_term(r, rt1, alpha=alpha, eps=eps) * np.exp(-r / rt1)


def v_model(r: np.ndarray, ve: float, vc: float, rt1: float, rt2: float, alpha: float = 2.0, eps: float = 0.15) -> np.ndarray:
    return ve * S_term(r, rt2, alpha=alpha, eps=eps) + vc * B_term(r, rt1, alpha=alpha, eps=eps)


def logg_rms(r: np.ndarray, vobs: np.ndarray, vmod: np.ndarray) -> float:
    mask = (r > 0) & (vobs > 0) & (vmod > 0)
    if not np.any(mask):
        return float("nan")
    d = np.log10((vobs[mask] ** 2) / r[mask]) - np.log10((vmod[mask] ** 2) / r[mask])
    return float(np.sqrt(np.mean(d * d)))


# -----------------------------
# Solvers
# -----------------------------

def solve_ve_vc(
    data: GalaxyData,
    rt1: float,
    rt2: float,
    alpha: float = 2.0,
    eps: float = 0.15,
    ridge_rel_vc: float = 0.0,
) -> Tuple[float, float]:
    """
    Deterministic 2x2 normal-equation solve in velocity space.

    This is the closest clean recovered BASE solve:
    X = [S(rt2), B(rt1)] and solve for [Ve, Vc].
    """
    mask = data.positive_mask
    r = data.r[mask]
    y = data.vobs[mask]
    s = S_term(r, rt2, alpha=alpha, eps=eps)
    b = B_term(r, rt1, alpha=alpha, eps=eps)
    X = np.column_stack([s, b])

    A = X.T @ X
    rhs = X.T @ y

    # Relative ridge only on Vc diagonal term if requested
    if ridge_rel_vc != 0.0:
        A = A.copy()
        A[1, 1] = A[1, 1] * (1.0 + ridge_rel_vc)

    try:
        ve, vc = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        ve, vc = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(ve), float(vc)


def grid_search_rt1_rt2(
    data: GalaxyData,
    rt1_vals: np.ndarray,
    rt2_vals: np.ndarray,
    alpha: float = 2.0,
    eps: float = 0.15,
    ridge_rel_vc: float = 0.0,
) -> Dict[str, float]:
    """
    Search over rt1/rt2 and solve Ve/Vc at each point.
    Return best by log10(g) RMS.
    """
    best = {
        "rms": float("inf"),
        "ve": float("nan"),
        "vc": float("nan"),
        "rt1": float("nan"),
        "rt2": float("nan"),
        "alpha": alpha,
        "eps": eps,
    }

    for rt1 in rt1_vals:
        for rt2 in rt2_vals:
            if rt2 <= rt1:
                continue
            ve, vc = solve_ve_vc(data, rt1=rt1, rt2=rt2, alpha=alpha, eps=eps, ridge_rel_vc=ridge_rel_vc)
            vmod = v_model(data.r, ve, vc, rt1, rt2, alpha=alpha, eps=eps)
            rms = logg_rms(data.r, data.vobs, vmod)
            if np.isfinite(rms) and rms < best["rms"]:
                best.update({
                    "rms": rms,
                    "ve": ve,
                    "vc": vc,
                    "rt1": float(rt1),
                    "rt2": float(rt2),
                })
    return best


# -----------------------------
# Plotting
# -----------------------------

def baryons_total(vgas: np.ndarray, vdisk: np.ndarray, vbul: np.ndarray) -> np.ndarray:
    # SPARC-style baryonic total from components
    return np.sqrt(np.maximum(0.0, vgas * np.abs(vgas) + vdisk**2 + vbul**2))


def plot_galaxy(
    data: GalaxyData,
    outpath: Path,
    ve: float,
    vc: float,
    rt1: float,
    rt2: float,
    alpha: float,
    eps: float,
    title_extra: str = "",
) -> None:
    vbar = baryons_total(data.vgas, data.vdisk, data.vbul)
    vmod = v_model(data.r, ve, vc, rt1, rt2, alpha=alpha, eps=eps)
    rms = logg_rms(data.r, data.vobs, vmod)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.errorbar(data.r, data.vobs, yerr=data.errv, fmt="o", markersize=4, capsize=2, label="Observed")
    plt.plot(data.r, np.abs(data.vgas), label="Gas", linewidth=1.2)
    plt.plot(data.r, data.vdisk, label="Disk", linewidth=1.2)
    plt.plot(data.r, data.vbul, label="Bulge", linewidth=1.2)
    plt.plot(data.r, vbar, label="Baryons total", linewidth=2.0)
    plt.plot(data.r, vmod, label="PT-8 model", linewidth=2.4)

    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    title = f"{data.name}: PT-8 / AVITY"
    if title_extra:
        title += f" — {title_extra}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")

    box = (
        f"Ve={ve:.3f}\n"
        f"Vc={vc:.3f}\n"
        f"rt1={rt1:.5f}\n"
        f"rt2={rt2:.5f}\n"
        f"alpha={alpha:.3f}\n"
        f"eps={eps:.3f}\n"
        f"RMS(log g)={rms:.6f}"
    )
    plt.text(
        0.02, 0.98, box,
        transform=plt.gca().transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main runner
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Recovered PT-8 / AVITY SPARC runner")
    ap.add_argument("--sparc-zip", required=True, help="Path to sparc_database.zip")
    ap.add_argument("--canonical", help="Optional canonical/full-sweep table for seeding/fixed parameters")
    ap.add_argument("--galaxy", help="Single galaxy name")
    ap.add_argument("--galaxies", nargs="*", help="Multiple galaxy names")
    ap.add_argument("--mode", choices=["canonical", "solve_vevc", "grid"], default="solve_vevc")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--ridge-vc", type=float, default=0.0, help="Relative ridge on Vc normal-equation term")

    # Optional direct parameters if not using canonical
    ap.add_argument("--rt1", type=float, help="Direct rt1_kpc")
    ap.add_argument("--rt2", type=float, help="Direct rt2_kpc")
    ap.add_argument("--ve", type=float, help="Direct Ve_kms (canonical mode only)")
    ap.add_argument("--vc", type=float, help="Direct Vc_kms (canonical mode only)")

    # Grid options
    ap.add_argument("--rt1-min", type=float, default=0.3)
    ap.add_argument("--rt1-max", type=float, default=8.0)
    ap.add_argument("--rt2-min", type=float, default=1.0)
    ap.add_argument("--rt2-max", type=float, default=20.0)
    ap.add_argument("--n-rt1", type=int, default=35)
    ap.add_argument("--n-rt2", type=int, default=50)

    ap.add_argument("--plot", action="store_true", help="Write PNG plot(s)")
    ap.add_argument("--outdir", default="pt8_out", help="Output directory")
    return ap.parse_args(argv)


def get_targets(args: argparse.Namespace) -> List[str]:
    targets: List[str] = []
    if args.galaxy:
        targets.append(args.galaxy)
    if args.galaxies:
        targets.extend(args.galaxies)
    if not targets:
        raise SystemExit("Provide --galaxy or --galaxies")
    return targets


def get_row(idx: Dict[str, Dict[str, str]], galaxy: str) -> Optional[Dict[str, str]]:
    return idx.get(galaxy)


def ffloat(row: Dict[str, str], key: str, default: Optional[float] = None) -> float:
    val = row.get(key)
    if val is None or val == "":
        if default is None:
            raise KeyError(key)
        return float(default)
    return float(val)


def run_one(
    zf: zipfile.ZipFile,
    galaxy: str,
    idx: Dict[str, Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    data = read_rotmod_from_zip(zf, galaxy)
    row = get_row(idx, galaxy) if idx else None

    alpha = args.alpha
    eps = args.eps

    # Seed/fixed geometry from canonical if available
    rt1 = args.rt1
    rt2 = args.rt2
    if row is not None:
        if rt1 is None and "rt1_kpc" in row:
            rt1 = float(row["rt1_kpc"])
        if rt2 is None and "rt2_kpc" in row:
            rt2 = float(row["rt2_kpc"])
        if "alpha" in row and row["alpha"]:
            alpha = float(row["alpha"])

    if args.mode == "canonical":
        if row is None and (args.ve is None or args.vc is None or rt1 is None or rt2 is None):
            raise ValueError(f"{galaxy}: canonical mode requires canonical table or direct --ve --vc --rt1 --rt2")
        ve = args.ve if args.ve is not None else float(row["Ve_kms"])
        vc = args.vc if args.vc is not None else float(row["Vc_kms"])
        if rt1 is None or rt2 is None:
            raise ValueError(f"{galaxy}: rt1/rt2 missing")
        rms = logg_rms(data.r, data.vobs, v_model(data.r, ve, vc, rt1, rt2, alpha=alpha, eps=eps))
        result = {
            "galaxy": galaxy,
            "mode": "canonical",
            "ve": ve,
            "vc": vc,
            "rt1": rt1,
            "rt2": rt2,
            "alpha": alpha,
            "eps": eps,
            "rms": rms,
        }

    elif args.mode == "solve_vevc":
        if rt1 is None or rt2 is None:
            raise ValueError(f"{galaxy}: solve_vevc mode requires rt1/rt2 (from canonical or args)")
        ve, vc = solve_ve_vc(data, rt1=rt1, rt2=rt2, alpha=alpha, eps=eps, ridge_rel_vc=args.ridge_vc)
        rms = logg_rms(data.r, data.vobs, v_model(data.r, ve, vc, rt1, rt2, alpha=alpha, eps=eps))
        result = {
            "galaxy": galaxy,
            "mode": "solve_vevc",
            "ve": ve,
            "vc": vc,
            "rt1": rt1,
            "rt2": rt2,
            "alpha": alpha,
            "eps": eps,
            "rms": rms,
        }

    elif args.mode == "grid":
        rt1_vals = np.linspace(args.rt1_min, args.rt1_max, args.n_rt1)
        rt2_vals = np.linspace(args.rt2_min, args.rt2_max, args.n_rt2)
        best = grid_search_rt1_rt2(
            data,
            rt1_vals=rt1_vals,
            rt2_vals=rt2_vals,
            alpha=alpha,
            eps=eps,
            ridge_rel_vc=args.ridge_vc,
        )
        result = {
            "galaxy": galaxy,
            "mode": "grid",
            "ve": best["ve"],
            "vc": best["vc"],
            "rt1": best["rt1"],
            "rt2": best["rt2"],
            "alpha": best["alpha"],
            "eps": best["eps"],
            "rms": best["rms"],
        }

    else:
        raise ValueError(args.mode)

    if args.plot:
        outpath = Path(args.outdir) / f"{galaxy}_{args.mode}.png"
        plot_galaxy(
            data=data,
            outpath=outpath,
            ve=float(result["ve"]),
            vc=float(result["vc"]),
            rt1=float(result["rt1"]),
            rt2=float(result["rt2"]),
            alpha=float(result["alpha"]),
            eps=float(result["eps"]),
            title_extra=str(args.mode),
        )

    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    targets = get_targets(args)

    idx: Dict[str, Dict[str, str]] = {}
    if args.canonical:
        idx = canonical_index(Path(args.canonical))

    with zipfile.ZipFile(args.sparc_zip) as zf:
        results: List[Dict[str, object]] = []
        for galaxy in targets:
            try:
                res = run_one(zf, galaxy, idx, args)
                results.append(res)
            except Exception as exc:
                print(f"[error] {galaxy}: {exc}", file=sys.stderr)

    if not results:
        return 1

    # Print a compact report
    print("galaxy,mode,Ve_kms,Vc_kms,rt1_kpc,rt2_kpc,alpha,eps,rms_logg")
    for r in results:
        print(
            f"{r['galaxy']},{r['mode']},"
            f"{float(r['ve']):.6f},{float(r['vc']):.6f},"
            f"{float(r['rt1']):.6f},{float(r['rt2']):.6f},"
            f"{float(r['alpha']):.6f},{float(r['eps']):.6f},"
            f"{float(r['rms']):.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
