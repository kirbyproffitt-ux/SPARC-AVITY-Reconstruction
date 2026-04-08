#!/usr/bin/env python3
"""
avity_plot_generator.py

Generate AVITY / PT-8 velocity-domain plots from the canonical table and SPARC
rotmod files, matching the documented figure content used in earlier AVITY
plot batches.

What it plots
-------------
For each galaxy:
- Gas
- Disk
- Bulge
- Baryons total = sqrt(vgas^2 + vdisk^2 + vbul^2)
- AVITY frozen operator model
- Observed velocity with error bars

Annotation box:
- Tier
- RMS
- Ve
- Vc
- rt1
- rt2

Usage
-----
Single galaxy:
python avity_plot_generator.py \
  --sparc-zip sparc_database.zip \
  --canonical Table1_AVITY_Canonical.csv \
  --galaxy UGC04499 \
  --outdir plots

Batch:
python avity_plot_generator.py \
  --sparc-zip sparc_database.zip \
  --canonical Table1_AVITY_Canonical.csv \
  --galaxies UGC04499 UGC05918 UGC06667 \
  --outdir plots

From file:
python avity_plot_generator.py \
  --sparc-zip sparc_database.zip \
  --canonical Table1_AVITY_Canonical.csv \
  --galaxy-list galaxies.txt \
  --outdir plots
"""

from __future__ import annotations

import argparse
import csv
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


EPSILON = 0.15
DEFAULT_DPI = 160


@dataclass
class CanonicalRow:
    galaxy: str
    tier: int
    op: str
    op_group: str
    rms_base: float
    rms_final: float
    Ve_kms: float
    Vc_kms: float
    alpha: float
    dInc: float
    rt1_kpc: float
    rt2_kpc: float


@dataclass
class GalaxyData:
    galaxy: str
    radius_kpc: List[float]
    vobs_kms: List[float]
    errv_kms: List[float]
    vgas_kms: List[float]
    vdisk_kms: List[float]
    vbul_kms: List[float]


def try_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if value == "":
        return None
    return float(value)


def read_tabular_text(path: Path) -> List[Dict[str, str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        return []
    delimiter = "\t" if "\t" in lines[0] else ","
    return list(csv.DictReader(lines, delimiter=delimiter))


def load_canonical_table(path: Path) -> Dict[str, CanonicalRow]:
    rows = read_tabular_text(path)
    out: Dict[str, CanonicalRow] = {}
    for row in rows:
        rec = CanonicalRow(
            galaxy=row["galaxy"].strip(),
            tier=int(float(row["tier"])),
            op=row["op"].strip(),
            op_group=row["op_group"].strip(),
            rms_base=float(row["rms_base"]),
            rms_final=float(row["rms_final"]),
            Ve_kms=float(row["Ve_kms"]),
            Vc_kms=float(row["Vc_kms"]),
            alpha=float(row["alpha"]),
            dInc=float(row["dInc"]),
            rt1_kpc=float(row["rt1_kpc"]),
            rt2_kpc=float(row["rt2_kpc"]),
        )
        out[rec.galaxy] = rec
    return out


def read_sparc_rotmod_from_zip(zf: zipfile.ZipFile, galaxy: str) -> GalaxyData:
    member = f"sparc_database/{galaxy}_rotmod.dat"
    raw = zf.read(member).decode("utf-8", errors="replace").splitlines()

    radius_kpc: List[float] = []
    vobs_kms: List[float] = []
    errv_kms: List[float] = []
    vgas_kms: List[float] = []
    vdisk_kms: List[float] = []
    vbul_kms: List[float] = []

    for ln in raw:
        line = ln.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        radius_kpc.append(float(parts[0]))
        vobs_kms.append(float(parts[1]))
        errv_kms.append(float(parts[2]))
        vgas_kms.append(float(parts[3]))
        vdisk_kms.append(float(parts[4]))
        vbul_kms.append(float(parts[5]))

    if not radius_kpc:
        raise ValueError(f"No data rows found in {member}")

    return GalaxyData(
        galaxy=galaxy,
        radius_kpc=radius_kpc,
        vobs_kms=vobs_kms,
        errv_kms=errv_kms,
        vgas_kms=vgas_kms,
        vdisk_kms=vdisk_kms,
        vbul_kms=vbul_kms,
    )


def saturation_operator(r: float, rt: float, alpha: float, eps: float = EPSILON) -> float:
    if rt <= 0:
        return 0.0
    reff = math.sqrt(r * r + (eps * rt) * (eps * rt))
    return (1.0 + (rt / reff) ** alpha) ** (-1.0 / alpha)


def bump_operator(r: float, rt1: float, alpha: float, eps: float = EPSILON) -> float:
    if rt1 <= 0:
        return 0.0
    return saturation_operator(r, rt1, alpha, eps=eps) * math.exp(-r / rt1)


def model_velocity(radius_kpc: List[float], Ve_kms: float, Vc_kms: float,
                   rt1_kpc: float, rt2_kpc: float, alpha: float,
                   eps: float = EPSILON) -> List[float]:
    out: List[float] = []
    for r in radius_kpc:
        s = saturation_operator(r, rt2_kpc, alpha, eps=eps)
        b = bump_operator(r, rt1_kpc, alpha, eps=eps)
        out.append(Ve_kms * s + Vc_kms * b)
    return out


def baryons_total(vgas: List[float], vdisk: List[float], vbul: List[float]) -> List[float]:
    out: List[float] = []
    for a, b, c in zip(vgas, vdisk, vbul):
        out.append(math.sqrt(max(0.0, a * a + b * b + c * c)))
    return out


def format_value(x: float) -> str:
    return f"{x:.2f}"


def plot_one(galaxy: str, row: CanonicalRow, data: GalaxyData, outdir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    bary = baryons_total(data.vgas_kms, data.vdisk_kms, data.vbul_kms)
    avity = model_velocity(
        data.radius_kpc,
        row.Ve_kms,
        row.Vc_kms,
        row.rt1_kpc,
        row.rt2_kpc,
        row.alpha,
        eps=EPSILON,
    )

    fig, ax = plt.subplots(figsize=(10, 6.6666667), dpi=dpi)

    # Keep order aligned with the earlier plots:
    ax.plot(data.radius_kpc, data.vgas_kms, label="Gas")
    ax.plot(data.radius_kpc, data.vdisk_kms, label="Disk")
    ax.plot(data.radius_kpc, data.vbul_kms, label="Bulge")
    ax.plot(data.radius_kpc, bary, linewidth=2.0, label="Baryons total")
    ax.plot(data.radius_kpc, avity, linewidth=2.2, label="AVITY frozen operator")
    ax.errorbar(
        data.radius_kpc,
        data.vobs_kms,
        yerr=data.errv_kms,
        fmt="o",
        capsize=3,
        label="Observed",
    )

    ax.set_title(f"{galaxy}: Observed vs Baryons + AVITY")
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(True, alpha=0.35)

    text_box = "\n".join([
        f"Tier={row.tier}",
        f"RMS={row.rms_final:.6f}",
        f"Ve={format_value(row.Ve_kms)}",
        f"Vc={format_value(row.Vc_kms)}",
        f"rt1={format_value(row.rt1_kpc)}",
        f"rt2={format_value(row.rt2_kpc)}",
    ])
    ax.text(
        0.02, 0.98, text_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.95),
    )

    ax.legend(loc="lower right")
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{galaxy}.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def load_galaxy_names(args: argparse.Namespace) -> List[str]:
    names: List[str] = []
    if args.galaxy:
        names.append(args.galaxy)
    if args.galaxies:
        names.extend(args.galaxies)
    if args.galaxy_list:
        lines = Path(args.galaxy_list).read_text(encoding="utf-8", errors="replace").splitlines()
        names.extend([ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")])

    # Preserve order, remove duplicates
    seen = set()
    unique: List[str] = []
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate AVITY velocity-domain plots.")
    ap.add_argument("--sparc-zip", required=True, help="Path to sparc_database.zip")
    ap.add_argument("--canonical", required=True, help="Path to Table1_AVITY_Canonical.csv")
    ap.add_argument("--galaxy", help="Single galaxy name")
    ap.add_argument("--galaxies", nargs="*", help="Multiple galaxy names")
    ap.add_argument("--galaxy-list", help="Text file with one galaxy name per line")
    ap.add_argument("--outdir", default="plots", help="Output directory")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG DPI")
    args = ap.parse_args()

    names = load_galaxy_names(args)
    if not names:
        raise SystemExit("ERROR: provide --galaxy, --galaxies, or --galaxy-list")

    canonical = load_canonical_table(Path(args.canonical))
    outdir = Path(args.outdir)

    with zipfile.ZipFile(args.sparc_zip, "r") as zf:
        for galaxy in names:
            if galaxy not in canonical:
                raise SystemExit(f"ERROR: galaxy not found in canonical table: {galaxy}")
            data = read_sparc_rotmod_from_zip(zf, galaxy)
            outpath = plot_one(galaxy, canonical[galaxy], data, outdir, args.dpi)
            print(f"Wrote {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
