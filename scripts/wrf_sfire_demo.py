#!/usr/bin/env python3
"""
Demo: drive the project's FireState pipeline from real WRF-SFIRE output.

Runs WRFReplaySimulator over a directory of wrfout history files (produced by a
real WRF-SFIRE run), prints the fire-spread progression as seen through the
project's FlameFont/FireState objects, and saves a figure of the flame front.

Usage:
    python scripts/wrf_sfire_demo.py --wrfout-dir /opt/wrf_run --out wrf_sfire_fire.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.wrf_replay import WRFReplaySimulator  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wrfout-dir", default="/opt/wrf_run")
    p.add_argument("--out", default="wrf_sfire_fire.png")
    args = p.parse_args()

    # hill_simple ideal case: atmos dx=50 m, fire refinement sr=4 -> 12.5 m mesh.
    sim = WRFReplaySimulator(args.wrfout_dir, fire_mesh_res=12.5)
    print(f"Found {len(sim)} WRF-SFIRE frames in {args.wrfout_dir}\n")
    print(f"{'frame':>5} {'burned cells':>13} {'peak kW/m^2':>12} {'perimeter pts':>14}")

    frames = []
    fs = sim.step()
    while fs is not None:
        ff = fs.current_flame_front
        burned = int(ff.burned_area.sum())
        peak = float(ff.heat_intensity.max()) / 1000.0
        print(f"{ff.time_step:>5} {burned:>13} {peak:>12.2f} {len(ff.fire_perimeter):>14}")
        frames.append(ff)
        fs = sim.step()

    last = frames[-1]
    print(f"\nFinal frame -> FireState/FlameFont bridged from real WRF-SFIRE:")
    print(f"  grid: {last.heat_intensity.shape}  resolution ~{last.grid_resolution:.1f} m/cell")
    print(f"  burned area: {last.get_fire_size():,.0f} m^2")
    print(f"  peak ground heat flux: {last.heat_intensity.max()/1000:.2f} kW/m^2")

    # Visualization: heat flux for the first few frames.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = frames[1:] if len(frames) > 1 else frames
        n = len(show)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.3))
        if n == 1:
            axes = [axes]
        vmax = max(f.heat_intensity.max() for f in show) / 1000.0
        for ax, ff in zip(axes, show):
            im = ax.imshow(ff.heat_intensity / 1000.0, origin="lower",
                           cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"t = {ff.time_step*2} min\nburned {int(ff.burned_area.sum())} cells")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Real WRF-SFIRE fire spread (ground heat flux, kW/m^2) "
                     "bridged into the drone-swarm FireState", fontsize=11)
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("kW/m^2")
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\nSaved figure: {args.out}")
    except Exception as e:
        print(f"\n(visualization skipped: {e})")


if __name__ == "__main__":
    main()
