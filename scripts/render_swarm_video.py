#!/usr/bin/env python3
"""
Render a video of a closed-loop run: real WRF-SFIRE fire + drone swarm on a
miniature map, one smooth animation across all coupling cycles.

It does NOT re-run WRF. It reads the per-cycle fire (FGRNHFX) from the saved
restart files and replays the (deterministic) controller over that same heat
sequence to reconstruct where every drone was each micro-step -- faithfully
reproducing the run -- then draws fire + drones (coloured by state) + lakes and
encodes an mp4.

Usage:
    python scripts/render_swarm_video.py --run-dir /opt/wrf_loop4/policy \
        --source policy --drones 3000 --out swarm_policy.mp4
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))


def rst_name(t_s):
    return f"wrfrst_d01_0001-01-01_00:{t_s//60:02d}:{t_s%60:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="/opt/wrf_loop4/policy")
    p.add_argument("--source", choices=["policy", "greedy"], default="policy")
    p.add_argument("--policy", default="agents/checkpoints/per_drone_ppo_large_g100_d40")
    p.add_argument("--drones", type=int, default=3000)
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--total", type=int, default=600)
    p.add_argument("--micro", type=int, default=40)
    p.add_argument("--show-cells", type=int, default=206, help="downsample heat to NxN for display")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default="swarm.mp4")
    args = p.parse_args()
    logging.disable(logging.CRITICAL)

    import netCDF4
    import warnings; warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from wrf_closed_loop import RealisticController, MICRO_STEPS

    rd = Path(args.run_dir)
    n_cycles = args.total // args.interval
    # Per-cycle fire heat (the field the controller acted on each cycle).
    heats = []
    for k in range(1, n_cycles):
        f = rd / rst_name(k * args.interval)
        if not f.exists():
            continue
        with netCDF4.Dataset(f) as ds:
            heats.append(ds.variables["FGRNHFX"][0].astype(float))
    grid = heats[0].shape[0]
    max_heat = max(1.0, max(h.max() for h in heats))
    print(f"loaded {len(heats)} cycle heat fields @ {grid}x{grid}, peak {max_heat/1000:.1f} kW/m^2")

    # Replay the deterministic controller to reconstruct drone trajectories.
    ctrl = RealisticController(args.source, args.policy, args.drones, grid, 35000.0)
    ctrl.record = True
    for h in heats:
        ctrl.step(h, args.micro)
    print(f"reconstructed {len(ctrl.frames)} micro-step frames; lakes={len(ctrl.lakes)}")

    # Downsample heat for display.
    ds_n = args.show_cells
    fac = max(1, grid // ds_n)
    def coarse(a):
        h2 = (a.shape[0] // fac) * fac
        return a[:h2, :h2].reshape(h2 // fac, fac, h2 // fac, fac).mean(axis=(1, 3))
    disp_heat = [coarse(h) for h in heats]
    lakes = ctrl.lakes / fac
    # Robust display ceiling (a few extreme cells would otherwise wash out the
    # fire); use the 98th percentile of active heat across frames.
    pos_vals = np.concatenate([h[h > 300] for h in heats if (h > 300).any()])
    vmax_disp = max(5.0, np.percentile(pos_vals, 98) / 1000.0)  # kW/m^2
    print(f"display vmax = {vmax_disp:.1f} kW/m^2")

    thr = 300.0
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=7, macro_block_size=None)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (pos, mode) in enumerate(ctrl.frames):
        cyc = i // args.micro
        H = disp_heat[min(cyc, len(disp_heat) - 1)]
        full_heat = heats[min(cyc, len(heats) - 1)]
        ax.clear()
        ax.imshow(H / 1000.0, origin="lower", cmap="inferno", vmin=0,
                  vmax=vmax_disp, extent=[0, ds_n, 0, ds_n])
        # drones, coloured by state
        d = pos / fac
        fight = mode == 0
        ref = mode == 1
        # is a fighting drone actually over active fire right now?
        ip = np.clip(np.round(pos).astype(int), 0, grid - 1)
        on_fire = fight & (full_heat[ip[:, 0], ip[:, 1]] > thr)
        ax.scatter(d[ref, 1], d[ref, 0], s=2, c="deepskyblue", alpha=0.5, label="refuelling")
        ax.scatter(d[fight & ~on_fire, 1], d[fight & ~on_fire, 0], s=2, c="white",
                   alpha=0.5, label="en route")
        ax.scatter(d[on_fire, 1], d[on_fire, 0], s=6, c="lime", label="dropping")
        ax.scatter(lakes[:, 1], lakes[:, 0], s=28, c="blue", marker="s",
                   edgecolors="white", linewidths=0.4, label="lakes")
        active = int((full_heat > thr).sum())
        ax.set_title(f"{args.source.upper()} swarm on real WRF-SFIRE fire\n"
                     f"cycle {cyc+1}/{len(heats)}  t={(cyc+1)*args.interval}s   "
                     f"active fire {active}  |  dropping {int(on_fire.sum())}  "
                     f"refuelling {int(ref.sum())}/{args.drones}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(loc="upper right", fontsize=6, markerscale=2, framealpha=0.6)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
