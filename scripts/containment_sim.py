#!/usr/bin/env python3
"""
Containment / asset-protection scenario.

Instead of attacking the fire, a "blocking" fleet lays a RETARDANT LINE between
the fire and key infrastructure that must not burn. In the mock fire model the
ignition probability of a cell scales with fuel_load*(1-moisture), so laying
retardant == driving fuel_load -> 0 in a band: the fire then physically cannot
cross it (a persistent firebreak), and the drones also drop foam to knock down
any heat that reaches the line.

The line is positioned from the predicted spread direction (fire->asset axis
blended with the wind vector) and oriented perpendicular to it, long enough to
cover the asset plus flanks so the fire can't go around.

Compares baseline (no fleet -> assets burn) vs containment fleet (assets saved)
and renders a video.

Usage:
  python scripts/containment_sim.py --out docs/figures/containment.mp4
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import MockFireSimulator          # noqa: E402
from scipy.spatial import cKDTree          # noqa: E402

CELL_M = 25.0
SUBSTEPS = 6
DRONE_MPS = 18.0
SUB_DT = 60.0 / SUBSTEPS
CELLS_PER_SUB = DRONE_MPS * SUB_DT / CELL_M
TANK = 24.0                 # retardant "units" a drone carries
LAY_COST = 1.0             # units to lay/maintain a metre of line per substep
SUPPRESS_COST = 2.0        # units to knock down heat reaching the line
LINE_HALF_WIDTH = 1        # retardant band half-thickness (cells)


def setup_scene(grid, seed):
    """Fire upwind (left), assets downwind (right), wind blowing toward them."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sim = MockFireSimulator(grid_size=(grid, grid), cell_size_meters=CELL_M)
    sim.fuel_moisture = rng.uniform(0.04, 0.10, (grid, grid))
    sim.fuel_load = rng.uniform(0.88, 1.0, (grid, grid))
    sim.base_spread_rate = 0.32
    sim.elevation[:] = 100.0   # flat: isolate the wind/containment effect

    # assets (row0,row1,col0,col1, name) on the downwind side
    ar = grid // 2
    assets = [
        (ar - 8, ar + 8, int(grid * 0.64), int(grid * 0.64) + 16, "town"),
        (ar - 26, ar - 18, int(grid * 0.66), int(grid * 0.66) + 7, "substation"),
        (ar + 20, ar + 30, int(grid * 0.68), int(grid * 0.68) + 8, "fuel depot"),
    ]
    asset_center = np.array([ar, grid * 0.66])

    # fire epicentre upwind
    F = np.array([ar + rng.uniform(-6, 6), grid * 0.24])
    yy, xx = np.mgrid[0:grid, 0:grid]
    disk = (yy - F[0]) ** 2 + (xx - F[1]) ** 2 <= 6.0 ** 2
    sim.burned_area[disk] = 1
    sim.heat_intensity[disk] = sim.max_heat_intensity * 0.75

    # wind: from fire toward assets, but a few degrees off-axis so the line
    # orientation visibly accounts for it.
    to_asset = asset_center - F
    base_ang = np.degrees(np.arctan2(to_asset[0], to_asset[1]))
    wind_ang = base_ang - 18.0
    sim.set_weather(wind_speed=16.0, wind_direction=float(wind_ang),
                    temperature=33.0, humidity=0.10)
    return sim, assets, asset_center, F, wind_ang


def asset_cells(assets, grid):
    m = np.zeros((grid, grid), bool)
    for r0, r1, c0, c1, _ in assets:
        m[r0:r1, c0:c1] = True
    return m


def defensive_line(F, asset_center, wind_ang, grid, n_pts, standoff=16, flank=None):
    """A retardant barrier between fire and assets, perpendicular to the blended
    fire->asset + wind spread direction, ANCHORED near the grid edges so the
    fire cannot flank around its ends. One sample point per drone -> contiguous,
    gap-free band when each drone lays a 3x3 patch."""
    if flank is None:
        flank = grid * 0.46                            # reach almost edge-to-edge
    u = asset_center - F
    u = u / (np.linalg.norm(u) + 1e-6)
    wvec = np.array([np.sin(np.radians(wind_ang)), np.cos(np.radians(wind_ang))])
    spread = u + 0.6 * wvec
    spread /= np.linalg.norm(spread) + 1e-6           # predicted attack axis
    perp = np.array([-spread[1], spread[0]])          # line runs perpendicular
    center = asset_center - spread * standoff          # standoff on the fire side
    ts = np.linspace(-flank, flank, n_pts)
    pts = center[None, :] + ts[:, None] * perp[None, :]
    return np.clip(pts, 1, grid - 2), perp


def run(grid, n_drones, ticks, seed, use_fleet=True, record=False):
    sim, assets, asset_center, F, wind_ang = setup_scene(grid, seed)
    amask = asset_cells(assets, grid)
    line_pts, perp = defensive_line(F, asset_center, wind_ang, grid, n_pts=n_drones)
    base = np.clip(asset_center + (asset_center - F) /
                   np.linalg.norm(asset_center - F) * 20, 1, grid - 2)  # depot behind assets

    rng = np.random.default_rng(seed + 7)
    if use_fleet:
        # one contiguous slot per drone -> a gap-free, edge-to-edge line
        target = line_pts.copy()
        pos = base[None, :] + rng.uniform(-2, 2, (n_drones, 2))
        tank = np.full(n_drones, TANK)
        mode = np.zeros(n_drones, int)        # 0 lay/hold, 1 return to refill
    treated = np.zeros((grid, grid), bool)

    frames, asset_burn_curve = [], []
    for t in range(ticks):
        if use_fleet:
            for _ in range(SUBSTEPS):
                mode[(mode == 0) & (tank <= 0)] = 1
                mode[(mode == 1) & (tank >= TANK - 1e-6)] = 0
                goal = np.where((mode == 1)[:, None], base[None, :], target)
                d = goal - pos
                nd = np.linalg.norm(d, axis=1, keepdims=True)
                stepmag = np.minimum(CELLS_PER_SUB, nd)
                pos = np.clip(pos + np.where(nd > 1e-6, d / nd, 0) * stepmag, 0, grid - 1)
                ip = np.round(pos).astype(int)
                at_line = (mode == 0) & (nd[:, 0] < 1.5) & (tank > 0)
                # lay retardant band: fuel_load -> 0 around the slot (skip cells
                # already treated so a drone doesn't waste its load re-laying)
                for i in np.where(at_line)[0]:
                    r, c = ip[i]
                    if treated[r, c]:
                        continue
                    rr = slice(max(0, r - LINE_HALF_WIDTH), r + LINE_HALF_WIDTH + 1)
                    cc = slice(max(0, c - LINE_HALF_WIDTH), c + LINE_HALF_WIDTH + 1)
                    sim.fuel_load[rr, cc] = 0.0
                    sim.fuel_moisture[rr, cc] = 1.0
                    treated[rr, cc] = True
                    tank[i] -= LAY_COST
                # refill at base
                at_base = (mode == 1) & (np.linalg.norm(pos - base, axis=1) < 1.5)
                tank[at_base] = TANK
            # foam knockdown: drones on the line suppress heat that reaches it
            thr = sim.ignition_threshold * 0.3
            on_line = (mode == 0) & (np.linalg.norm(target - pos, axis=1) < 2.0) & (tank > 0)
            for i in np.where(on_line)[0]:
                r, c = ip[i]
                rr = slice(max(0, r - 2), r + 3); cc = slice(max(0, c - 2), c + 3)
                if sim.heat_intensity[rr, cc].max() > thr:
                    sim.apply_water_drop(r, c, 12.0)
                    tank[i] -= SUPPRESS_COST

        if record and t % 2 == 0:
            p = pos.copy() if use_fleet else None
            md = mode.copy() if use_fleet else None
            frames.append((sim.heat_intensity.copy(), p, md, treated.copy(),
                           sim.burned_area.copy()))
        sim.step()
        asset_burn_curve.append(int((sim.burned_area & amask).sum()))

    asset_burned = int((sim.burned_area & amask).sum())
    total_burned = int(sim.burned_area.sum())
    return (asset_burned, total_burned, frames, assets, amask, line_pts,
            base, asset_burn_curve)


def render(frames, grid, assets, line_pts, base, out, fps, status):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import imageio.v2 as imageio
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7,
                                macro_block_size=None)
    fig, ax = plt.subplots(figsize=(7, 6.6))
    for t, (heat, pos, mode, treated, burned) in enumerate(frames):
        ax.clear()
        ax.imshow(heat, origin="lower", cmap="inferno", vmin=0, vmax=1000,
                  extent=[0, grid, 0, grid])
        # retardant line (treated cells)
        ty, tx = np.where(treated)
        if len(ty):
            ax.scatter(tx, ty, s=3, c="deepskyblue", alpha=0.5, marker="s")
        # assets: green if intact, red if any cell has burned
        for r0, r1, c0, c1, name in assets:
            hit = bool(burned[r0:r1, c0:c1].any())
            col = "red" if hit else "lime"
            ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                   ec=col, lw=2.0))
            ax.text((c0 + c1) / 2, r1 + 2, name, color=col, fontsize=7,
                    ha="center", fontweight="bold")
        if pos is not None:
            hold = mode == 0
            ax.scatter(pos[hold, 1], pos[hold, 0], s=6, c="white", alpha=0.8,
                       label="laying/holding line" if t == 0 else None)
            ax.scatter(pos[~hold, 1], pos[~hold, 0], s=6, c="orange", alpha=0.8,
                       label="returning to refill" if t == 0 else None)
        ax.scatter([base[1]], [base[0]], s=160, marker="*", c="yellow",
                   edgecolors="k", label="retardant depot" if t == 0 else None)
        ax.set_title(f"Containment fleet: hold a retardant line, protect assets   "
                     f"tick {t*2}\n{status}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.legend(loc="lower left", fontsize=6, markerscale=2, framealpha=0.7)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close(); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=240)
    p.add_argument("--drones", type=int, default=220)
    p.add_argument("--ticks", type=int, default=220)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--out", default="docs/figures/containment.mp4")
    p.add_argument("--fps", type=int, default=18)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    tot = args.grid ** 2

    print("baseline (no fleet) ...")
    ab0, tb0, _, assets, amask, _, _, _ = run(args.grid, args.drones, args.ticks,
                                              args.seed, use_fleet=False)
    na = int(amask.sum())
    print(f"  asset cells burned: {ab0}/{na} ({ab0/na:.0%})   total burned {tb0} ({tb0/tot:.0%})")

    print("containment fleet ...")
    ab1, tb1, frames, assets, amask, line_pts, base, curve = run(
        args.grid, args.drones, args.ticks, args.seed, use_fleet=True, record=True)
    print(f"  asset cells burned: {ab1}/{na} ({ab1/na:.0%})   total burned {tb1} ({tb1/tot:.0%})")
    saved = "ASSETS SAVED" if ab1 == 0 else f"assets {ab1}/{na} hit"
    status = (f"baseline: assets {ab0}/{na} burned, {tb0/tot:.0%} grid  |  "
              f"fleet: {saved}, {tb1/tot:.0%} grid")
    render(frames, args.grid, assets, line_pts, base, args.out, args.fps, status)
    print(f"  {saved}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
