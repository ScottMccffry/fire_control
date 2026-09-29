#!/usr/bin/env python3
"""
Scenario: a pre-established circular fire spreading from the centre, fought by a
swarm launched from 2 nests (which double as refuel bases).

Closed-loop on the fast mock fire (water drops directly suppress spread, so no
WRF restart cycling needed). Each drone:
  * launches from its nest, flies (continuous, real-ish speed) to the nearest
    active fire cell, and drops water while over fire,
  * when empty, flies back to the NEAREST nest, refuels, and returns.

Runs baseline (fire alone) vs swarm, reports burned area, and renders a video.

Usage:
    python scripts/nest_scenario.py --grid 160 --drones 600 --steps 200 \
        --out docs/figures/nest_scenario.mp4
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import MockFireSimulator  # noqa: E402

TANK = 10.0
DROP = 5.0           # litres delivered per step while over fire
REFILL_PER_STEP = 2.0
SPEED = 2.0          # cells per step


def make_fire(grid, init_cells, seed):
    np.random.seed(seed)
    sim = MockFireSimulator(grid_size=(grid, grid), cell_size_meters=30.0)
    sim.set_weather(wind_speed=4.0, wind_direction=45.0, temperature=30.0, humidity=0.2)
    rng = np.random.default_rng(seed)
    sim.fuel_moisture = rng.uniform(0.08, 0.18, (grid, grid))
    sim.fuel_load = rng.uniform(0.8, 1.0, (grid, grid))
    sim.base_spread_rate = 0.17
    # Circular burning patch at centre sized to ~init_cells.
    cy = cx = grid // 2
    R = (init_cells / np.pi) ** 0.5
    yy, xx = np.mgrid[0:grid, 0:grid]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= R * R
    sim.burned_area[mask] = 1
    sim.heat_intensity[mask] = sim.max_heat_intensity * 0.7
    return sim


def active_mask(sim):
    return sim.heat_intensity > sim.ignition_threshold * 0.3


def run(grid, n_drones, steps, nests, seed, use_drones,
        strategy="reactive", ring_radius=35.0, tanker=False, record=False):
    from scipy.spatial import cKDTree
    sim = make_fire(grid, 500, seed)
    frames, burned_curve = [], []
    nests = np.array(nests, dtype=float)
    if use_drones:
        pos = np.repeat(nests, n_drones // len(nests) + 1, axis=0)[:n_drones].astype(float)
        pos += np.random.default_rng(seed).uniform(-3, 3, pos.shape)
        pos = np.clip(pos, 0, grid - 1)
        water = np.full(n_drones, TANK)
        mode = np.zeros(n_drones, dtype=int)   # 0 outbound/fight, 1 return, 2 refill

    for t in range(steps):
        if use_drones:
            am = active_mask(sim)
            fr, fc = np.where(am)
            br, bc = np.where(sim.burned_area > 0)
            centroid = (np.array([br.mean(), bc.mean()]) if len(br)
                        else np.array([grid / 2, grid / 2]))
            ipos = np.clip(np.round(pos).astype(int), 0, grid - 1)

            # Refuel points: nests, plus a Canadair tanker circling the fire.
            refuel = nests
            if tanker:
                ang = 0.25 * t
                tpos = centroid + (ring_radius + 6) * np.array([np.sin(ang), np.cos(ang)])
                tpos = np.clip(tpos, 0, grid - 1)
                refuel = np.vstack([nests, tpos])

            # Outbound target depends on strategy.
            if strategy == "ring":
                # radial projection of each drone onto the containment ring
                v = pos - centroid
                n = np.linalg.norm(v, axis=1, keepdims=True)
                u = np.where(n > 1e-6, v / n, np.array([1.0, 0.0]))
                outbound_tgt = centroid + u * ring_radius
            else:  # reactive: nearest active fire cell
                if len(fr):
                    fire_xy = np.stack([fr, fc], axis=1).astype(float)
                    _, idx = cKDTree(fire_xy).query(pos, k=1)
                    outbound_tgt = fire_xy[idx]
                else:
                    outbound_tgt = pos.copy()

            dn = np.linalg.norm(pos[:, None, :] - refuel[None, :, :], axis=2)
            home = refuel[dn.argmin(axis=1)]
            mode[(mode == 0) & (water < DROP)] = 1
            tgt = np.where((mode == 0)[:, None], outbound_tgt, home)

            d = tgt - pos
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            pos = np.clip(pos + np.where(dist > 1e-6, d / dist, 0.0) * SPEED, 0, grid - 1)
            ipos = np.clip(np.round(pos).astype(int), 0, grid - 1)

            at_home = (mode == 1) & (np.linalg.norm(pos - home, axis=1) < 2.5)
            mode[at_home] = 2
            water[mode == 2] = np.minimum(TANK, water[mode == 2] + REFILL_PER_STEP)
            mode[(mode == 2) & (water >= TANK)] = 0

            fighting = mode == 0
            if strategy == "ring":
                # preventive soaking: drop while holding the ring (wets fuel ahead)
                rdist = np.linalg.norm(pos - centroid, axis=1)
                on = fighting & (np.abs(rdist - ring_radius) < 2.5) & (water > 0)
            else:
                on = fighting & am[ipos[:, 0], ipos[:, 1]] & (water > 0)
            for i in np.where(on)[0]:
                amt = min(water[i], DROP)
                sim.apply_water_drop(int(ipos[i, 0]), int(ipos[i, 1]), amt)
                water[i] -= amt
            if record:
                st = np.where(mode == 2, 4, np.where(mode == 1, 3, np.where(on, 2, 1)))
                tk = (refuel[-1] if tanker else None)
                frames.append((sim.heat_intensity.copy(), pos.copy(), st, tk))
        elif record:
            frames.append((sim.heat_intensity.copy(), None, None, None))
        sim.step()
        burned_curve.append(int(sim.burned_area.sum()))
    return int(sim.burned_area.sum()), burned_curve, frames, nests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=160)
    p.add_argument("--drones", type=int, default=600)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="docs/figures/nest_scenario.mp4")
    p.add_argument("--nest-frac", type=float, default=0.18,
                   help="nest distance from corner toward centre (0.18=corners, 0.5=centre)")
    p.add_argument("--strategy", choices=["reactive", "ring"], default="reactive",
                   help="reactive=chase fire; ring=preventive soaking of a containment ring")
    p.add_argument("--ring-radius", type=float, default=35.0, help="containment ring radius (cells)")
    p.add_argument("--tanker", action="store_true", help="Canadair tanker circling the fire as a mobile refuel point")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)

    g = args.grid
    f = args.nest_frac
    nests = [(int(g * f), int(g * f)), (int(g * (1 - f)), int(g * (1 - f)))]
    tot = g * g

    print("baseline (no drones) ...")
    b_burn, b_curve, _, _ = run(g, args.drones, args.steps, nests, args.seed, False)
    print(f"  baseline final burned: {b_burn} ({b_burn/tot:.0%} of grid)")

    print(f"swarm ({args.drones} drones, {len(nests)} nests, strategy={args.strategy}, "
          f"tanker={args.tanker}) ...")
    d_burn, d_curve, frames, nests_arr = run(g, args.drones, args.steps, nests,
                                             args.seed, True, strategy=args.strategy,
                                             ring_radius=args.ring_radius,
                                             tanker=args.tanker, record=True)
    print(f"  swarm final burned:    {d_burn} ({d_burn/tot:.0%} of grid)")
    print(f"  reduction: {b_burn - d_burn} cells ({(b_burn-d_burn)/b_burn:+.0%})")

    # Render
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    vmax = 1000.0
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=7, macro_block_size=None)
    colors = {1: ("white", "en route", 2), 2: ("lime", "dropping", 6),
              3: ("gold", "returning", 2), 4: ("deepskyblue", "refuelling", 4)}
    fig, ax = plt.subplots(figsize=(6, 6))
    for t, (heat, pos, st, tk) in enumerate(frames):
        ax.clear()
        ax.imshow(heat, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                  extent=[0, g, 0, g])
        if pos is not None:
            for s, (col, lab, sz) in colors.items():
                m = st == s
                if m.any():
                    ax.scatter(pos[m, 1], pos[m, 0], s=sz, c=col, alpha=0.7,
                               label=lab if t == 0 else None)
        ax.scatter(nests_arr[:, 1], nests_arr[:, 0], s=120, marker="*",
                   c="cyan", edgecolors="black", linewidths=0.6,
                   label="nest" if t == 0 else None)
        if tk is not None:
            ax.scatter([tk[1]], [tk[0]], s=180, marker="P", c="magenta",
                       edgecolors="white", linewidths=0.8,
                       label="tanker" if t == 0 else None)
        ax.set_title(f"Circular fire + swarm from 2 nests   step {t+1}/{len(frames)}\n"
                     f"burned {int((heat>0).sum())}  (baseline ends {b_burn}, "
                     f"swarm ends {d_burn})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.legend(loc="upper right", fontsize=6, markerscale=2, framealpha=0.6)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close(); plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
