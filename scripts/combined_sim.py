#!/usr/bin/env python3
"""
Combined scenario: attack + defense on one map.

  * A dense central "foyer" (hot, fuel-rich) that spreads radially outward.
  * 8 EXTINGUISHER trucks: each launches attack drones that fly to the nearest
    active fire cell, drop water, and refuel at their truck.
  * 2 DEFENDER trucks: launch drones that lay closed RETARDANT RINGS
    (fuel_load -> 0, so the fire cannot cross) around critical infrastructure
    placed randomly on the map, and refuel at the defender trucks.
  * Critical infrastructure must not burn.

Compares baseline (no fleet) vs the combined fleet and renders a video.

Usage:
  python scripts/combined_sim.py --out docs/figures/combined.mp4
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
TANK = 10.0                 # attack drone water (L)
DTANK = 24.0                # defender retardant units
DROP_RATE_LPS = 5.0
DRONE_MPS = 16.0
SUBSTEPS = 4
SUB_DT = 60.0 / SUBSTEPS
CELLS_PER_SUB = DRONE_MPS * SUB_DT / CELL_M
DROP_PER_SUB = DROP_RATE_LPS * SUB_DT
TRUCK_RESERVE = 20000.0
TRUCK_DISPENSE_LPM = 200.0
TRUCK_TOPUP_LPM = 100.0
SAFE_M = 1000.0
RETREAT_M = 200.0
DROP_EFF = 0.7
LINE_HALF_WIDTH = 1
LAY_COST = 1.0
# fire regime (overridable from the CLI)
WIND = 4.0
SPREAD = 0.34
FOYER_R = 11.0
MOIST = (0.05, 0.12)


def setup(grid, n_assets, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sim = MockFireSimulator(grid_size=(grid, grid), cell_size_meters=CELL_M)
    sim.fuel_moisture = rng.uniform(MOIST[0], MOIST[1], (grid, grid))
    sim.fuel_load = rng.uniform(0.9, 1.0, (grid, grid))
    sim.base_spread_rate = SPREAD
    sim.elevation[:] = 100.0
    sim.set_weather(wind_speed=WIND, wind_direction=float(rng.uniform(0, 360)),
                    temperature=36.0, humidity=0.08)
    # dense central foyer
    c = np.array([grid / 2, grid / 2])
    yy, xx = np.mgrid[0:grid, 0:grid]
    disk = (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= FOYER_R ** 2
    sim.burned_area[disk] = 1
    sim.heat_intensity[disk] = sim.max_heat_intensity

    # critical infrastructure: random, ring-shaped band around the foyer
    assets = []
    names = ["town", "substation", "hospital", "fuel depot", "data center",
             "water plant", "school", "comms tower"]
    tries = 0
    while len(assets) < n_assets and tries < 2000:
        tries += 1
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.16, 0.34) * grid
        ctr = c + rad * np.array([np.sin(ang), np.cos(ang)])
        hh, hw = int(rng.integers(4, 9)), int(rng.integers(4, 9))
        r0, r1 = int(ctr[0] - hh), int(ctr[0] + hh)
        c0, c1 = int(ctr[1] - hw), int(ctr[1] + hw)
        if r0 < 6 or c0 < 6 or r1 > grid - 6 or c1 > grid - 6:
            continue
        if assets and min(np.linalg.norm(ctr - np.array([a[5] for a in assets]), axis=1)) < grid * 0.12:
            continue
        assets.append((r0, r1, c0, c1, names[len(assets) % len(names)], ctr))
    return sim, c, assets


def asset_mask(assets, grid):
    m = np.zeros((grid, grid), bool)
    for r0, r1, c0, c1, *_ in assets:
        m[r0:r1, c0:c1] = True
    return m


def asset_rings(assets, foyer, grid, standoff=6):
    """A closed retardant ring around each asset (fire can come from any side as
    the radial fire wraps around it)."""
    pts = []
    for r0, r1, c0, c1, name, ctr in assets:
        R = max(r1 - r0, c1 - c0) / 2 + standoff
        # space patches ~1.8 cells apart; the 3-cell-wide bands still overlap so
        # the ring is gap-free, while using far fewer drones than 1-cell spacing.
        n = max(16, int(2 * np.pi * R / 1.8))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = ctr[None, :] + R * np.stack([np.sin(th), np.cos(th)], 1)
        pts.append(np.clip(ring, 1, grid - 2))
    return np.vstack(pts)


def place_trucks(grid, n, foyer, seed):
    rng = np.random.default_rng(seed + 1)
    safe = SAFE_M / CELL_M
    trucks, tries = [], 0
    while len(trucks) < n and tries < 5000:
        tries += 1
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.30, 0.46) * grid
        p = foyer + rad * np.array([np.sin(ang), np.cos(ang)])
        if not (2 < p[0] < grid - 2 and 2 < p[1] < grid - 2):
            continue
        if np.linalg.norm(p - foyer) < safe:
            continue
        if trucks and min(np.linalg.norm(p - np.array(trucks), axis=1)) < grid * 0.07:
            continue
        trucks.append(p)
    return np.array(trucks)


def run(grid, n_attack_trucks, n_def_trucks, dpt_attack, dpt_def, n_assets,
        ticks, seed, use_fleet=True, record=False):
    sim, foyer, assets = setup(grid, n_assets, seed)
    amask = asset_mask(assets, grid)
    na_cells = int(amask.sum())
    n_trucks = n_attack_trucks + n_def_trucks
    trucks = place_trucks(grid, n_trucks, foyer, seed)
    n_trucks = len(trucks)
    n_attack_trucks = min(n_attack_trucks, n_trucks)
    n_def_trucks = n_trucks - n_attack_trucks
    atk_trucks = trucks[:n_attack_trucks]
    def_trucks = trucks[n_attack_trucks:]
    rng = np.random.default_rng(seed + 2)

    if use_fleet:
        # attack drones
        Na = n_attack_trucks * dpt_attack
        a_home = np.repeat(np.arange(n_attack_trucks), dpt_attack)[:Na]
        a_pos = atk_trucks[a_home] + rng.uniform(-2, 2, (Na, 2))
        a_water = np.full(Na, TANK)
        a_mode = np.zeros(Na, int)            # 0 fight, 1 refuel
        a_reserve = np.full(n_attack_trucks, TRUCK_RESERVE)
        # defender drones
        rings = asset_rings(assets, foyer, grid)
        Nd = n_def_trucks * dpt_def
        d_home = np.repeat(np.arange(n_def_trucks), dpt_def)[:Nd]
        d_pos = def_trucks[d_home] + rng.uniform(-2, 2, (Nd, 2))
        d_slot = np.arange(Nd) % len(rings)
        d_target = rings[d_slot]
        d_tank = np.full(Nd, DTANK)
        d_mode = np.zeros(Nd, int)            # 0 lay/hold, 1 refuel
        d_reserve = np.full(n_def_trucks, TRUCK_RESERVE)
    treated = np.zeros((grid, grid), bool)

    retreat = RETREAT_M / CELL_M
    frames = []
    for t in range(ticks):
        thr = sim.ignition_threshold * 0.3
        fr, fc = np.where(sim.heat_intensity > thr)
        fire_xy = np.stack([fr, fc], 1).astype(float) if len(fr) else np.zeros((0, 2))
        ftree = cKDTree(fire_xy) if len(fire_xy) else None
        # trucks retreat from an encroaching front
        if ftree is not None:
            for k in range(n_trucks):
                d, _ = ftree.query(trucks[k])
                if d < retreat:
                    away = trucks[k] - foyer
                    trucks[k] = np.clip(trucks[k] + away / (np.linalg.norm(away) + 1e-6)
                                        * retreat * 1.5, 0, grid - 1)
            atk_trucks = trucks[:n_attack_trucks]; def_trucks = trucks[n_attack_trucks:]

        if use_fleet:
            a_reserve = np.minimum(TRUCK_RESERVE, a_reserve + TRUCK_TOPUP_LPM)
            d_reserve = np.minimum(TRUCK_RESERVE, d_reserve + TRUCK_TOPUP_LPM)
            # attack: nearest active fire per drone (once per tick)
            if ftree is not None:
                _, idx = ftree.query(a_pos); a_fire = fire_xy[idx]
            else:
                a_fire = a_pos
            a_dt = np.linalg.norm(a_pos[:, None, :] - atk_trucks[None, :, :], axis=2)
            a_homeidx = a_dt.argmin(axis=1)
            for _ in range(SUBSTEPS):
                a_mode[(a_mode == 0) & (a_water < 0.5)] = 1
                # move
                home = atk_trucks[a_homeidx]
                step = np.zeros((Na, 2))
                m1 = a_mode == 1
                dh = home - a_pos; nh = np.linalg.norm(dh, axis=1, keepdims=True)
                step = np.where(m1[:, None], np.where(nh > 1e-6, dh / nh, 0)
                                * np.minimum(CELLS_PER_SUB, nh), step)
                df = a_fire - a_pos; nf = np.linalg.norm(df, axis=1, keepdims=True)
                step = np.where((~m1)[:, None], np.where(nf > 1e-6, df / nf, 0) * CELLS_PER_SUB, step)
                a_pos = np.clip(a_pos + step, 0, grid - 1)
                ip = np.round(a_pos).astype(int)
                on = (a_mode == 0) & (sim.heat_intensity[ip[:, 0], ip[:, 1]] > thr) & (a_water > 0)
                for i in np.where(on)[0]:
                    amt = min(a_water[i], DROP_PER_SUB)
                    sim.apply_water_drop(int(ip[i, 0]), int(ip[i, 1]), amt * DROP_EFF)
                    a_water[i] -= amt
            # attack refuel
            at = np.linalg.norm(a_pos - atk_trucks[a_homeidx], axis=1) < 1.5
            for k in range(n_attack_trucks):
                here = np.where(at & (a_homeidx == k) & (a_water < TANK) & (a_mode == 1))[0]
                budget = min(TRUCK_DISPENSE_LPM, a_reserve[k])
                for i in here:
                    if budget <= 0:
                        break
                    give = min(TANK - a_water[i], budget)
                    a_water[i] += give; budget -= give; a_reserve[k] -= give
                    if a_water[i] >= TANK - 1e-6:
                        a_mode[i] = 0

            # defenders
            d_dt = np.linalg.norm(d_pos[:, None, :] - def_trucks[None, :, :], axis=2)
            d_homeidx = d_dt.argmin(axis=1)
            for _ in range(SUBSTEPS):
                d_mode[(d_mode == 0) & (d_tank <= 0)] = 1
                d_mode[(d_mode == 1) & (d_tank >= DTANK - 1e-6)] = 0
                goal = np.where((d_mode == 1)[:, None], def_trucks[d_homeidx], d_target)
                dd = goal - d_pos; nd = np.linalg.norm(dd, axis=1, keepdims=True)
                d_pos = np.clip(d_pos + np.where(nd > 1e-6, dd / nd, 0)
                                * np.minimum(CELLS_PER_SUB, nd), 0, grid - 1)
                ip = np.round(d_pos).astype(int)
                at_line = (d_mode == 0) & (nd[:, 0] < 1.5) & (d_tank > 0)
                for i in np.where(at_line)[0]:
                    r, c = ip[i]
                    if treated[r, c]:
                        continue
                    rr = slice(max(0, r - LINE_HALF_WIDTH), r + LINE_HALF_WIDTH + 1)
                    cc = slice(max(0, c - LINE_HALF_WIDTH), c + LINE_HALF_WIDTH + 1)
                    sim.fuel_load[rr, cc] = 0.0; sim.fuel_moisture[rr, cc] = 1.0
                    treated[rr, cc] = True; d_tank[i] -= LAY_COST
            d_at = np.linalg.norm(d_pos - def_trucks[d_homeidx], axis=1) < 1.5
            for k in range(n_def_trucks):
                here = np.where(d_at & (d_homeidx == k) & (d_tank < DTANK) & (d_mode == 1))[0]
                budget = min(TRUCK_DISPENSE_LPM, d_reserve[k])
                for i in here:
                    if budget <= 0:
                        break
                    give = min(DTANK - d_tank[i], budget)
                    d_tank[i] += give; budget -= give; d_reserve[k] -= give
                    if d_tank[i] >= DTANK - 1e-6:
                        d_mode[i] = 0

        if record and t % 2 == 0:
            snap = dict(heat=sim.heat_intensity.copy(), burned=sim.burned_area.copy(),
                        treated=treated.copy())
            if use_fleet:
                snap.update(a_pos=a_pos.copy(), a_mode=a_mode.copy(),
                            d_pos=d_pos.copy(), d_mode=d_mode.copy(),
                            atk=atk_trucks.copy(), dft=def_trucks.copy())
            frames.append(snap)
        sim.step()

    asset_burned = int((sim.burned_area & amask).sum())
    total_burned = int(sim.burned_area.sum())
    return asset_burned, na_cells, total_burned, frames, assets


def render(frames, grid, assets, out, fps, status):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import imageio.v2 as imageio
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7,
                                macro_block_size=None)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    for t, s in enumerate(frames):
        ax.clear()
        ax.imshow(s["heat"], origin="lower", cmap="inferno", vmin=0, vmax=1000,
                  extent=[0, grid, 0, grid])
        ty, tx = np.where(s["treated"])
        if len(ty):
            ax.scatter(tx, ty, s=2, c="deepskyblue", alpha=0.5, marker="s")
        for r0, r1, c0, c1, name, ctr in assets:
            hit = bool(s["burned"][r0:r1, c0:c1].any())
            col = "red" if hit else "lime"
            ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=col, lw=1.8))
            ax.text((c0 + c1) / 2, r1 + 1.5, name, color=col, fontsize=6,
                    ha="center", fontweight="bold")
        if "a_pos" in s:
            ap, am = s["a_pos"], s["a_mode"]
            ax.scatter(ap[am == 0, 1], ap[am == 0, 0], s=4, c="lime", alpha=0.7,
                       label="attack (fighting)" if t == 0 else None)
            ax.scatter(ap[am == 1, 1], ap[am == 1, 0], s=4, c="orange", alpha=0.7,
                       label="attack (refuel)" if t == 0 else None)
            dp, dm = s["d_pos"], s["d_mode"]
            ax.scatter(dp[dm == 0, 1], dp[dm == 0, 0], s=4, c="white", alpha=0.8,
                       label="defender (laying)" if t == 0 else None)
            ax.scatter(dp[dm == 1, 1], dp[dm == 1, 0], s=4, c="violet", alpha=0.8,
                       label="defender (refuel)" if t == 0 else None)
            for j, tr in enumerate(s["atk"]):
                ax.scatter(tr[1], tr[0], s=80, marker="s", c="red", edgecolors="k", linewidths=0.6)
                ax.text(tr[1], tr[0], str(j + 1), color="white", fontsize=6, ha="center", va="center")
            for j, tr in enumerate(s["dft"]):
                ax.scatter(tr[1], tr[0], s=90, marker="s", c="blue", edgecolors="k", linewidths=0.6)
                ax.text(tr[1], tr[0], f"D{j+1}", color="white", fontsize=6, ha="center", va="center")
        ax.set_title(f"Combined attack + defense   tick {t*2}\n{status}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.legend(loc="upper right", fontsize=5.5, markerscale=2, framealpha=0.7)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close(); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=300)
    p.add_argument("--attack-trucks", type=int, default=8)
    p.add_argument("--defender-trucks", type=int, default=2)
    p.add_argument("--dpt-attack", type=int, default=110)
    p.add_argument("--dpt-def", type=int, default=180)
    p.add_argument("--assets", type=int, default=6)
    p.add_argument("--ticks", type=int, default=320)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--tank", type=float, default=None,
                   help="per-drone payload in L (sets attack water + defender retardant)")
    p.add_argument("--wind", type=float, default=None, help="wind speed m/s")
    p.add_argument("--spread", type=float, default=None, help="base spread rate")
    p.add_argument("--foyer", type=float, default=None, help="foyer radius (cells)")
    p.add_argument("--dry", action="store_true", help="very dry fuel (intense fire)")
    p.add_argument("--out", default="docs/figures/combined.mp4")
    p.add_argument("--fps", type=int, default=18)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    global TANK, DTANK, WIND, SPREAD, FOYER_R, MOIST
    if args.tank:
        TANK = args.tank
        DTANK = args.tank
    if args.wind is not None:
        WIND = args.wind
    if args.spread is not None:
        SPREAD = args.spread
    if args.foyer is not None:
        FOYER_R = args.foyer
    if args.dry:
        MOIST = (0.02, 0.05)
    tot = args.grid ** 2

    print("baseline (no fleet) ...")
    ab0, na, tb0, _, _ = run(args.grid, args.attack_trucks, args.defender_trucks,
                             args.dpt_attack, args.dpt_def, args.assets,
                             args.ticks, args.seed, use_fleet=False)
    print(f"  assets burned {ab0}/{na} ({ab0/max(na,1):.0%})  total {tb0} ({tb0/tot:.0%})")

    print("combined fleet (8 attack + 2 defender) ...")
    ab1, na, tb1, frames, assets = run(args.grid, args.attack_trucks, args.defender_trucks,
                                       args.dpt_attack, args.dpt_def, args.assets,
                                       args.ticks, args.seed, use_fleet=True, record=True)
    print(f"  assets burned {ab1}/{na} ({ab1/max(na,1):.0%})  total {tb1} ({tb1/tot:.0%})")
    print(f"  total-burn reduction {(tb0-tb1)/max(tb0,1):+.0%}")
    status = (f"baseline: {ab0}/{na} assets lost, {tb0/tot:.0%} grid  |  "
              f"fleet: {ab1}/{na} assets lost, {tb1/tot:.0%} grid")
    render(frames, args.grid, assets, args.out, args.fps, status)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
