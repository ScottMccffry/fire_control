#!/usr/bin/env python3
"""
Realistic truck-based drone firefighting simulation.

Model (units: 25 m / cell, 1 tick = 1 minute):
  * Fire: an irregular blob with a random epicentre, fit inside a 500 m
    circumscribed circle, spreading via the (vectorized) mock fire physics.
  * Trucks (up to 10): each carries `drones_per_truck` drones and 20,000 L of
    water, dispensed at 100 L/min (the throughput bottleneck). Placed at random
    but >= 1 km from the fire front; a truck RETREATS when active fire comes
    within 200 m.
  * Drones (10 L tank): launch from their truck, fly to the nearest ACTIVE fire
    cell (never burned-out cells), drop water, and return to the nearest truck
    with water to refuel (rate-limited by the truck).

Heuristic dispatcher (baseline). A learned dispatch policy plugs in via
`--policy` (see truck_env.py / train_truck_dispatch.py).

Usage:
  python scripts/truck_sim.py --grid 400 --trucks 10 --drones-per-truck 300 \
      --ticks 400 --out docs/figures/truck_sim.mp4
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
TANK = 10.0
DROP_RATE_LPS = 5.0
DRONE_MPS = 15.0
SUBSTEPS = 6                    # movement increments per tick (1 tick = 60 s)
SUB_DT = 60.0 / SUBSTEPS       # seconds per sub-step
DRONE_CELLS_PER_SUB = DRONE_MPS * SUB_DT / CELL_M     # cells moved per sub-step
DROP_PER_SUB = DROP_RATE_LPS * SUB_DT                 # litres per sub-step over fire
TRUCK_RESERVE = 20000.0
TRUCK_DISPENSE_LPM = 100.0     # litres/min a truck can hand to drones
SAFE_M = 1000.0                # min truck distance from fire front at placement
RETREAT_M = 200.0             # fire-front distance that triggers a truck retreat


def make_irregular_fire(grid, circ_radius_m, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sim = MockFireSimulator(grid_size=(grid, grid), cell_size_meters=CELL_M)
    sim.set_weather(wind_speed=5.0, wind_direction=float(rng.uniform(0, 360)),
                    temperature=32.0, humidity=0.15)
    sim.fuel_moisture = rng.uniform(0.06, 0.16, (grid, grid))
    sim.fuel_load = rng.uniform(0.8, 1.0, (grid, grid))
    sim.base_spread_rate = 0.16
    R = circ_radius_m / CELL_M
    cy = grid / 2 + rng.uniform(-grid * 0.1, grid * 0.1)
    cx = grid / 2 + rng.uniform(-grid * 0.1, grid * 0.1)
    # irregular radial boundary from a few Fourier harmonics, scaled <= R
    th = np.linspace(0, 2 * np.pi, 360)
    rad = np.ones_like(th)
    for k in range(1, 5):
        rad += rng.uniform(-0.5, 0.5) / k * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    rad = np.clip(rad, 0.3, None)
    rad *= R / rad.max()                      # circumscribed radius == R
    yy, xx = np.mgrid[0:grid, 0:grid]
    ang = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    boundary = np.interp(ang, th, rad)
    mask = dist <= boundary
    sim.burned_area[mask] = 1
    sim.heat_intensity[mask] = sim.max_heat_intensity * 0.7
    return sim, (cy, cx)


def active_cells(sim):
    thr = sim.ignition_threshold * 0.3
    fr, fc = np.where(sim.heat_intensity > thr)
    return np.stack([fr, fc], axis=1).astype(float) if len(fr) else np.zeros((0, 2))


def place_trucks(grid, n, fire_xy, seed):
    rng = np.random.default_rng(seed + 1)
    tree = cKDTree(fire_xy) if len(fire_xy) else None
    safe = SAFE_M / CELL_M
    trucks = []
    tries = 0
    while len(trucks) < n and tries < 5000:
        tries += 1
        p = rng.uniform(0.05, 0.95, 2) * grid
        if tree is not None and tree.query(p)[0] < safe:
            continue
        if trucks and min(np.linalg.norm(p - np.array(trucks), axis=1)) < grid * 0.08:
            continue
        trucks.append(p)
    return np.array(trucks)


def run(grid, n_trucks, dpt, ticks, seed, record=False, use_drones=True, deploy_delay=0, policy_path=None):
    sim, _ = make_irregular_fire(grid, 500.0, seed)
    fire_xy = active_cells(sim)
    trucks = place_trucks(grid, n_trucks, fire_xy, seed)
    n_trucks = len(trucks)
    N = n_trucks * dpt
    rng = np.random.default_rng(seed + 2)

    if use_drones:
        home_truck = np.repeat(np.arange(n_trucks), dpt)[:N]
        pos = trucks[home_truck].astype(float) + rng.uniform(-2, 2, (N, 2))
        water = np.full(N, TANK)
        mode = np.zeros(N, dtype=int)               # 0 fight, 1 return/refuel
        reserve = np.full(n_trucks, TRUCK_RESERVE)
        ever_dropped = np.zeros(N, bool)
        diag = []

    obs_env, policy_model = None, None
    if use_drones and policy_path:
        from stable_baselines3 import PPO
        from envs.per_drone_env import PerDroneSwarmVecEnv
        policy_model = PPO.load(policy_path)
        obs_env = PerDroneSwarmVecEnv(grid_size=grid, n_drones=N, continuous=True)

    frames, burned_curve, water_used = [], [], 0.0
    retreat = RETREAT_M / CELL_M
    for t in range(ticks):
        fire_xy = active_cells(sim)
        ftree = cKDTree(fire_xy) if len(fire_xy) else None
        # --- trucks retreat from an encroaching front ---
        if len(fire_xy):
            for k in range(n_trucks):
                d, _ = ftree.query(trucks[k])
                if d < retreat:
                    cen = fire_xy.mean(axis=0)
                    away = trucks[k] - cen
                    away = away / (np.linalg.norm(away) + 1e-6)
                    trucks[k] = np.clip(trucks[k] + away * (retreat * 1.5), 0, grid - 1)
        if use_drones:
            reserve = np.minimum(TRUCK_RESERVE, reserve + TRUCK_DISPENSE_LPM)  # supply line tops up

        # Response delay: before deployment, drones wait at their (possibly
        # retreating) trucks while the fire grows -- a realistic mobilization lag.
        if use_drones and t < deploy_delay:
            pos = trucks[home_truck] + rng.uniform(-2, 2, (N, 2))
            if record and t % 2 == 0:
                frames.append((sim.heat_intensity.copy(), pos.copy(),
                               np.ones(N, dtype=int), trucks.copy(), reserve.copy()))
            sim.step(); burned_curve.append(int(sim.burned_area.sum()))
            continue

        if use_drones:
            tick_drops = 0
            for _ in range(SUBSTEPS):
                ipos = np.clip(np.round(pos).astype(int), 0, grid - 1)
                # nearest active fire per drone
                if ftree is not None:
                    _, idx = ftree.query(pos)
                    nearest_fire = fire_xy[idx]
                else:
                    nearest_fire = pos
                # nearest truck
                dt_ = np.linalg.norm(pos[:, None, :] - trucks[None, :, :], axis=2)
                home_idx = dt_.argmin(axis=1)
                home = trucks[home_idx]
                mode[(mode == 0) & (water < 0.5)] = 1   # empty -> go refuel
                # Movement: fighting drones steered by the trained vec policy (if
                # given) else heuristic toward nearest fire; refuelling drones
                # always head to their nearest truck (logistics override).
                step = np.zeros((N, 2))
                m1 = mode == 1
                dh = home - pos; nh = np.linalg.norm(dh, axis=1, keepdims=True)
                step = np.where((m1)[:, None], np.where(nh > 1e-6, dh / nh, 0) * DRONE_CELLS_PER_SUB, step)
                m0 = mode == 0
                if obs_env is not None:
                    obs_env.sim = sim; obs_env.drone_pos = ipos; obs_env.drone_water = water
                    act, _ = policy_model.predict(obs_env._build_obs(), deterministic=True)
                    step = np.where(m0[:, None], np.clip(act, -1, 1) * DRONE_CELLS_PER_SUB, step)
                else:
                    df = nearest_fire - pos; nf = np.linalg.norm(df, axis=1, keepdims=True)
                    step = np.where(m0[:, None], np.where(nf > 1e-6, df / nf, 0) * DRONE_CELLS_PER_SUB, step)
                pos = np.clip(pos + step, 0, grid - 1)
                ipos = np.clip(np.round(pos).astype(int), 0, grid - 1)
                # drops on ACTIVE fire only
                if ftree is not None:
                    thr = sim.ignition_threshold * 0.3
                    on = (mode == 0) & (sim.heat_intensity[ipos[:, 0], ipos[:, 1]] > thr) & (water > 0)
                    didx = np.where(on)[0]; tick_drops += len(didx); ever_dropped[didx] = True
                    for i in didx:
                        amt = min(water[i], DROP_PER_SUB)
                        sim.apply_water_drop(int(ipos[i, 0]), int(ipos[i, 1]), amt)
                        water[i] -= amt
                        water_used += amt
            # --- refuel at trucks (rate + reserve limited), once per tick ---
            at = np.linalg.norm(pos - home, axis=1) < 1.5
            for k in range(n_trucks):
                here = np.where(at & (home_idx == k) & (water < TANK) & (mode == 1))[0]
                budget = min(TRUCK_DISPENSE_LPM, reserve[k])
                for i in here:
                    if budget <= 0:
                        break
                    give = min(TANK - water[i], budget)
                    water[i] += give
                    budget -= give
                    reserve[k] -= give
                    if water[i] >= TANK - 1e-6:
                        mode[i] = 0
            if t >= deploy_delay:
                diag.append((tick_drops, int((mode == 1).sum()), int((mode == 0).sum())))
            if record and t % 2 == 0:
                st = np.where(mode == 1, 1, 0)  # 0 fight, 1 refuel/return
                frames.append((sim.heat_intensity.copy(), pos.copy(), st,
                               trucks.copy(), reserve.copy()))
        elif record and t % 2 == 0:
            frames.append((sim.heat_intensity.copy(), None, None, trucks.copy(), None))

        sim.step()
        burned_curve.append(int(sim.burned_area.sum()))
    if use_drones and diag:
        d = np.array(diag, float)
        print(f"  DIAG: per tick avg -> dropping {d[:,0].mean():.0f}/{N} drones, "
              f"refuelling {d[:,1].mean():.0f}, fighting {d[:,2].mean():.0f}; "
              f"drones that EVER dropped: {ever_dropped.sum()}/{N} ({ever_dropped.mean():.0%})")
        print(f"  refuel ceiling = {int(len(trucks)*TRUCK_DISPENSE_LPM/TANK)} drones/min "
              f"(={len(trucks)}x100 L/min / {TANK:.0f} L)")
    return int(sim.burned_area.sum()), burned_curve, frames, water_used, n_trucks


def render(frames, grid, out, fps, b_final, d_final):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7,
                                macro_block_size=None)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for t, (heat, pos, st, trucks, reserve) in enumerate(frames):
        ax.clear()
        ax.imshow(heat, origin="lower", cmap="inferno", vmin=0, vmax=1000,
                  extent=[0, grid, 0, grid])
        if pos is not None:
            f = st == 0
            ax.scatter(pos[f, 1], pos[f, 0], s=7, c="lime", alpha=0.8,
                       label="fighting" if t == 0 else None)
            ax.scatter(pos[~f, 1], pos[~f, 0], s=7, c="deepskyblue", alpha=0.8,
                       label="refuel/return" if t == 0 else None)
        ax.scatter(trucks[:, 1], trucks[:, 0], s=130, marker="s", c="cyan",
                   edgecolors="black", linewidths=0.8, label="truck" if t == 0 else None)
        ax.set_title(f"Truck-based swarm vs irregular fire   tick {t*2}\n"
                     f"burned now {int((heat>0).sum())}  "
                     f"(no-swarm ends {b_final}, swarm ends {d_final})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.legend(loc="upper right", fontsize=6, markerscale=3, framealpha=0.6)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close(); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=400)
    p.add_argument("--trucks", type=int, default=10)
    p.add_argument("--drones-per-truck", type=int, default=300)
    p.add_argument("--ticks", type=int, default=400)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="docs/figures/truck_sim.mp4")
    p.add_argument("--deploy-delay", type=int, default=0, help="ticks before drones launch (mobilization lag)")
    p.add_argument("--policy", default=None, help="path to a trained vec policy to steer fighting drones")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    tot = args.grid * args.grid

    print("baseline (no swarm) ...")
    b_final, b_curve, _, _, _ = run(args.grid, args.trucks, args.drones_per_truck,
                                    args.ticks, args.seed, use_drones=False)
    print(f"  baseline burned: {b_final} ({b_final/tot:.0%} of {args.grid*25/1000:.0f}km grid)")

    print(f"swarm ({args.trucks} trucks x {args.drones_per_truck} drones) ...")
    d_final, d_curve, frames, water, ntr = run(args.grid, args.trucks,
                                               args.drones_per_truck, args.ticks,
                                               args.seed, record=True,
                                               deploy_delay=args.deploy_delay,
                                               policy_path=args.policy)
    print(f"  trucks placed: {ntr}")
    print(f"  swarm burned:    {d_final} ({d_final/tot:.0%})")
    print(f"  reduction: {b_final - d_final} cells ({(b_final-d_final)/b_final:+.0%})")
    print(f"  water delivered: {water:,.0f} L")
    render(frames, args.grid, args.out, args.fps, b_final, d_final)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
