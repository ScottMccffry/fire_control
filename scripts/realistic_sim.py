#!/usr/bin/env python3
"""
More realistic combined scenario:
  * 3D terrain (real elevation) -> fire runs faster UPHILL (slope_factor in the
    fire model), rendered as a hillshade.
  * Several small ignition triggers (not one big foyer) that grow and merge.
  * Attack trucks (water) + defender trucks (RETARDANT) -- both carry a FINITE
    reserve. When a truck runs low it drives to a resupply DEPOT at the map edge,
    refills, and returns; its drones refuel only while it is on station.
  * Defenders compartmentalize (grid of retardant lines) so trapped pockets can
    be extinguished; retardant raises the ignition bar (not a fireproof wall).

Usage:
  python scripts/realistic_sim.py --out docs/figures/realistic.mp4
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from sim import MockFireSimulator          # noqa: E402
from scipy.spatial import cKDTree          # noqa: E402
import combined_sim as C                    # noqa: E402  (reuse geometry helpers)

CELL_M = 25.0
SUBSTEPS = 4
DRONE_CPS = 16.0 * (60.0 / SUBSTEPS) / CELL_M     # drone cells/substep
TRUCK_SPEED = 6.0                                  # cells/tick (slow, off-road)
WATER_TANK = 60.0
RET_TANK = 60.0
DROP_PER_SUB = 5.0 * (60.0 / SUBSTEPS)
DROP_EFF = 0.7
LINE_HALF = 1
LINE_STRENGTH = 1.0
LINE_KW = 800.0
TRUCK_RESERVE = 9000.0          # finite per-truck reserve (water or retardant)
TRUCK_LOW = 1800.0              # drive to depot below this
DEPOT_RATE = 1800.0            # reserve refilled per tick while at a depot
DISPENSE_LPM = 220.0           # handed to drones per tick while on station
REFUEL_R = 2.0


def adaptive_lattice(center, radius, wind_dir, comp, grid):
    """Internal compartment lattice ROTATED to the wind/spread axis (clipped to
    the containment disk). Same density as a fixed square grid, but its lines run
    ALONG and ACROSS the spread direction -> the division pattern rotates with
    each fire's wind instead of always being axis-aligned. The robust circular
    perimeter (built separately) handles enclosure."""
    wd = np.radians(wind_dir)
    u = np.array([np.sin(wd), np.cos(wd)])         # downwind / spread axis
    v = np.array([-u[1], u[0]])                     # cross-spread axis
    pts = []
    for uu in np.arange(-radius + comp, radius, comp):     # lines across spread
        half = np.sqrt(max(radius ** 2 - uu ** 2, 0))
        vv = np.arange(-half, half, 1.8)
        pts.append(center + uu * u + np.outer(vv, v))
    for vl in np.arange(-radius + comp, radius, comp):     # lines along spread
        half = np.sqrt(max(radius ** 2 - vl ** 2, 0))
        uu = np.arange(-half, half, 1.8)
        pts.append(center + np.outer(uu, u) + vl * v)
    return np.clip(np.vstack(pts), 1, grid - 2)


def spider_web(center, r_in, r_out, grid, ring_gap=16.0, n_spokes=16):
    """Concentric rings + radial spokes (a spider web) from the outer perimeter
    inward to just ahead of the fire front -- DEFENSE IN DEPTH: each ring is a
    fallback line, so if the fire breaches one the next ring out still catches
    it; the spokes stop the fire running laterally within an annulus."""
    pts = []
    radii = np.arange(r_in, r_out + 1e-6, ring_gap)
    for R in radii:                                   # concentric rings
        n = max(24, int(2 * np.pi * R / 1.8))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(center + R * np.stack([np.sin(th), np.cos(th)], 1))
    for k in range(n_spokes):                         # radial spokes
        ang = 2 * np.pi * k / n_spokes
        rr = np.arange(r_in, r_out, 1.8)
        d = np.array([np.sin(ang), np.cos(ang)])
        pts.append(center + np.outer(rr, d))
    return np.clip(np.vstack(pts), 1, grid - 2)


def make_terrain(grid, rng, relief=180.0):
    """Hills/valleys; amplified so slope actually drives spread."""
    x = np.linspace(0, 4 * np.pi, grid)
    X, Y = np.meshgrid(x, x)
    t = (np.sin(X) * np.cos(Y) + 0.5 * np.sin(2.1 * X + 1) * np.cos(1.7 * Y)
         + 0.3 * np.sin(3.3 * X) * np.cos(3.1 * Y + 2))
    t += 0.4 * rng.standard_normal((grid, grid))
    t = (t - t.min()) / (t.max() - t.min())
    return t * relief


def setup(grid, n_assets, n_fires, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sim = MockFireSimulator(grid_size=(grid, grid), cell_size_meters=CELL_M)
    sim.fuel_moisture = rng.uniform(0.03, 0.08, (grid, grid))
    sim.fuel_load = rng.uniform(0.85, 1.0, (grid, grid))
    sim.base_spread_rate = 0.45
    sim.max_heat_intensity = 1400.0
    sim.retardant_kw = LINE_KW
    sim.elevation = make_terrain(grid, rng)        # 3D terrain (slope matters)
    sim.set_weather(wind_speed=12.0, wind_direction=float(rng.uniform(0, 360)),
                    temperature=34.0, humidity=0.10)

    # several small ignition triggers scattered in the middle of the map
    c = np.array([grid / 2, grid / 2])
    fires = []
    yy, xx = np.mgrid[0:grid, 0:grid]
    for _ in range(n_fires):
        f = c + rng.uniform(-0.18, 0.18, 2) * grid
        disk = (yy - f[0]) ** 2 + (xx - f[1]) ** 2 <= 3.0 ** 2
        sim.burned_area[disk] = 1
        sim.heat_intensity[disk] = sim.max_heat_intensity * 0.85
        fires.append(f)
    foyer = np.mean(fires, axis=0)

    # critical infrastructure scattered around
    assets, names = [], ["town", "substation", "hospital", "fuel depot",
                         "data center", "water plant"]
    tries = 0
    while len(assets) < n_assets and tries < 2000:
        tries += 1
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.18, 0.36) * grid
        ctr = foyer + rad * np.array([np.sin(ang), np.cos(ang)])
        hh, hw = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        r0, r1, c0, c1 = int(ctr[0]-hh), int(ctr[0]+hh), int(ctr[1]-hw), int(ctr[1]+hw)
        if r0 < 6 or c0 < 6 or r1 > grid-6 or c1 > grid-6:
            continue
        if assets and min(np.linalg.norm(ctr-np.array([a[5] for a in assets]), axis=1)) < grid*0.12:
            continue
        assets.append((r0, r1, c0, c1, names[len(assets) % len(names)], ctr))
    return sim, foyer, assets


def amask(assets, grid):
    m = np.zeros((grid, grid), bool)
    for r0, r1, c0, c1, *_ in assets:
        m[r0:r1, c0:c1] = True
    return m


def hillshade(ele, az=315.0, alt=45.0):
    gy, gx = np.gradient(ele, edge_order=1)
    slope = np.pi/2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az_r, alt_r = np.radians(az), np.radians(alt)
    hs = (np.sin(alt_r)*np.sin(slope) +
          np.cos(alt_r)*np.cos(slope)*np.cos(az_r - aspect))
    return np.clip(hs, 0, 1)


def run(grid, n_atk, n_def, dpt_a, dpt_d, n_assets, n_fires, comp, ticks, seed,
        use_fleet=True, record=False, deploy_delay=0, pattern="web"):
    sim, foyer, assets = setup(grid, n_assets, n_fires, seed)
    am = amask(assets, grid)
    na = int(am.sum())
    perim_r = 0.34 * grid
    depots = np.array([[grid*0.04, grid*0.04], [grid*0.96, grid*0.96],
                       [grid*0.04, grid*0.96]])

    atk_t = C.place_trucks(grid, n_atk, foyer, seed)
    def_t = C.place_trucks(grid, n_def, foyer, seed + 50)
    # truck dynamic state: pos, home, reserve, mode (0 on-station,1 to depot,2 back)
    trucks = np.vstack([atk_t, def_t])
    nT = len(trucks)
    t_home = trucks.copy()
    t_pos = trucks.copy().astype(float)
    t_res = np.full(nT, TRUCK_RESERVE)
    t_mode = np.zeros(nT, int)
    na_t = len(atk_t)

    if use_fleet:
        Na = na_t * dpt_a
        a_home = np.repeat(np.arange(na_t), dpt_a)[:Na]
        a_pos = t_pos[a_home] + np.random.uniform(-2, 2, (Na, 2))
        a_w = np.full(Na, WATER_TANK)
        a_mode = np.zeros(Na, int)
        Nd = (nT - na_t) * dpt_d
        d_home = np.repeat(np.arange(nT - na_t), dpt_d)[:Nd]
        d_pos = t_pos[na_t + d_home] + np.random.uniform(-2, 2, (Nd, 2))
        d_r = np.full(Nd, RET_TANK)
        d_mode = np.zeros(Nd, int)
        # containment lines are sized to the fire AS FOUND on arrival, so build
        # them at deployment time (after the mobilization delay), not up front.
        rings, d_tgt, deployed = None, d_pos.copy(), False
    treated = np.zeros((grid, grid), bool)
    frames = []

    for t in range(ticks):
        thr = sim.ignition_threshold * 0.3
        fr, fc = np.where(sim.heat_intensity > thr)
        fire_xy = np.stack([fr, fc], 1).astype(float) if len(fr) else np.zeros((0, 2))
        ftree = cKDTree(fire_xy) if len(fire_xy) else None

        # --- deploy after the mobilization delay: size containment to the fire
        #     as found on arrival (crews build lines around the bigger fire) ---
        if use_fleet and not deployed and t >= deploy_delay:
            if len(fire_xy):
                cen = fire_xy.mean(0)
                rad = float(np.linalg.norm(fire_xy - cen, axis=1).max()) + 14.0
            else:
                cen, rad = foyer, perim_r
            rad = float(np.clip(rad, 20.0, 0.46 * grid))
            if pattern == "web":
                # spider web: rings (defense in depth) + spokes, from the fire
                # front (r_in) out to the perimeter (r_out)
                r_in = float(np.clip(rad - 4, 12.0, rad))
                r_out = float(np.clip(rad + 0.18 * grid, rad + comp, 0.47 * grid))
                # finer rings = deeper defense in depth (many closely-spaced
                # fallback lines), which is the whole point of the web
                lattice = spider_web(cen, r_in, r_out, grid, ring_gap=12.0, n_spokes=20)
            else:
                lattice = np.vstack([C.containment_perimeter(cen, grid, rad),
                                     adaptive_lattice(cen, rad, sim.wind_direction, comp, grid)])
            rings = np.vstack([C.asset_rings(assets, foyer, grid), lattice])
            d_tgt = rings[np.arange(Nd) % len(rings)]
            deployed = True

        if not (use_fleet and deployed):
            if record and t % 2 == 0:
                snap = dict(heat=sim.heat_intensity.copy(), burned=sim.burned_area.copy(),
                            treated=treated.copy(), tpos=t_pos.copy(), tmode=t_mode.copy())
                if use_fleet:
                    snap.update(a_pos=a_pos.copy(), a_mode=a_mode.copy(),
                                d_pos=d_pos.copy(), d_mode=d_mode.copy())
                frames.append(snap)
            sim.step()
            continue

        # --- trucks: resupply runs to depots when low ---
        for k in range(nT):
            if t_mode[k] == 0 and t_res[k] < TRUCK_LOW:
                t_mode[k] = 1
            if t_mode[k] == 1:
                d = depots[np.argmin(np.linalg.norm(depots - t_pos[k], axis=1))]
                v = d - t_pos[k]; n = np.linalg.norm(v) + 1e-9
                t_pos[k] += v / n * min(TRUCK_SPEED, n)
                if n < 2:
                    t_res[k] = min(TRUCK_RESERVE, t_res[k] + DEPOT_RATE)
                    if t_res[k] >= TRUCK_RESERVE - 1:
                        t_mode[k] = 2
            elif t_mode[k] == 2:
                v = t_home[k] - t_pos[k]; n = np.linalg.norm(v) + 1e-9
                t_pos[k] += v / n * min(TRUCK_SPEED, n)
                if n < 2:
                    t_mode[k] = 0
        atk_pos = t_pos[:na_t]; def_pos = t_pos[na_t:]
        atk_on = t_mode[:na_t] == 0           # available to refuel drones
        def_on = t_mode[na_t:] == 0

        if use_fleet:
            # ATTACK drones
            if ftree is not None:
                _, idx = ftree.query(a_pos); a_fire = fire_xy[idx]
            else:
                a_fire = a_pos
            avail = np.where(atk_on)[0]
            an = avail if len(avail) else np.arange(na_t)
            adt = np.linalg.norm(a_pos[:, None] - atk_pos[None, an], axis=2)
            ahi = an[adt.argmin(1)]
            home = atk_pos[ahi]
            for _ in range(SUBSTEPS):
                a_mode[(a_mode == 0) & (a_w < 0.5)] = 1
                m1 = a_mode == 1
                dh = home - a_pos; nh = np.linalg.norm(dh, axis=1, keepdims=True)
                step = np.where(m1[:, None], np.where(nh > 1e-6, dh/nh, 0)*np.minimum(DRONE_CPS, nh), 0)
                df = a_fire - a_pos; nf = np.linalg.norm(df, axis=1, keepdims=True)
                step = np.where((~m1)[:, None], np.where(nf > 1e-6, df/nf, 0)*DRONE_CPS, step)
                a_pos = np.clip(a_pos + step, 0, grid-1)
                ip = np.round(a_pos).astype(int)
                on = (a_mode == 0) & (sim.heat_intensity[ip[:, 0], ip[:, 1]] > thr) & (a_w > 0)
                for i in np.where(on)[0]:
                    amt = min(a_w[i], DROP_PER_SUB)
                    sim.apply_water_drop(int(ip[i, 0]), int(ip[i, 1]), amt*DROP_EFF)
                    a_w[i] -= amt
            at = (np.linalg.norm(a_pos - home, axis=1) < REFUEL_R) & (a_mode == 1)
            for k in an:
                hh = np.where(at & (ahi == k) & (a_w < WATER_TANK))[0]
                b = min(DISPENSE_LPM, t_res[k])
                for i in hh:
                    if b <= 0:
                        break
                    g = min(WATER_TANK - a_w[i], b); a_w[i] += g; b -= g; t_res[k] -= g
                    if a_w[i] >= WATER_TANK-1e-6:
                        a_mode[i] = 0

            # DEFENDER drones
            availd = np.where(def_on)[0]
            dn = availd if len(availd) else np.arange(nT-na_t)
            ddt = np.linalg.norm(d_pos[:, None] - def_pos[None, dn], axis=2)
            dhi = dn[ddt.argmin(1)]
            for _ in range(SUBSTEPS):
                d_mode[(d_mode == 0) & (d_r <= 0)] = 1
                d_mode[(d_mode == 1) & (d_r >= RET_TANK-1e-6)] = 0
                goal = np.where((d_mode == 1)[:, None], def_pos[dhi], d_tgt)
                dd = goal - d_pos; nd = np.linalg.norm(dd, axis=1, keepdims=True)
                d_pos = np.clip(d_pos + np.where(nd > 1e-6, dd/nd, 0)*np.minimum(DRONE_CPS, nd), 0, grid-1)
                ip = np.round(d_pos).astype(int)
                lay = (d_mode == 0) & (nd[:, 0] < 1.5) & (d_r > 0)
                for i in np.where(lay)[0]:
                    r, c = ip[i]
                    if treated[r, c] or sim.burned_area[r, c]:
                        continue
                    rr = slice(max(0, r-LINE_HALF), r+LINE_HALF+1)
                    cc = slice(max(0, c-LINE_HALF), c+LINE_HALF+1)
                    sim.retardant[rr, cc] = LINE_STRENGTH
                    treated[rr, cc] = True; d_r[i] -= 1.0
            dat = (np.linalg.norm(d_pos - def_pos[dhi], axis=1) < REFUEL_R) & (d_mode == 1)
            for k in dn:
                hh = np.where(dat & (dhi == k) & (d_r < RET_TANK))[0]
                b = min(DISPENSE_LPM, t_res[na_t + k])
                for i in hh:
                    if b <= 0:
                        break
                    g = min(RET_TANK - d_r[i], b); d_r[i] += g; b -= g; t_res[na_t+k] -= g
                    if d_r[i] >= RET_TANK-1e-6:
                        d_mode[i] = 0

        if record and t % 2 == 0:
            snap = dict(heat=sim.heat_intensity.copy(), burned=sim.burned_area.copy(),
                        treated=treated.copy(), tpos=t_pos.copy(), tmode=t_mode.copy())
            if use_fleet:
                snap.update(a_pos=a_pos.copy(), a_mode=a_mode.copy(),
                            d_pos=d_pos.copy(), d_mode=d_mode.copy())
            frames.append(snap)
        sim.step()

    return (int((sim.burned_area & am).sum()), na, int(sim.burned_area.sum()),
            frames, assets, sim.elevation.copy(), depots, na_t)


def render(frames, grid, assets, ele, depots, na_t, out, fps, status):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LightSource
    import imageio.v2 as imageio
    hs = hillshade(ele)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=None)
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    for t, s in enumerate(frames):
        ax.clear()
        ax.imshow(hs, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[0, grid, 0, grid])
        ax.contour(ele, levels=10, colors="k", linewidths=0.25, alpha=0.35,
                   extent=[0, grid, 0, grid])
        heat = np.ma.masked_less(s["heat"], 1)
        ax.imshow(heat, origin="lower", cmap="inferno", vmin=0, vmax=1000, alpha=0.9,
                  extent=[0, grid, 0, grid])
        ty, tx = np.where(s["treated"])
        if len(ty):
            ax.scatter(tx, ty, s=1.5, c="deepskyblue", alpha=0.5, marker="s")
        for r0, r1, c0, c1, name, ctr in assets:
            hit = bool(s["burned"][r0:r1, c0:c1].any())
            col = "red" if hit else "lime"
            ax.add_patch(Rectangle((c0, r0), c1-c0, r1-r0, fill=False, ec=col, lw=1.6))
            ax.text((c0+c1)/2, r1+1.5, name, color=col, fontsize=6, ha="center", fontweight="bold")
        ax.scatter(depots[:, 1], depots[:, 0], s=140, marker="H", c="yellow",
                   edgecolors="k", label="resupply depot" if t == 0 else None)
        if "a_pos" in s:
            ap, am_ = s["a_pos"], s["a_mode"]
            ax.scatter(ap[am_ == 0, 1], ap[am_ == 0, 0], s=3, c="lime", alpha=.6,
                       label="attack" if t == 0 else None)
            ax.scatter(ap[am_ == 1, 1], ap[am_ == 1, 0], s=3, c="orange", alpha=.6)
            dp, dm = s["d_pos"], s["d_mode"]
            ax.scatter(dp[dm == 0, 1], dp[dm == 0, 0], s=3, c="white", alpha=.7,
                       label="defender" if t == 0 else None)
            ax.scatter(dp[dm == 1, 1], dp[dm == 1, 0], s=3, c="violet", alpha=.7)
        tp, tm = s["tpos"], s["tmode"]
        for j in range(len(tp)):
            col = "red" if j < na_t else "blue"
            mk = "s" if tm[j] == 0 else "X"      # X = resupplying / moving
            ax.scatter(tp[j, 1], tp[j, 0], s=70, marker=mk, c=col, edgecolors="k", linewidths=.5)
        ax.set_title(f"Realistic sim: 3D terrain, scattered fires, truck resupply   tick {t*2}\n{status}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.legend(loc="upper right", fontsize=5.5, markerscale=2, framealpha=.7)
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    w.close(); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=300)
    p.add_argument("--attack-trucks", type=int, default=5)
    p.add_argument("--defender-trucks", type=int, default=7)
    p.add_argument("--dpt-attack", type=int, default=700)
    p.add_argument("--dpt-def", type=int, default=700)
    p.add_argument("--assets", type=int, default=5)
    p.add_argument("--fires", type=int, default=5)
    p.add_argument("--compartment", type=float, default=22.0)
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--deploy-delay", type=int, default=0,
                   help="minutes the fire grows before the fleet deploys (1 tick = 1 min)")
    p.add_argument("--pattern", choices=["web", "grid"], default="web",
                   help="containment geometry: spider web (rings+spokes, defense in depth) or rotated grid")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="docs/figures/realistic.mp4")
    p.add_argument("--fps", type=int, default=18)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    global LINE_KW
    C.LINE_STRENGTH = LINE_STRENGTH
    tot = args.grid ** 2

    print("baseline (no fleet) ...")
    ab0, na, tb0, *_ = run(args.grid, args.attack_trucks, args.defender_trucks,
                           args.dpt_attack, args.dpt_def, args.assets, args.fires,
                           args.compartment, args.ticks, args.seed, use_fleet=False)
    print(f"  assets {ab0}/{na} ({ab0/max(na,1):.0%})  total {tb0} ({tb0/tot:.0%})")
    print(f"combined fleet (deploy delay {args.deploy_delay} min) ...")
    ab, na, tb, frames, assets, ele, depots, na_t = run(
        args.grid, args.attack_trucks, args.defender_trucks, args.dpt_attack,
        args.dpt_def, args.assets, args.fires, args.compartment, args.ticks,
        args.seed, use_fleet=True, record=True, deploy_delay=args.deploy_delay,
        pattern=args.pattern)
    print(f"  assets {ab}/{na} ({ab/max(na,1):.0%})  total {tb} ({tb/tot:.0%})  "
          f"reduction {(tb0-tb)/max(tb0,1):+.0%}")
    status = (f"deploy delay {args.deploy_delay} min  |  baseline {tb0/tot:.0%} grid  |  "
              f"fleet {ab/na:.0%} assets / {tb/tot:.0%} grid")
    render(frames, args.grid, assets, ele, depots, na_t, args.out, args.fps, status)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
