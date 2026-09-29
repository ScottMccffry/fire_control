#!/usr/bin/env python3
"""
Closed-loop drone-swarm suppression coupled to a live WRF-SFIRE fire.

WRF-SFIRE has no Python stepping API, so coupling is done by RESTART CYCLING:
run WRF for one interval, stop, read the real fire, run the trained per-drone
policy to position the swarm, then ZERO THE RATE-OF-SPREAD coefficients (R_0,
BBB, PHIWC, FGIP) at the cells where drones drop water -- a firebreak WRF then
cannot spread through -- write it into the restart, and continue. The fire WRF
computes next genuinely responds to the drones.

NOTE on the lever: editing FMC_G (fuel moisture) or NFUEL_CAT (fuel category)
in the restart has NO effect here -- this case uses constant fuel moisture and
SFIRE computes spread from precomputed per-cell coefficients (R_0, ...) stored
as restart state. Zeroing those is what actually stops spread (verified by A/B).

Runs two arms with identical numerics and compares burned area:
  * baseline : restart-cycled, no edits (no swarm)
  * drones   : restart-cycled, swarm zeroes spread coefficients where it drops

Usage:
  python scripts/wrf_closed_loop.py --src /opt/wrf_cl \
      --policy agents/checkpoints/per_drone_ppo_large_g100_d40 \
      --interval 60 --total 600 --drones 60 --out-dir /opt/wrf_loop
"""

import sys
import argparse
import shutil
import subprocess
import logging
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# SFIRE computes the rate of spread from precomputed per-cell coefficients
# (R_0 = base ROS, FGIP = fuel load/heat) stored as restart state -- not from
# NFUEL_CAT/FMC_G each step (verified by A/B). So suppression scales R_0/FGIP.
#
# Calibrated (graded, water-limited) model instead of a perfect firebreak:
# a drop delivers the drone's water over a small footprint; the local water
# loading w [L/m^2] reduces the rate of spread by exp(-w / W_E). Suppression
# ACCUMULATES across cycles (water persists), and drones have a finite tank that
# refills slowly -- so a single 10 L drop only dents ROS; sustained, coordinated
# drops are needed to contain the front.
SUPPRESS_FIELDS = ("R_0", "FGIP")   # base rate-of-spread and fuel load/heat
SUPPRESS_RADIUS = 1                  # drop footprint: 3x3 fire cells (~37 m)
CELL_AREA_M2 = 12.5 ** 2            # fire-mesh cell (dx/sr = 50/4 = 12.5 m)
W_E = 0.05                           # L/m^2 e-folding suppression effectiveness
MICRO_STEPS = 40                    # policy micro-steps per coupling interval
ACTIVE_W = 300.0                    # W/m^2 FGRNHFX threshold for "active fire"


def set_times(namelist: Path, start_s: int, end_s: int, restart: bool, interval: int):
    """Rewrite start/end minute:second, restart flag, and restart interval."""
    txt = namelist.read_text().splitlines()
    sm, ss = start_s // 60, start_s % 60
    em, es = end_s // 60, end_s % 60
    repl = {
        " start_minute": f" start_minute = {sm:02d},",
        " start_second": f" start_second = {ss:02d},",
        " end_minute": f" end_minute = {em:02d},",
        " end_second": f" end_second = {es:02d},",
        " run_minutes": f" run_minutes = {(end_s-start_s)//60},",
        " run_seconds": f" run_seconds = {(end_s-start_s)%60},",
        " restart ": f" restart = {'.true.' if restart else '.false.'},",
        " restart_interval_s": f" restart_interval_s = {interval},",
    }
    out = []
    for line in txt:
        key = next((k for k in repl if line.startswith(k) and "=" in line), None)
        out.append(repl[key] if key else line)
    namelist.write_text("\n".join(out) + "\n")


def rst_name(t_s: int) -> str:
    return f"wrfrst_d01_0001-01-01_00:{t_s//60:02d}:{t_s%60:02d}"


def run_wrf(run_dir: Path, log: Path) -> bool:
    with open(log, "w") as f:
        subprocess.run(["./wrf.exe"], cwd=run_dir, stdout=f, stderr=subprocess.STDOUT)
    return "SUCCESS COMPLETE WRF" in log.read_text()


def suppress_in_restart(rst_path: Path, controller, micro: int):
    """Read fire from restart, advance the swarm, apply graded water suppression."""
    import netCDF4 as nc
    with nc.Dataset(rst_path, "r+") as ds:
        heat = ds.variables["FGRNHFX"][0].astype(float)   # (sn_sub, we_sub)
        drops = controller.step(heat, micro)              # {(r,c): liters delivered}
        R = SUPPRESS_RADIUS
        H, W = heat.shape
        arrs = {v: ds.variables[v][0] for v in SUPPRESS_FIELDS if v in ds.variables}
        footprint = (2 * R + 1) ** 2 * CELL_AREA_M2
        total_l = 0.0
        for (r, c), liters in drops.items():
            total_l += liters
            w = liters / footprint                    # L/m^2 over the drop footprint
            mult = float(np.exp(-w / W_E))            # graded ROS/heat reduction
            r0, r1 = max(0, r - R), min(H, r + R + 1)
            c0, c1 = max(0, c - R), min(W, c + R + 1)
            for a in arrs.values():
                a[r0:r1, c0:c1] *= mult              # accumulates across cycles
        for v, a in arrs.items():
            ds.variables[v][0] = a
        return len(drops), int((heat > ACTIVE_W).sum()), total_l


def burned_cells(rst_path: Path) -> int:
    import netCDF4 as nc
    with nc.Dataset(rst_path) as ds:
        return int((ds.variables["FIRE_AREA"][0] > 0).sum())


class SwarmController:
    """Drives the trained per-drone policy on the WRF fire grid.

    Maintains a finite per-drone water budget across coupling cycles: a drone on
    an active cell with enough water drops its full tank (then refills slowly),
    so suppression is water-limited, not unlimited.
    """
    TANK = 10.0       # litres per drone
    DROP_MIN = 5.0    # only drop if at least this much water
    REFILL = 4.0      # litres regained per coupling cycle (slow resupply)

    def __init__(self, policy_path, n_drones, grid, max_heat):
        from stable_baselines3 import PPO
        from envs.per_drone_env import WRFGridDroneEnv
        self.model = PPO.load(policy_path)
        self.env = WRFGridDroneEnv(grid, n_drones, max_heat)
        self.pos = None
        self.water = np.full(n_drones, self.TANK)

    def step(self, heat, micro):
        env = self.env
        env.set_fire(heat)
        obs = env.reset_keep(self.pos)
        for _ in range(micro):        # position the swarm (no water logic)
            a, _ = self.model.predict(obs, deterministic=True)
            obs = env.move(a)
        self.pos = env.drone_pos.copy()

        thr = env.sim.ignition_threshold * 0.3
        drops = {}
        for i in range(env.n_drones):
            r, c = int(self.pos[i, 0]), int(self.pos[i, 1])
            if heat[r, c] > thr and self.water[i] >= self.DROP_MIN:
                drops[(r, c)] = drops.get((r, c), 0.0) + self.water[i]
                self.water[i] = 0.0
        self.water = np.minimum(self.TANK, self.water + self.REFILL)  # resupply
        return drops


class RealisticController:
    """Per-drone swarm with continuous kinematics + lake refuelling.

    Realistic upgrade over SwarmController:
      * continuous (float) positions integrated at a real speed (m/s),
      * drones drop at a real rate (L/s) until empty,
      * when empty, a drone flies to the NEAREST lake, refills over a fixed
        time, then resumes -- so suppression is limited by travel + refuel
        downtime, not free regeneration.

    `source` selects the heading each fire-fighting drone takes:
      * "greedy"  -> a true continuous vector toward its nearest active cell,
      * "policy"  -> the trained policy's cardinal action (no retrain; the
                     action is integrated at the real speed).
    Refuel travel always uses a true vector toward the assigned lake.
    """
    TANK = 10.0          # litres
    DROP_MIN = 0.5       # refuel once below this
    DROP_RATE = 5.0      # litres/second delivered while over fire
    REFILL_TIME = 30.0   # seconds to refill a tank at a lake
    MAX_SPEED = 15.0     # m/s
    ACTIVE_W = 300.0

    def __init__(self, source, policy_path, n_drones, grid, max_heat,
                 cell_m=12.5, dt=1.5, lake_spacing=82):
        from envs.per_drone_env import WRFGridDroneEnv, _MOVES
        self.source = source
        self.model = None
        if source == "policy":
            from stable_baselines3 import PPO
            self.model = PPO.load(policy_path)
        self.env = WRFGridDroneEnv(grid, n_drones, max_heat)
        self._MOVES = _MOVES.astype(float)
        self.n = n_drones
        self.g = grid
        self.cell_m = cell_m
        self.dt = dt
        self.step_cells = self.MAX_SPEED * dt / cell_m          # cells / micro-step
        self.refill_per_step = self.TANK / self.REFILL_TIME * dt
        self.drop_per_step = self.DROP_RATE * dt
        # Lakes on a regular grid across the domain.
        gs = np.arange(lake_spacing // 2, grid, lake_spacing)
        self.lakes = np.array([(r, c) for r in gs for c in gs], dtype=float)
        self.pos = None
        self.water = np.full(n_drones, self.TANK)
        self.mode = np.zeros(n_drones, dtype=int)               # 0 fight, 1 refuel
        self.lake_target = np.zeros((n_drones, 2))
        self.record = False                                     # log per-micro-step
        self.frames = []                                        # (pos, mode) snapshots

    def _init_pos(self):
        per_row = int(np.ceil(np.sqrt(self.n)))
        pos = np.zeros((self.n, 2))
        for i in range(self.n):
            gr, gc = divmod(i, per_row)
            pos[i] = ((gr + 0.5) * self.g / per_row, (gc + 0.5) * self.g / per_row)
        return np.clip(pos, 0, self.g - 1)

    def step(self, heat, micro):
        env = self.env
        env.set_fire(heat)
        if self.pos is None:
            self.pos = self._init_pos()
        thr = self.ACTIVE_W
        drops = {}
        for _ in range(micro):
            ipos = np.clip(np.round(self.pos).astype(int), 0, self.g - 1)
            env.drone_pos = ipos
            obs = env._build_obs()                 # refresh fire-bearing caches
            # --- heading for fire-fighting drones ---
            if self.source == "policy":
                a, _ = self.model.predict(obs, deterministic=True)
                fdir = self._MOVES[np.asarray(a)]            # cardinal (unit/zero)
            else:                                            # greedy: true vector to fire
                fdir = np.stack([env._fire_dr, env._fire_dc], axis=1).astype(float)
                fn = np.linalg.norm(fdir, axis=1, keepdims=True)
                fdir = np.where(fn > 0, fdir / fn, 0.0)
            # --- heading for refuelling drones: true vector to assigned lake ---
            to_lake = self.lake_target - self.pos
            ln = np.linalg.norm(to_lake, axis=1, keepdims=True)
            rdir = np.where(ln > 1e-6, to_lake / ln, 0.0)
            direction = np.where(self.mode[:, None] == 1, rdir, fdir)
            self.pos = np.clip(self.pos + direction * self.step_cells, 0, self.g - 1)

            ipos = np.clip(np.round(self.pos).astype(int), 0, self.g - 1)
            # --- refuel state machine ---
            at_lake = (self.mode == 1) & (
                np.linalg.norm(self.lake_target - self.pos, axis=1) < 1.5)
            self.water[at_lake] = np.minimum(self.TANK,
                                             self.water[at_lake] + self.refill_per_step)
            done = (self.mode == 1) & (self.water >= self.TANK - 1e-6)
            self.mode[done] = 0
            need = (self.mode == 0) & (self.water < self.DROP_MIN)
            if need.any():
                d = np.linalg.norm(self.pos[need, None, :] - self.lakes[None, :, :], axis=2)
                self.lake_target[need] = self.lakes[d.argmin(axis=1)]
                self.mode[need] = 1
            # --- drops: fighting drones over active fire deliver water ---
            fight = self.mode == 0
            on_fire = heat[ipos[:, 0], ipos[:, 1]] > thr
            can = fight & on_fire & (self.water > 0)
            if can.any():
                deliver = np.minimum(self.water[can], self.drop_per_step)
                self.water[can] -= deliver
                for k, i in enumerate(np.where(can)[0]):
                    cell = (int(ipos[i, 0]), int(ipos[i, 1]))
                    drops[cell] = drops.get(cell, 0.0) + float(deliver[k])
            if self.record:
                self.frames.append((self.pos.copy().astype("float32"),
                                    self.mode.copy().astype("int8")))
        return drops


def make_controller(arm, args):
    """Build the swarm controller for an arm ('baseline' -> None)."""
    if arm == "baseline":
        return None
    source = "greedy" if arm == "greedy" else "policy"
    if args.realistic:
        return RealisticController(source, args.policy, args.drones, 412, 35000.0,
                                   lake_spacing=args.lake_spacing)
    return SwarmController(args.policy, args.drones, 412, 35000.0)  # policy only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/opt/wrf_cl", help="dir with exes/tables/namelist")
    p.add_argument("--policy", default="agents/checkpoints/per_drone_ppo_large_g100_d40")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--total", type=int, default=600)
    p.add_argument("--drones", type=int, default=60)
    p.add_argument("--arms", default="baseline,greedy,policy",
                   help="comma list of arms: baseline, greedy, policy")
    p.add_argument("--realistic", action="store_true",
                   help="continuous kinematics (15 m/s) + lake refuelling")
    p.add_argument("--lake-spacing", type=int, default=82, help="fire cells between lakes")
    p.add_argument("--out-dir", default="/opt/wrf_loop")
    args = p.parse_args()
    logging.disable(logging.CRITICAL)

    src = Path(args.src)
    n_cycles = args.total // args.interval
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = {}

    for arm in arms:
        run_dir = Path(args.out_dir) / arm
        run_dir.mkdir(parents=True, exist_ok=True)
        for f in ["ideal.exe", "wrf.exe", "input_sounding", "namelist.fire",
                  "ETAMPNEW_DATA", "LANDUSE.TBL", "RRTMG_SW_DATA", "URBPARM.TBL",
                  "GENPARM.TBL", "RRTMG_LW_DATA", "SOILPARM.TBL", "VEGPARM.TBL",
                  "RRTM_DATA"]:
            tgt = run_dir / f
            if not tgt.exists():
                tgt.symlink_to((src / f).resolve())
        shutil.copy(src / "namelist.input", run_dir / "namelist.input")
        nml = run_dir / "namelist.input"

        controller = make_controller(arm, args)

        set_times(nml, 0, args.interval, restart=False, interval=args.interval)
        subprocess.run(["./ideal.exe"], cwd=run_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = run_wrf(run_dir, run_dir / "c0.log")
        print(f"[{arm}] cycle 0 (0-{args.interval}s): {'OK' if ok else 'FAIL'}", flush=True)

        for k in range(1, n_cycles):
            t0 = k * args.interval
            rst = run_dir / rst_name(t0)
            if controller is not None and rst.exists():
                nd, na, lit = suppress_in_restart(rst, controller, MICRO_STEPS)
                print(f"[{arm}] cycle {k}: {na} active cells, {nd} drops, {lit:.0f} L",
                      flush=True)
            set_times(nml, t0, t0 + args.interval, restart=True, interval=args.interval)
            ok = run_wrf(run_dir, run_dir / f"c{k}.log")
            if not ok:
                print(f"[{arm}] cycle {k} FAILED"); break

        final_rst = run_dir / rst_name(args.total)
        results[arm] = burned_cells(final_rst) if final_rst.exists() else -1
        print(f"[{arm}] final burned cells: {results[arm]}", flush=True)

    print("\n" + "=" * 56)
    print(f"CLOSED-LOOP RESULT (final burned fire cells){'  [realistic]' if args.realistic else ''}")
    base = results.get("baseline", -1)
    for arm in arms:
        v = results.get(arm, -1)
        red = f"  ({(base-v)/base:+.0%} vs baseline)" if base > 0 and arm != "baseline" and v >= 0 else ""
        print(f"  {arm:10s}: {v}{red}")


if __name__ == "__main__":
    main()
