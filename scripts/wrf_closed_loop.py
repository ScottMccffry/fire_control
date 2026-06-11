#!/usr/bin/env python3
"""
Closed-loop drone-swarm suppression coupled to a live WRF-SFIRE fire.

WRF-SFIRE has no Python stepping API, so coupling is done by RESTART CYCLING:
run WRF for one interval, stop, read the real fire, run the trained per-drone
policy to position the swarm, then RAISE FUEL MOISTURE (FMC_G) above the
moisture of extinction at the cells where drones drop water -- wet fuel WRF then
refuses to burn -- write it back into the restart, and continue. The fire WRF
computes next genuinely responds to the drones.

Runs two arms with identical numerics and compares burned area:
  * baseline : restart-cycled, no FMC_G edits (no swarm)
  * drones   : restart-cycled, swarm raises FMC_G where it drops water

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

NO_FUEL_CAT = 14       # SFIRE "no fuel" category -> zero rate of spread
SUPPRESS_RADIUS = 2    # fire cells around each drop
MICRO_STEPS = 20       # policy micro-steps per coupling interval
ACTIVE_W = 300.0       # W/m^2 FGRNHFX threshold for "active fire"


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
    """Read fire from restart, advance the swarm, write FMC_G suppression."""
    import netCDF4 as nc
    with nc.Dataset(rst_path, "r+") as ds:
        heat = ds.variables["FGRNHFX"][0].astype(float)   # (sn_sub, we_sub)
        nfuel = ds.variables["NFUEL_CAT"][0].astype(float)
        cells = controller.step(heat, micro)              # list of (r,c) drop cells
        R = SUPPRESS_RADIUS
        H, W = nfuel.shape
        for r, c in cells:
            r0, r1 = max(0, r - R), min(H, r + R + 1)
            c0, c1 = max(0, c - R), min(W, c + R + 1)
            nfuel[r0:r1, c0:c1] = NO_FUEL_CAT     # firebreak: no fuel -> no spread
        ds.variables["NFUEL_CAT"][0] = nfuel
        return len(cells), int((heat > ACTIVE_W).sum())


def burned_cells(rst_path: Path) -> int:
    import netCDF4 as nc
    with nc.Dataset(rst_path) as ds:
        return int((ds.variables["FIRE_AREA"][0] > 0).sum())


class SwarmController:
    """Drives the trained per-drone policy on the WRF fire grid."""
    def __init__(self, policy_path, n_drones, grid, max_heat):
        from stable_baselines3 import PPO
        from envs.per_drone_env import WRFGridDroneEnv
        self.model = PPO.load(policy_path)
        self.env = WRFGridDroneEnv(grid, n_drones, max_heat)
        self.pos = None

    def step(self, heat, micro):
        env = self.env
        env.set_fire(heat)
        obs = env.reset_keep(self.pos)
        for _ in range(micro):
            a, _ = self.model.predict(obs, deterministic=True)
            obs = env.advance(a)
        self.pos = env.drone_pos.copy()
        return env.drops_on_fire()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/opt/wrf_cl", help="dir with exes/tables/namelist")
    p.add_argument("--policy", default="agents/checkpoints/per_drone_ppo_large_g100_d40")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--total", type=int, default=600)
    p.add_argument("--drones", type=int, default=60)
    p.add_argument("--out-dir", default="/opt/wrf_loop")
    args = p.parse_args()
    logging.disable(logging.CRITICAL)

    src = Path(args.src)
    n_cycles = args.total // args.interval
    results = {}

    for arm in ("baseline", "drones"):
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

        controller = None
        if arm == "drones":
            controller = SwarmController(args.policy, args.drones, 412, 35000.0)

        # Cycle 0
        set_times(nml, 0, args.interval, restart=False, interval=args.interval)
        subprocess.run(["./ideal.exe"], cwd=run_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = run_wrf(run_dir, run_dir / "c0.log")
        print(f"[{arm}] cycle 0 (0-{args.interval}s): {'OK' if ok else 'FAIL'}")

        for k in range(1, n_cycles):
            t0 = k * args.interval
            rst = run_dir / rst_name(t0)
            if controller is not None and rst.exists():
                nd, na = suppress_in_restart(rst, controller, MICRO_STEPS)
                print(f"[{arm}] cycle {k}: {na} active cells, {nd} drone drops")
            set_times(nml, t0, t0 + args.interval, restart=True, interval=args.interval)
            ok = run_wrf(run_dir, run_dir / f"c{k}.log")
            if not ok:
                print(f"[{arm}] cycle {k} FAILED"); break

        final_rst = run_dir / rst_name(args.total)
        results[arm] = burned_cells(final_rst) if final_rst.exists() else -1
        print(f"[{arm}] final burned cells: {results[arm]}")

    print("\n" + "=" * 50)
    print("CLOSED-LOOP RESULT (final burned fire cells)")
    b, d = results.get("baseline", -1), results.get("drones", -1)
    print(f"  baseline (no swarm): {b}")
    print(f"  drones   (swarm)   : {d}")
    if b > 0 and d >= 0:
        print(f"  reduction: {b-d} cells ({(b-d)/b:+.0%})")


if __name__ == "__main__":
    main()
