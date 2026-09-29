#!/usr/bin/env python3
"""
Run the trained per-drone swarm policy on REAL WRF-SFIRE fire frames.

Connects the two halves of the project: the decentralized per-drone policy
(trained on the mock fire) is evaluated on genuine WRF-SFIRE output produced by
a real compiled run (see docs/wrf_sfire.md). The WRFReplaySimulator frames drive
the fire; the swarm observes and tracks the real flame front.

This is OPEN-LOOP (replay): drones cannot change a precomputed WRF fire, so the
metric is flame-front COVERAGE -- the fraction of active fire cells within a
drone's suppression radius -- not burned-area reduction. Closed-loop suppression
would require running WRF-SFIRE live with drone water fed back each step.

Usage:
    python scripts/evaluate_on_wrf.py --wrfout-dir /opt/wrf_fine \
        --policy agents/checkpoints/per_drone_ppo_large_g100_d40 \
        --coarsen 4 --drones 40 --out wrf_swarm.png
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim.wrf_replay import WRFReplaySimulator           # noqa: E402
from envs.per_drone_env import PerDroneSwarmVecEnv, _MOVES  # noqa: E402


class _FireShim:
    """Minimal MockFireSimulator-like surface backed by a fixed WRF frame."""
    def __init__(self, frames, burned, max_heat, ignition_threshold=1000.0):
        self.frames = frames
        self.burned = burned
        self.max_heat_intensity = float(max_heat)
        self.ignition_threshold = ignition_threshold
        self.idx = 0

    @property
    def heat_intensity(self):
        return self.frames[self.idx]

    @property
    def burned_area(self):
        return self.burned[self.idx]

    def apply_water_drop(self, r, c, amount=5.0):
        pass  # replay is one-way; drops cannot alter a precomputed WRF fire

    def advance(self):
        self.idx = min(self.idx + 1, len(self.frames) - 1)


class WRFReplayDroneVecEnv(PerDroneSwarmVecEnv):
    """Per-drone swarm whose fire is a sequence of real WRF-SFIRE frames."""
    def __init__(self, frames, burned, max_heat, n_drones=40, hold=4,
                 cover_radius=2):
        self._frames = frames
        self._burned = burned
        self._max_heat = max_heat
        self._hold = hold
        self.cover_radius = cover_radius
        grid = frames[0].shape[0]
        super().__init__(grid_size=grid, n_drones=n_drones,
                         max_steps=len(frames) * hold)

    def _new_episode(self):
        self.sim = _FireShim(self._frames, self._burned, self._max_heat)
        per_row = int(np.ceil(np.sqrt(self.n_drones)))
        for i in range(self.n_drones):
            gr, gc = divmod(i, per_row)
            self.drone_pos[i] = (int((gr + 0.5) * self.grid / per_row),
                                 int((gc + 0.5) * self.grid / per_row))
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid - 1)
        self.drone_water[:] = self.water_capacity
        self.steps = 0

    def coverage(self):
        """Fraction of active fire cells within cover_radius of any drone."""
        thr = self.sim.ignition_threshold * 0.3
        ar, ac = np.where(self.sim.heat_intensity > thr)
        if len(ar) == 0:
            return None
        R = self.cover_radius
        covered = 0
        dp = self.drone_pos
        for r, c in zip(ar, ac):
            if np.any((np.abs(dp[:, 0] - r) <= R) & (np.abs(dp[:, 1] - c) <= R)):
                covered += 1
        return covered / len(ar)

    def step_wait(self):
        self.drone_pos = np.clip(self.drone_pos + _MOVES[self._actions],
                                 0, self.grid - 1)
        self.steps += 1
        if self.steps % self._hold == 0:
            self.sim.advance()
        obs = self._build_obs()           # refreshes fire-bearing caches
        cov = self.coverage()
        done = self.steps >= self.max_steps
        infos = [{} for _ in range(self.n_drones)]
        if cov is not None:
            infos[0]["coverage"] = cov
        dones = np.full(self.n_drones, done, dtype=bool)
        return obs, np.zeros(self.n_drones, np.float32), dones, infos


def load_wrf_frames(wrfout_dir, coarsen):
    sim = WRFReplaySimulator(wrfout_dir, coarsen=coarsen, fire_mesh_res=12.5)
    frames, burned = [], []
    fs = sim.step()
    while fs is not None:
        ff = fs.current_flame_front
        frames.append(ff.heat_intensity.astype(np.float32))
        burned.append(ff.burned_area.astype(int))
        fs = sim.step()
    max_heat = max(1.0, max(f.max() for f in frames))
    return frames, burned, max_heat


def run(env, kind, model=None):
    obs = env.reset()
    covs = []
    while True:
        if kind == "random":
            actions = np.random.randint(0, 5, env.n_drones)
        elif kind == "greedy":
            actions = env.greedy_actions()
        else:
            actions, _ = model.predict(obs, deterministic=True)
        env.step_async(actions)
        obs, _, dones, infos = env.step_wait()
        if "coverage" in infos[0]:
            covs.append(infos[0]["coverage"])
        if dones[0]:
            return float(np.mean(covs)) if covs else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wrfout-dir", default="/opt/wrf_fine")
    p.add_argument("--policy", default="agents/checkpoints/per_drone_ppo_large_g100_d40")
    p.add_argument("--coarsen", type=int, default=4)
    p.add_argument("--drones", type=int, default=40)
    p.add_argument("--hold", type=int, default=4)
    p.add_argument("--out", default="wrf_swarm.png")
    args = p.parse_args()

    logging.disable(logging.CRITICAL)
    from stable_baselines3 import PPO

    frames, burned, max_heat = load_wrf_frames(args.wrfout_dir, args.coarsen)
    grid = frames[0].shape[0]
    active_frames = sum(1 for f in frames if (f > 300).any())
    print(f"Loaded {len(frames)} real WRF-SFIRE frames @ {grid}x{grid} "
          f"({active_frames} with active fire); peak heat {max_heat/1000:.1f} kW/m^2")

    model = PPO.load(args.policy)
    print(f"Policy: {args.policy}\n")

    print(f"{'controller':<10}{'mean flame-front coverage':>28}")
    results = {}
    for kind in ("random", "greedy", "trained"):
        env = WRFReplayDroneVecEnv(frames, burned, max_heat, n_drones=args.drones,
                                   hold=args.hold)
        results[kind] = run(env, kind, model)
        print(f"{kind:<10}{results[kind]:>27.0%}")

    # Visualization: real WRF fire + trained-drone positions at 3 timepoints.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        env = WRFReplayDroneVecEnv(frames, burned, max_heat, n_drones=args.drones,
                                   hold=args.hold)
        obs = env.reset()
        snaps = {}
        targets = [int(len(frames) * args.hold * f) for f in (0.45, 0.7, 0.98)]
        while True:
            actions, _ = model.predict(obs, deterministic=True)
            env.step_async(actions)
            obs, _, dones, _ = env.step_wait()
            if env.steps in targets:
                snaps[env.steps] = (env.sim.heat_intensity.copy(),
                                    env.drone_pos.copy())
            if dones[0]:
                break
        fig, axes = plt.subplots(1, len(snaps), figsize=(4 * len(snaps), 4))
        if len(snaps) == 1:
            axes = [axes]
        for ax, (step, (heat, dp)) in zip(axes, sorted(snaps.items())):
            ax.imshow(heat / 1000.0, origin="lower", cmap="inferno",
                      vmin=0, vmax=max_heat / 1000.0)
            ax.scatter(dp[:, 1], dp[:, 0], s=14, c="cyan",
                       edgecolors="white", linewidths=0.4, label="drones")
            ax.set_title(f"sim frame {env.sim.idx} / {len(frames)-1}")
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle("Trained per-drone swarm tracking a REAL WRF-SFIRE fire "
                     "(ground heat flux, kW/m^2)", fontsize=11)
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\nSaved figure: {args.out}")
    except Exception as e:
        print(f"\n(visualization skipped: {e})")


if __name__ == "__main__":
    main()
