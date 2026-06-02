"""
Gymnasium environment for RL-based drone swarm wildfire suppression.

This environment wraps the repository's own fire physics (``MockFireSimulator``)
and exposes a tractable single-policy control problem: a small team of drones is
driven over a grid to suppress an actively spreading fire. It is intentionally
scaled down from the full 5,000-10,000 drone vision in the README so that an RL
agent (e.g. PPO/DQN via stable-baselines3) can be trained quickly on CPU while
still exercising the real spread/suppression dynamics defined in ``sim``.

Design summary
--------------
* The agent is a centralized controller for ``n_drones`` drones.
* Each drone occupies a single grid cell and carries a finite water tank that
  depletes on a drop and slowly regenerates (an abstraction of refilling).
* A drone automatically drops water on its current cell when that cell is
  actively burning and it has water remaining, using the simulator's own
  ``apply_water_drop`` so suppression behaves exactly as in the heuristic runs.
* Action: per-drone movement (stay / up / down / left / right).
* Observation: the (normalized) heat map plus each drone's position and water.
* Reward: penalizes newly burned cells and standing fire, rewards effective
  water drops, so the policy learns to get ahead of the flame front.
"""

import random
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
import gymnasium as gym
from gymnasium import spaces

from sim import MockFireSimulator


# Movement deltas indexed by action id: stay, up, down, left, right
_MOVES = np.array(
    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    dtype=np.int64,
)


class DroneFireEnv(gym.Env):
    """A scaled-down, RL-ready drone-swarm wildfire suppression environment."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 20,
        n_drones: int = 6,
        max_steps: int = 120,
        n_ignitions: int = 2,
        water_capacity: float = 10.0,
        water_regen_per_step: float = 1.5,
        drop_amount: float = 5.0,
        wind_speed: float = 6.0,
        wind_direction: float = 45.0,
        # Fire-regime "difficulty" controls. The defaults reproduce the
        # MockFireSimulator's own random ranges; harder presets use drier,
        # denser fuel and a faster base spread rate so the fire becomes a
        # sustained, spreading threat that drones must actively contain
        # (otherwise it self-extinguishes and there is nothing to learn).
        fuel_moisture_range: Tuple[float, float] = (0.1, 0.4),
        fuel_load_range: Tuple[float, float] = (0.5, 1.0),
        base_spread_rate: float = 0.1,
        engagement_coef: float = 0.1,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.grid = int(grid_size)
        self.n_drones = int(n_drones)
        self.max_steps = int(max_steps)
        self.n_ignitions = int(n_ignitions)
        self.water_capacity = float(water_capacity)
        self.water_regen = float(water_regen_per_step)
        self.drop_amount = float(drop_amount)
        self.wind_speed = float(wind_speed)
        self.wind_direction = float(wind_direction)
        self.fuel_moisture_range = tuple(fuel_moisture_range)
        self.fuel_load_range = tuple(fuel_load_range)
        self.base_spread_rate = float(base_spread_rate)
        self.engagement_coef = float(engagement_coef)
        self.render_mode = render_mode

        # Per-drone discrete movement -> MultiDiscrete([5, 5, ...])
        self.action_space = spaces.MultiDiscrete([len(_MOVES)] * self.n_drones)

        # Observation: flattened normalized heat map + (row, col, water) per drone
        obs_dim = self.grid * self.grid + 3 * self.n_drones
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # State, populated on reset()
        self.sim: Optional[MockFireSimulator] = None
        self.drone_pos = np.zeros((self.n_drones, 2), dtype=np.int64)
        self.drone_water = np.full(self.n_drones, self.water_capacity, dtype=np.float32)
        self.steps = 0
        self._prev_burned = 0

        # RNG
        self._np_random = np.random.default_rng(seed)

    # ------------------------------------------------------------------ utils
    def _build_observation(self) -> np.ndarray:
        heat = self.sim.heat_intensity / self.sim.max_heat_intensity
        heat = np.clip(heat, 0.0, 1.0).astype(np.float32).ravel()

        drones = np.empty(3 * self.n_drones, dtype=np.float32)
        for i in range(self.n_drones):
            drones[3 * i + 0] = self.drone_pos[i, 0] / max(1, self.grid - 1)
            drones[3 * i + 1] = self.drone_pos[i, 1] / max(1, self.grid - 1)
            drones[3 * i + 2] = self.drone_water[i] / self.water_capacity

        return np.concatenate([heat, drones]).astype(np.float32)

    def _random_ignitions(self):
        pts = []
        margin = max(1, self.grid // 5)
        for _ in range(self.n_ignitions):
            r = int(self._np_random.integers(margin, self.grid - margin))
            c = int(self._np_random.integers(margin, self.grid - margin))
            pts.append((r, c))
        return pts

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            # MockFireSimulator's spread uses Python's global RNG, so seed it
            # too. This makes whole episodes reproducible, which is essential
            # for *paired* policy comparison (same fire, different controller).
            random.seed(seed)

        self.sim = MockFireSimulator(
            grid_size=(self.grid, self.grid),
            cell_size_meters=100.0,
            bounds=(34.0, -119.0, 35.0, -118.0),
        )
        self.sim.set_weather(
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            temperature=30.0,
            humidity=0.2,
        )

        # Apply the configured fire regime, overriding the simulator's default
        # random fuel fields and base spread rate. Drier/denser fuel and a
        # higher spread rate make the fire sustained and spreading.
        lo, hi = self.fuel_moisture_range
        self.sim.fuel_moisture = self._np_random.uniform(lo, hi, (self.grid, self.grid))
        lo, hi = self.fuel_load_range
        self.sim.fuel_load = self._np_random.uniform(lo, hi, (self.grid, self.grid))
        self.sim.base_spread_rate = self.base_spread_rate

        self.sim.set_ignition_points(self._random_ignitions())

        # Pre-deploy drones in a roughly square grid pattern across the whole
        # domain for coverage, so the swarm can reach a fire anywhere on the map.
        self.drone_pos = np.zeros((self.n_drones, 2), dtype=np.int64)
        per_row = int(np.ceil(np.sqrt(self.n_drones)))
        for i in range(self.n_drones):
            gr, gc = divmod(i, per_row)
            self.drone_pos[i, 0] = int((gr + 0.5) * self.grid / per_row)
            self.drone_pos[i, 1] = int((gc + 0.5) * self.grid / per_row)
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid - 1)
        self.drone_water = np.full(self.n_drones, self.water_capacity, dtype=np.float32)

        self.steps = 0
        self._prev_burned = int(self.sim.burned_area.sum())

        return self._build_observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(self.n_drones)

        # 1) Move drones (clamped to grid).
        deltas = _MOVES[action]
        self.drone_pos = np.clip(
            self.drone_pos + deltas, 0, self.grid - 1
        ).astype(np.int64)

        # 2) Drones automatically drop water on actively burning cells.
        drops = 0
        suppressed_heat = 0.0
        for i in range(self.n_drones):
            r, c = int(self.drone_pos[i, 0]), int(self.drone_pos[i, 1])
            burning = self.sim.heat_intensity[r, c] > self.sim.ignition_threshold * 0.3
            if burning and self.drone_water[i] >= self.drop_amount:
                before = float(self.sim.heat_intensity[r, c])
                self.sim.apply_water_drop(r, c, water_amount=self.drop_amount)
                after = float(self.sim.heat_intensity[r, c])
                suppressed_heat += max(0.0, before - after)
                self.drone_water[i] -= self.drop_amount
                drops += 1
            # Slow refill (abstraction of returning to a lake/base).
            self.drone_water[i] = min(
                self.water_capacity, self.drone_water[i] + self.water_regen
            )

        # 3) Advance the fire one step.
        self.sim.step()
        self.steps += 1

        # 4) Reward shaping.
        #
        # The true objective is to minimize burned area, but that signal is
        # sparse and dominated by natural fire spread. Earlier shaping that
        # rewarded mere *proximity* to the fire was gamed by the policy (hover
        # next to the flame front without ever dropping water). So the dense
        # terms here reward *actual suppression* — water drops and heat removed,
        # which can only happen when a drone stands on a burning cell with water
        # — plus productive positioning (drones on active cells). A small
        # proximity term (strictly weaker than the on-fire/drop terms) aids
        # exploration on large grids without being gameable.
        burned = int(self.sim.burned_area.sum())
        new_burned = max(0, burned - self._prev_burned)
        self._prev_burned = burned

        thr = self.sim.ignition_threshold * 0.3
        active_mask = self.sim.heat_intensity > thr
        active_cells = int(active_mask.sum())

        rows = self.drone_pos[:, 0]
        cols = self.drone_pos[:, 1]
        on_fire = int(active_mask[rows, cols].sum())

        # Cheap exploration nudge: closeness to the nearest active cell via an
        # exact Euclidean distance transform (scales to large grids).
        proximity = 0.0
        if active_cells > 0:
            dist_map = ndimage.distance_transform_edt(~active_mask)
            d = dist_map[rows, cols] / self.grid
            proximity = float(np.exp(-5.0 * d).mean())

        reward = (
            -1.0 * new_burned
            - 0.01 * active_cells
            + 1.0 * drops                                       # actual drop (on-fire + water)
            + 0.5 * (suppressed_heat / self.sim.max_heat_intensity)
            + 0.3 * (on_fire / self.n_drones)                   # productive positioning
            + self.engagement_coef * proximity                  # small exploration aid
        )

        fire_extinguished = active_cells == 0 and self.steps > 1
        terminated = bool(fire_extinguished)
        truncated = bool(self.steps >= self.max_steps)
        if fire_extinguished:
            # Bonus for putting the fire out, scaled by how much land was saved.
            reward += 50.0 * (1.0 - burned / (self.grid * self.grid))

        info = {
            "burned_cells": burned,
            "active_fire_cells": active_cells,
            "new_burned": new_burned,
            "water_drops": drops,
            "fire_extinguished": fire_extinguished,
        }

        return self._build_observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi" or self.sim is None:
            return None
        rows = []
        drone_cells = {(int(r), int(c)) for r, c in self.drone_pos}
        for r in range(self.grid):
            line = []
            for c in range(self.grid):
                if (r, c) in drone_cells:
                    line.append("D")
                elif self.sim.heat_intensity[r, c] > self.sim.ignition_threshold:
                    line.append("#")
                elif self.sim.burned_area[r, c]:
                    line.append(".")
                else:
                    line.append(" ")
            rows.append("".join(line))
        return "\n".join(rows)

    def close(self):
        self.sim = None
