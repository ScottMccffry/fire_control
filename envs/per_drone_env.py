"""
Per-drone (decentralized, parameter-shared) RL environment.

Instead of one centralized policy controlling the whole fleet, every drone runs
the SAME policy on its own egocentric observation. This is the architecture
that scales to very large swarms: the policy size is independent of fleet size,
and a drone's decision depends on
  * where the fire is (nearest-fire bearing, fire centroid, local heat patch),
  * where the other drones are (k-nearest-neighbour offsets),
  * its own position in the fleet (offset from the fleet centroid, and its
    rank: the fraction of the fleet closer to the fire than itself).

Training trick: the class implements stable-baselines3's VecEnv interface with
``num_envs = n_drones`` — each drone is one "environment" stream, all drones
share a single fire simulation, and PPO updates one shared policy from all
streams (parameter sharing). Every drone receives its own reward (own water
drops, spacing penalty, plus shared fire terms), which gives far better credit
assignment than a single fleet-level reward.
"""

from typing import List, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sim import MockFireSimulator

# Movement deltas indexed by action id: stay, up, down, left, right
_MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)
_PATCH = 5  # local heat window (5x5)
_KNN = 3    # nearest neighbour drones in the observation


class PerDroneSwarmVecEnv(VecEnv):
    """VecEnv where each 'env' is one drone of a shared-fire swarm."""

    def __init__(
        self,
        grid_size: int = 32,
        n_drones: int = 16,
        max_steps: int = 200,
        n_ignitions: int = 3,
        wind_speed: float = 9.0,
        fuel_moisture_range=(0.06, 0.22),
        fuel_load_range=(0.65, 1.0),
        base_spread_rate: float = 0.17,
        water_capacity: float = 10.0,
        water_regen_per_step: float = 1.5,
        drop_amount: float = 5.0,
        episode_seed: Optional[int] = None,
    ):
        self.grid = int(grid_size)
        self.n_drones = int(n_drones)
        self.max_steps = int(max_steps)
        self.n_ignitions = int(n_ignitions)
        self.wind_speed = float(wind_speed)
        self.fuel_moisture_range = tuple(fuel_moisture_range)
        self.fuel_load_range = tuple(fuel_load_range)
        self.base_spread_rate = float(base_spread_rate)
        self.water_capacity = float(water_capacity)
        self.water_regen = float(water_regen_per_step)
        self.drop_amount = float(drop_amount)
        # If set, every episode replays this seed (used for paired evaluation);
        # otherwise each episode draws a fresh random fire.
        self.episode_seed = episode_seed

        # own r,c (2) + water (1) + nearest-fire unit dx,dy,dist (3)
        # + fire centroid dx,dy + active fraction (3)
        # + fleet centroid dx,dy (2) + rank (1)
        # + KNN drone offsets (2*_KNN) + local heat patch (_PATCH^2)
        obs_dim = 2 + 1 + 3 + 3 + 2 + 1 + 2 * _KNN + _PATCH * _PATCH
        super().__init__(
            num_envs=self.n_drones,
            observation_space=spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
            action_space=spaces.Discrete(len(_MOVES)),
        )
        self.render_mode = None

        self._rng = np.random.default_rng(episode_seed)
        self._actions: Optional[np.ndarray] = None
        self._ep_returns = np.zeros(self.n_drones, dtype=np.float64)
        self.sim: Optional[MockFireSimulator] = None
        self.drone_pos = np.zeros((self.n_drones, 2), dtype=np.int64)
        self.drone_water = np.full(self.n_drones, self.water_capacity, np.float32)
        self.steps = 0
        self._prev_burned = 0
        # Cached fire geometry for observations / greedy actions
        self._fire_dr = np.zeros(self.n_drones)
        self._fire_dc = np.zeros(self.n_drones)
        self._fire_dist = np.full(self.n_drones, 1.0)

    # ----------------------------------------------------------- episode setup
    def _new_episode(self):
        seed = self.episode_seed
        if seed is not None:
            np.random.seed(seed)  # MockFireSimulator spread uses global RNG
            rng = np.random.default_rng(seed)
        else:
            rng = self._rng
        self.sim = MockFireSimulator(grid_size=(self.grid, self.grid),
                                     cell_size_meters=100.0)
        self.sim.set_weather(wind_speed=self.wind_speed, wind_direction=45.0,
                             temperature=30.0, humidity=0.2)
        lo, hi = self.fuel_moisture_range
        self.sim.fuel_moisture = rng.uniform(lo, hi, (self.grid, self.grid))
        lo, hi = self.fuel_load_range
        self.sim.fuel_load = rng.uniform(lo, hi, (self.grid, self.grid))
        self.sim.base_spread_rate = self.base_spread_rate
        margin = max(1, self.grid // 5)
        pts = [(int(rng.integers(margin, self.grid - margin)),
                int(rng.integers(margin, self.grid - margin)))
               for _ in range(self.n_ignitions)]
        self.sim.set_ignition_points(pts)

        per_row = int(np.ceil(np.sqrt(self.n_drones)))
        for i in range(self.n_drones):
            gr, gc = divmod(i, per_row)
            self.drone_pos[i] = (int((gr + 0.5) * self.grid / per_row),
                                 int((gc + 0.5) * self.grid / per_row))
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid - 1)
        self.drone_water[:] = self.water_capacity
        self.steps = 0
        self._prev_burned = int(self.sim.burned_area.sum())
        self._ep_returns[:] = 0.0

    # ----------------------------------------------------------- observations
    def _build_obs(self) -> np.ndarray:
        g = self.grid
        heat = np.clip(self.sim.heat_intensity / self.sim.max_heat_intensity, 0, 1)
        thr = self.sim.ignition_threshold * 0.3
        active = self.sim.heat_intensity > thr
        r = self.drone_pos[:, 0]
        c = self.drone_pos[:, 1]

        obs = np.zeros((self.n_drones, self.observation_space.shape[0]), np.float32)
        obs[:, 0] = r / (g - 1) * 2 - 1
        obs[:, 1] = c / (g - 1) * 2 - 1
        obs[:, 2] = self.drone_water / self.water_capacity * 2 - 1

        if active.any():
            dist_map, (ir, ic) = ndimage.distance_transform_edt(
                ~active, return_indices=True)
            dr = ir[r, c] - r
            dc = ic[r, c] - c
            dist = np.sqrt(dr * dr + dc * dc)
            norm = np.maximum(dist, 1e-6)
            self._fire_dr, self._fire_dc, self._fire_dist = dr, dc, dist / g
            obs[:, 3] = dr / norm
            obs[:, 4] = dc / norm
            obs[:, 5] = np.clip(dist / g, 0, 1)
            ar, ac = np.where(active)
            obs[:, 6] = np.clip((ar.mean() - r) / g, -1, 1)
            obs[:, 7] = np.clip((ac.mean() - c) / g, -1, 1)
            obs[:, 8] = np.clip(active.sum() / (g * g) * 10, 0, 1)
            # Rank: fraction of the fleet strictly closer to the fire than me.
            d = dist_map[r, c]
            obs[:, 9] = (d[:, None] > d[None, :]).mean(axis=1) * 2 - 1
        else:
            self._fire_dist = np.full(self.n_drones, 1.0)

        # Fleet centroid offset (my position relative to the fleet as a whole).
        obs[:, 10] = np.clip((self.drone_pos[:, 0].mean() - r) / g, -1, 1)
        obs[:, 11] = np.clip((self.drone_pos[:, 1].mean() - c) / g, -1, 1)

        # K nearest neighbour drone offsets (KD-tree: O(n log n), exact Manhattan
        # query, p=1 -- identical neighbours to the old O(n^2) argsort but scales
        # to thousands of drones).
        kq = min(_KNN + 1, self.n_drones)
        tree = cKDTree(self.drone_pos)
        nn_d, nn_i = tree.query(self.drone_pos, k=kq, p=1)
        if nn_i.ndim == 1:           # n_drones == 1 edge case
            nn_i = nn_i[:, None]; nn_d = nn_d[:, None]
        nn = nn_i[:, 1:]             # drop self (nearest, distance 0)
        for k in range(_KNN):
            j = nn[:, k] if k < nn.shape[1] else np.arange(self.n_drones)
            obs[:, 12 + 2 * k] = np.clip((self.drone_pos[j, 0] - r) / g, -1, 1)
            obs[:, 13 + 2 * k] = np.clip((self.drone_pos[j, 1] - c) / g, -1, 1)
        self._nn_dist = nn_d[:, 1] if nn_d.shape[1] > 1 else np.full(self.n_drones, g)

        # Local heat patch.
        pad = _PATCH // 2
        hp = np.pad(heat, pad)
        base = 12 + 2 * _KNN
        for i in range(self.n_drones):
            patch = hp[r[i]:r[i] + _PATCH, c[i]:c[i] + _PATCH]
            obs[i, base:base + _PATCH * _PATCH] = patch.ravel()
        return obs

    def greedy_actions(self) -> np.ndarray:
        """Each drone steps toward its nearest active fire cell."""
        acts = np.zeros(self.n_drones, dtype=np.int64)
        for i in range(self.n_drones):
            if self._fire_dist[i] >= 1.0 or (self._fire_dr[i] == 0 and self._fire_dc[i] == 0):
                continue
            if abs(self._fire_dr[i]) >= abs(self._fire_dc[i]):
                acts[i] = 1 if self._fire_dr[i] < 0 else 2
            else:
                acts[i] = 3 if self._fire_dc[i] < 0 else 4
        return acts

    # ------------------------------------------------------------- VecEnv API
    def reset(self):
        self._new_episode()
        return self._build_obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int64).reshape(self.n_drones)

    def step_wait(self):
        self.drone_pos = np.clip(self.drone_pos + _MOVES[self._actions],
                                 0, self.grid - 1)

        thr = self.sim.ignition_threshold * 0.3
        dropped = np.zeros(self.n_drones, dtype=np.float32)
        on_fire = np.zeros(self.n_drones, dtype=np.float32)
        for i in range(self.n_drones):
            r, c = self.drone_pos[i]
            if self.sim.heat_intensity[r, c] > thr:
                on_fire[i] = 1.0
                if self.drone_water[i] >= self.drop_amount:
                    self.sim.apply_water_drop(int(r), int(c), self.drop_amount)
                    self.drone_water[i] -= self.drop_amount
                    dropped[i] = 1.0
            self.drone_water[i] = min(self.water_capacity,
                                      self.drone_water[i] + self.water_regen)

        self.sim.step()
        self.steps += 1

        burned = int(self.sim.burned_area.sum())
        new_burned = max(0, burned - self._prev_burned)
        self._prev_burned = burned
        active_cells = int((self.sim.heat_intensity > thr).sum())

        obs = self._build_obs()  # also refreshes fire dist / nn dist caches

        # Per-drone reward: shared fire terms + own contribution + spacing.
        rewards = (
            -1.0 * new_burned / self.n_drones
            - 0.01 * active_cells / self.n_drones
            + 1.0 * dropped
            + 0.3 * on_fire
            + 0.1 * np.exp(-5.0 * self._fire_dist)
            - 0.1 * (self._nn_dist <= 1.0)
        ).astype(np.float32)

        extinguished = active_cells == 0 and self.steps > 1
        truncated = self.steps >= self.max_steps
        done = extinguished or truncated
        if extinguished:
            rewards += 50.0 * (1.0 - burned / (self.grid * self.grid)) / self.n_drones

        self._ep_returns += rewards
        dones = np.full(self.n_drones, done, dtype=bool)
        infos: List[dict] = [{} for _ in range(self.n_drones)]
        if done:
            for i in range(self.n_drones):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["episode"] = {"r": float(self._ep_returns[i]),
                                       "l": self.steps}
                infos[i]["burned_cells"] = burned
                infos[i]["fire_extinguished"] = extinguished
                if truncated and not extinguished:
                    infos[i]["TimeLimit.truncated"] = True
            self._new_episode()
            obs = self._build_obs()
        return obs, rewards, dones, infos

    def close(self):
        self.sim = None

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self.n_drones

    def get_attr(self, attr_name, indices=None):
        n = self.n_drones if indices is None else len(self._get_indices(indices))
        return [getattr(self, attr_name)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        n = self.n_drones if indices is None else len(self._get_indices(indices))
        return [getattr(self, method_name)(*args, **kwargs)] * n

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.n_drones if indices is None else len(self._get_indices(indices))
        return [False] * n


class _StaticFire:
    """Minimal fire surface holding one fixed heat field (for WRF coupling)."""
    def __init__(self, heat, max_heat, ignition_threshold):
        self.heat_intensity = heat
        self.max_heat_intensity = float(max_heat)
        self.ignition_threshold = ignition_threshold
        self.burned_area = np.zeros_like(heat, dtype=int)

    def apply_water_drop(self, r, c, amount=5.0):
        pass


class WRFGridDroneEnv(PerDroneSwarmVecEnv):
    """Run the trained per-drone policy on a *fixed* WRF-SFIRE heat grid.

    Used by the closed-loop coupler: each coupling interval, the current WRF
    fire is loaded with ``set_fire``, the swarm is advanced for several
    micro-steps, and ``drops_on_fire`` reports the fire cells the drones
    suppressed (drone on an active cell with water). Reuses the parent's exact
    ``_build_obs`` so the policy sees training-identical observations.
    """
    def __init__(self, grid, n_drones, max_heat,
                 ignition_threshold=1000.0):
        super().__init__(grid_size=grid, n_drones=n_drones, max_steps=10 ** 9)
        self._max_heat = float(max_heat)
        self._ign = ignition_threshold
        self._drops = set()

    def set_fire(self, heat):
        self.sim = _StaticFire(np.asarray(heat, dtype=float), self._max_heat,
                               self._ign)

    def reset_keep(self, pos=None):
        if pos is None:
            per_row = int(np.ceil(np.sqrt(self.n_drones)))
            for i in range(self.n_drones):
                gr, gc = divmod(i, per_row)
                self.drone_pos[i] = (int((gr + 0.5) * self.grid / per_row),
                                     int((gc + 0.5) * self.grid / per_row))
            self.drone_pos = np.clip(self.drone_pos, 0, self.grid - 1)
        else:
            self.drone_pos = np.clip(np.asarray(pos), 0, self.grid - 1)
        self.drone_water[:] = self.water_capacity
        self._drops = set()
        return self._build_obs()

    def move(self, actions):
        """Move drones one step (no water logic); return fresh observations."""
        self.drone_pos = np.clip(self.drone_pos + _MOVES[np.asarray(actions)],
                                 0, self.grid - 1)
        return self._build_obs()

    def advance(self, actions):
        self.drone_pos = np.clip(self.drone_pos + _MOVES[np.asarray(actions)],
                                 0, self.grid - 1)
        thr = self.sim.ignition_threshold * 0.3
        for i in range(self.n_drones):
            r, c = int(self.drone_pos[i, 0]), int(self.drone_pos[i, 1])
            if self.sim.heat_intensity[r, c] > thr and self.drone_water[i] >= 5.0:
                self.drone_water[i] -= 5.0
                self._drops.add((r, c))
            self.drone_water[i] = min(self.water_capacity,
                                      self.drone_water[i] + self.water_regen)
        return self._build_obs()

    def drops_on_fire(self):
        return list(self._drops)
