"""
Per-drone (decentralized, parameter-shared) DEFENDER environment.

Where PerDroneSwarmVecEnv learns to *attack* (drop water on active fire), this
learns to *defend*: each drone decides where to move and lay long-term RETARDANT
so the fleet collectively builds a containment line that minimises burned area --
the learned, continuously-adaptive alternative to the hand-coded perimeter/grid
geometry in scripts/realistic_sim.py.

Mechanics:
  * The drone lays retardant on its current (unburned, not-yet-burning) cell each
    step, consuming a finite reserve that regenerates slowly (abstracted refill).
  * Retardant uses the sim's `retardant` field: it raises the local ignition
    threshold, so treated cells resist fire (a strong line here, to test
    POSITIONING rather than line chemistry).

Observation (egocentric, scale-free): own pos + retardant, nearest-fire bearing/
distance, fire centroid + active fraction, wind/spread direction, k-nearest
neighbour drones, fleet centroid, and BOTH a local heat patch and a local
retardant patch (so a drone sees where the line already exists and fills gaps).

Reward (per drone): shared -new_burned (stop the fire) + dense shaping that
rewards laying retardant *just ahead of the front* (a standoff band), penalises
wasteful laying (far from the fire or on already-treated cells), and discourages
clumping -- which is what makes "build a line" learnable.
"""
from typing import List, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sim import MockFireSimulator

_PATCH = 5
_KNN = 3


class DefenderSwarmVecEnv(VecEnv):
    """VecEnv where each 'env' is one retardant-laying defender drone."""

    def __init__(
        self,
        grid_size: int = 48,
        n_drones: int = 30,
        max_steps: int = 160,
        n_ignitions: int = 1,
        wind_speed: float = 12.0,
        fuel_moisture_range=(0.05, 0.15),
        fuel_load_range=(0.8, 1.0),
        base_spread_rate: float = 0.22,
        ret_capacity: float = 10.0,
        ret_regen_per_step: float = 1.0,
        max_move: float = 1.5,
        move_frac: float = None,        # if set, max_move = move_frac*grid (scale-relative)
        peak_kw: float = None,          # fire peak intensity (match realistic_sim)
        fire_headstart: int = 0,        # steps the fire grows before drones engage
        front_band: float = 5.0,        # ideal standoff (cells) ahead of the fire
        line_kw: float = 900.0,         # retardant strength (raises ignition bar)
        spread_penalty: float = 1.0,
        grid_choices=None,              # if set, randomize grid each episode (scale invariance)
        ign_choices=None,               # if set, randomize #ignitions each episode
        episode_seed: Optional[int] = None,
    ):
        self.grid_choices = list(grid_choices) if grid_choices else None
        self.ign_choices = list(ign_choices) if ign_choices else None
        self.grid = int(grid_size)
        self.n_drones = int(n_drones)
        self.max_steps = int(max_steps)
        self.n_ignitions = int(n_ignitions)
        self.wind_speed = float(wind_speed)
        self.fuel_moisture_range = tuple(fuel_moisture_range)
        self.fuel_load_range = tuple(fuel_load_range)
        self.base_spread_rate = float(base_spread_rate)
        self.ret_capacity = float(ret_capacity)
        self.ret_regen = float(ret_regen_per_step)
        self.max_move = float(max_move)
        self.move_frac = move_frac
        self.peak_kw = peak_kw
        self.fire_headstart = int(fire_headstart)
        self.front_band = float(front_band)
        self.line_kw = float(line_kw)
        self.spread_penalty = float(spread_penalty)
        self.episode_seed = episode_seed

        # pos(2)+ret(1)+nearest-fire dx,dy,dist(3)+fire centroid dx,dy(2)
        # +active frac(1)+wind dx,dy(2)+fleet centroid dx,dy(2)
        # +KNN offsets(2K)+heat patch(P^2)+retardant patch(P^2)
        obs_dim = 2 + 1 + 3 + 2 + 1 + 2 + 2 + 2 * _KNN + 2 * _PATCH * _PATCH
        super().__init__(
            num_envs=self.n_drones,
            observation_space=spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
            action_space=spaces.Box(-1.0, 1.0, (2,), np.float32),
        )
        self.render_mode = None
        self._rng = np.random.default_rng(episode_seed)
        self._actions = None
        self.sim: Optional[MockFireSimulator] = None
        self.drone_pos = np.zeros((self.n_drones, 2), np.int64)
        self.drone_posf = np.zeros((self.n_drones, 2), float)
        self.ret = np.full(self.n_drones, self.ret_capacity, np.float32)
        self.steps = 0
        self._fire_dr = np.zeros(self.n_drones)
        self._fire_dc = np.zeros(self.n_drones)
        self._fire_dist = np.full(self.n_drones, 1.0)
        self._dist_cells = np.full(self.n_drones, self.grid, float)
        self._nn_dist = np.full(self.n_drones, self.grid, float)
        self._ep_returns = np.zeros(self.n_drones)

    # ----------------------------------------------------------- episode setup
    def _new_episode(self):
        seed = self.episode_seed
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        if seed is not None:
            np.random.seed(seed)
        # domain randomization: vary grid size / #ignitions per episode so the
        # shared per-drone policy becomes genuinely scale-invariant
        if self.grid_choices is not None:
            self.grid = int(rng.choice(self.grid_choices))
        if self.ign_choices is not None:
            self.n_ignitions = int(rng.choice(self.ign_choices))
        g = self.grid
        if self.move_frac is not None:
            self.max_move = float(self.move_frac) * g    # grid-relative drone step
        self.sim = MockFireSimulator(grid_size=(g, g), cell_size_meters=100.0)
        if self.peak_kw is not None:
            self.sim.max_heat_intensity = float(self.peak_kw)
        self.sim.set_weather(wind_speed=self.wind_speed,
                             wind_direction=float(rng.uniform(0, 360)),
                             temperature=30.0, humidity=0.2)
        lo, hi = self.fuel_moisture_range
        self.sim.fuel_moisture = rng.uniform(lo, hi, (g, g))
        lo, hi = self.fuel_load_range
        self.sim.fuel_load = rng.uniform(lo, hi, (g, g))
        self.sim.base_spread_rate = self.base_spread_rate
        self.sim.retardant_kw = self.line_kw
        m = max(2, g // 4)
        pts = [(int(rng.integers(m, g - m)), int(rng.integers(m, g - m)))
               for _ in range(self.n_ignitions)]
        self.sim.set_ignition_points(pts)
        for _ in range(self.fire_headstart):    # let the fire grow before engaging
            self.sim.step()
        # drones start spread in a ring AROUND the fire (like trucks surrounding
        # it), so the fleet must form up and lay ahead of the front
        c = np.array([g / 2, g / 2])
        ang = rng.uniform(0, 2 * np.pi, self.n_drones)
        rad = rng.uniform(0.06, 0.20, self.n_drones) * g
        self.drone_posf = np.clip(c + np.stack([rad * np.sin(ang), rad * np.cos(ang)], 1),
                                  0, g - 1)
        self.drone_pos = np.round(self.drone_posf).astype(np.int64)
        self.ret[:] = self.ret_capacity
        self.steps = 0
        self._prev_burned = int(self.sim.burned_area.sum())
        self._ret_laid = 0.0
        self._ep_returns[:] = 0.0

    # ----------------------------------------------------------- observations
    def _build_obs(self):
        g = self.grid
        heat = np.clip(self.sim.heat_intensity / self.sim.max_heat_intensity, 0, 1)
        ret = np.clip(self.sim.retardant, 0, 1)
        thr = self.sim.ignition_threshold * 0.3
        active = self.sim.heat_intensity > thr
        r = self.drone_pos[:, 0]; c = self.drone_pos[:, 1]
        obs = np.zeros((self.n_drones, self.observation_space.shape[0]), np.float32)
        obs[:, 0] = r / (g - 1) * 2 - 1
        obs[:, 1] = c / (g - 1) * 2 - 1
        obs[:, 2] = self.ret / self.ret_capacity * 2 - 1
        if active.any():
            dist_map, (ir, ic) = ndimage.distance_transform_edt(~active, return_indices=True)
            dr = ir[r, c] - r; dc = ic[r, c] - c
            dist = np.sqrt(dr * dr + dc * dc)
            norm = np.maximum(dist, 1e-6)
            self._fire_dr, self._fire_dc = dr, dc
            self._fire_dist = dist / g
            self._dist_cells = dist_map[r, c]
            obs[:, 3] = dr / norm; obs[:, 4] = dc / norm
            obs[:, 5] = np.clip(dist / g, 0, 1)
            ar, ac = np.where(active)
            obs[:, 6] = np.clip((ar.mean() - r) / g, -1, 1)
            obs[:, 7] = np.clip((ac.mean() - c) / g, -1, 1)
            obs[:, 8] = np.clip(active.sum() / (g * g) * 10, 0, 1)
        else:
            self._fire_dist = np.full(self.n_drones, 1.0)
            self._dist_cells = np.full(self.n_drones, g, float)
        wd = np.radians(self.sim.wind_direction)
        obs[:, 9] = np.cos(wd); obs[:, 10] = np.sin(wd)
        obs[:, 11] = np.clip((self.drone_pos[:, 0].mean() - r) / g, -1, 1)
        obs[:, 12] = np.clip((self.drone_pos[:, 1].mean() - c) / g, -1, 1)
        # KNN neighbours
        kq = min(_KNN + 1, self.n_drones)
        tree = cKDTree(self.drone_pos)
        nn_d, nn_i = tree.query(self.drone_pos, k=kq, p=1)
        if nn_i.ndim == 1:
            nn_i = nn_i[:, None]; nn_d = nn_d[:, None]
        nn = nn_i[:, 1:]
        for k in range(_KNN):
            j = nn[:, k] if k < nn.shape[1] else np.arange(self.n_drones)
            obs[:, 13 + 2 * k] = np.clip((self.drone_pos[j, 0] - r) / g, -1, 1)
            obs[:, 14 + 2 * k] = np.clip((self.drone_pos[j, 1] - c) / g, -1, 1)
        self._nn_dist = nn_d[:, 1] if nn_d.shape[1] > 1 else np.full(self.n_drones, g)
        # heat + retardant patches
        pad = _PATCH // 2
        hp = np.pad(heat, pad); rp = np.pad(ret, pad)
        base = 13 + 2 * _KNN
        np2 = _PATCH * _PATCH
        for i in range(self.n_drones):
            obs[i, base:base + np2] = hp[r[i]:r[i] + _PATCH, c[i]:c[i] + _PATCH].ravel()
            obs[i, base + np2:base + 2 * np2] = rp[r[i]:r[i] + _PATCH, c[i]:c[i] + _PATCH].ravel()
        return obs

    def greedy_actions(self):
        """Baseline defender: head toward the nearest fire (and lay en route)."""
        v = np.stack([self._fire_dr, self._fire_dc], axis=1).astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(n > 0, v / n, 0.0).astype(np.float32)

    # ------------------------------------------------------------- VecEnv API
    def reset(self):
        self._new_episode()
        return self._build_obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, np.float32).reshape(self.n_drones, 2)

    def step_wait(self):
        g = self.grid
        self.drone_posf = np.clip(self.drone_posf + np.clip(self._actions, -1, 1) * self.max_move,
                                  0, g - 1)
        self.drone_pos = np.round(self.drone_posf).astype(np.int64)
        thr = self.sim.ignition_threshold * 0.3
        active = self.sim.heat_intensity > thr
        if active.any():
            dist_map = ndimage.distance_transform_edt(~active)
        else:
            dist_map = np.full((g, g), g, float)

        productive = np.zeros(self.n_drones, np.float32)
        wasteful = np.zeros(self.n_drones, np.float32)
        r = self.drone_pos[:, 0]; c = self.drone_pos[:, 1]
        for i in range(self.n_drones):
            ri, ci = int(r[i]), int(c[i])
            d = dist_map[ri, ci]
            can = (self.sim.burned_area[ri, ci] == 0
                   and self.sim.heat_intensity[ri, ci] <= thr
                   and self.ret[i] >= 1.0)
            if can:
                already = self.sim.retardant[ri, ci] > 0.5
                if not already and d <= 2 * self.front_band + 4:
                    rr = slice(max(0, ri - 1), ri + 2); cc = slice(max(0, ci - 1), ci + 2)
                    self.sim.retardant[rr, cc] = 1.0
                    self.ret[i] -= 1.0
                    self._ret_laid += 1.0
                    if 1.0 <= d <= self.front_band + 2:
                        productive[i] = 1.0
                    else:
                        wasteful[i] = 0.5
                elif already:
                    wasteful[i] = 0.3
            self.ret[i] = min(self.ret_capacity, self.ret[i] + self.ret_regen)

        self.sim.step()
        self.steps += 1
        burned = int(self.sim.burned_area.sum())
        new_burned = max(0, burned - self._prev_burned)
        self._prev_burned = burned
        active_cells = int((self.sim.heat_intensity > thr).sum())
        obs = self._build_obs()

        # standoff shaping: be at ~front_band ahead of the fire (the line zone)
        standoff = np.exp(-((self._dist_cells - self.front_band) / 3.0) ** 2)
        rewards = (
            -self.spread_penalty * new_burned / self.n_drones
            - 0.01 * active_cells / self.n_drones
            + 1.0 * productive
            - 0.3 * wasteful
            + 0.15 * standoff
            - 0.1 * (self._nn_dist <= 1.0)
        ).astype(np.float32)

        contained = active_cells == 0 and self.steps > 1
        truncated = self.steps >= self.max_steps
        done = contained or truncated
        if contained:
            rewards += 50.0 * (1.0 - burned / (g * g)) / self.n_drones
        self._ep_returns += rewards
        dones = np.full(self.n_drones, done, bool)
        infos: List[dict] = [{} for _ in range(self.n_drones)]
        if done:
            for i in range(self.n_drones):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["episode"] = {"r": float(self._ep_returns[i]), "l": self.steps}
                infos[i]["burned_cells"] = burned
                infos[i]["fire_extinguished"] = contained
                infos[i]["retardant_laid"] = self._ret_laid
                if truncated and not contained:
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
