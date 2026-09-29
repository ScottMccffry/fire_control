"""
Decentralized per-drone RL env with TRUCK logistics — drones learn to dispatch.

Extends the per-drone idea with mobile depots: each drone observes not just the
fire and its neighbours but the **nearest truck and how much water it has left**,
and learns when to fight vs. peel off to refuel (and implicitly which truck to
use). Trucks dispense at a limited rate from a finite reserve and retreat from
the advancing front; the fire is an irregular blob.

VecEnv with num_envs = n_drones (one shared world, one shared policy). Continuous
(vector) action. Train small (scale-free observation) and deploy at 10x700.
"""

from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sim import MockFireSimulator

_KNN = 3
_PATCH = 5
CELL_M = 25.0


class TruckDispatchVecEnv(VecEnv):
    def __init__(self, grid_size=120, n_trucks=3, drones_per_truck=60,
                 max_steps=300, circ_radius_m=500.0, wind_speed=5.0,
                 base_spread_rate=0.16, max_move=1.5, water_cost=0.2,
                 tank=10.0, dispense_lpm=100.0, truck_reserve=20000.0,
                 refuel_dist=2.0, episode_seed: Optional[int] = None):
        self.grid = int(grid_size)
        self.n_trucks = int(n_trucks)
        self.dpt = int(drones_per_truck)
        self.n_drones = self.n_trucks * self.dpt
        self.max_steps = int(max_steps)
        self.circ_radius = circ_radius_m / CELL_M
        self.wind_speed = float(wind_speed)
        self.base_spread_rate = float(base_spread_rate)
        self.max_move = float(max_move)
        self.water_cost = float(water_cost)
        self.tank = float(tank)
        self.dispense = dispense_lpm
        self.reserve0 = truck_reserve
        self.refuel_dist = refuel_dist
        self.safe = 1000.0 / CELL_M
        self.retreat = 200.0 / CELL_M
        self.episode_seed = episode_seed

        # own(3) fire(6) truck(4) fleet(3) knn(2K) patch(P^2)
        obs_dim = 3 + 6 + 4 + 3 + 2 * _KNN + _PATCH * _PATCH
        super().__init__(self.n_drones,
                         spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
                         spaces.Box(-1.0, 1.0, (2,), np.float32))
        self.render_mode = None
        self._rng = np.random.default_rng(episode_seed)
        self._actions = None
        self._ep_returns = np.zeros(self.n_drones)

    # ------------------------------------------------------------- episode
    def _new_episode(self):
        seed = self.episode_seed
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        if seed is not None:
            np.random.seed(seed)
        g = self.grid
        self.sim = MockFireSimulator(grid_size=(g, g), cell_size_meters=CELL_M)
        self.sim.set_weather(self.wind_speed, float(rng.uniform(0, 360)), 32.0, 0.15)
        self.sim.fuel_moisture = rng.uniform(0.06, 0.16, (g, g))
        self.sim.fuel_load = rng.uniform(0.8, 1.0, (g, g))
        self.sim.base_spread_rate = self.base_spread_rate
        # irregular fire
        R = self.circ_radius
        cy = g / 2 + rng.uniform(-g * 0.08, g * 0.08)
        cx = g / 2 + rng.uniform(-g * 0.08, g * 0.08)
        th = np.linspace(0, 2 * np.pi, 360)
        r = np.ones_like(th)
        for k in range(1, 5):
            r += rng.uniform(-0.5, 0.5) / k * np.cos(k * th + rng.uniform(0, 2 * np.pi))
        r = np.clip(r, 0.3, None); r *= R / r.max()
        yy, xx = np.mgrid[0:g, 0:g]
        ang = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = dist <= np.interp(ang, th, r)
        self.sim.burned_area[mask] = 1
        self.sim.heat_intensity[mask] = self.sim.max_heat_intensity * 0.7
        # trucks: random, >= 1 km from fire front
        fa = self._active()
        tree = cKDTree(fa) if len(fa) else None
        trucks = []
        while len(trucks) < self.n_trucks:
            p = rng.uniform(0.05, 0.95, 2) * g
            if tree is None or tree.query(p)[0] >= self.safe:
                trucks.append(p)
        self.trucks = np.array(trucks)
        self.reserve = np.full(self.n_trucks, self.reserve0)
        # drones at trucks
        self.pos = np.repeat(self.trucks, self.dpt, axis=0).astype(float)
        self.pos += rng.uniform(-2, 2, self.pos.shape)
        self.pos = np.clip(self.pos, 0, g - 1)
        self.water = np.full(self.n_drones, self.tank)
        self.steps = 0
        self._prev_burned = int(self.sim.burned_area.sum())
        self._ep_returns[:] = 0.0

    def _active(self):
        thr = self.sim.ignition_threshold * 0.3
        fr, fc = np.where(self.sim.heat_intensity > thr)
        return np.stack([fr, fc], axis=1).astype(float) if len(fr) else np.zeros((0, 2))

    # ------------------------------------------------------------- obs
    def _build_obs(self):
        g = self.grid
        r, c = self.pos[:, 0], self.pos[:, 1]
        obs = np.zeros((self.n_drones, self.observation_space.shape[0]), np.float32)
        obs[:, 0] = r / (g - 1) * 2 - 1
        obs[:, 1] = c / (g - 1) * 2 - 1
        obs[:, 2] = self.water / self.tank * 2 - 1
        thr = self.sim.ignition_threshold * 0.3
        active = self.sim.heat_intensity > thr
        self._fire_vec = np.zeros((self.n_drones, 2))
        self._fire_dist = np.full(self.n_drones, 1.0)
        if active.any():
            dmap, (ir, ic) = ndimage.distance_transform_edt(~active, return_indices=True)
            ri, ci = np.round(r).astype(int).clip(0, g-1), np.round(c).astype(int).clip(0, g-1)
            dr, dc = ir[ri, ci] - r, ic[ri, ci] - c
            dd = np.hypot(dr, dc); nz = np.maximum(dd, 1e-6)
            self._fire_vec = np.stack([dr / nz, dc / nz], axis=1)
            self._fire_dist = np.clip(dd / g, 0, 1)
            obs[:, 3] = dr / nz; obs[:, 4] = dc / nz; obs[:, 5] = self._fire_dist
            ar, ac = np.where(active)
            obs[:, 6] = np.clip((ar.mean() - r) / g, -1, 1)
            obs[:, 7] = np.clip((ac.mean() - c) / g, -1, 1)
            obs[:, 8] = np.clip(active.sum() / (g * g) * 10, 0, 1)
        # nearest truck + its reserve fraction
        dt_ = np.linalg.norm(self.pos[:, None, :] - self.trucks[None, :, :], axis=2)
        ti = dt_.argmin(axis=1)
        self._truck_idx = ti
        tv = self.trucks[ti] - self.pos
        tn = np.linalg.norm(tv, axis=1, keepdims=True)
        tu = np.where(tn > 1e-6, tv / tn, 0.0)
        obs[:, 9] = tu[:, 0]; obs[:, 10] = tu[:, 1]
        obs[:, 11] = np.clip(tn[:, 0] / g, 0, 1)
        obs[:, 12] = self.reserve[ti] / self.reserve0 * 2 - 1
        # fleet centroid + rank
        obs[:, 13] = np.clip((r.mean() - r) / g, -1, 1)
        obs[:, 14] = np.clip((c.mean() - c) / g, -1, 1)
        if active.any():
            dd2 = dmap[ri, ci]
            obs[:, 15] = (dd2[:, None] > dd2[None, :]).mean(1) * 2 - 1
        # KNN
        kq = min(_KNN + 1, self.n_drones)
        nn_d, nn_i = cKDTree(self.pos).query(self.pos, k=kq, p=1)
        if nn_i.ndim == 1:
            nn_i = nn_i[:, None]; nn_d = nn_d[:, None]
        nn = nn_i[:, 1:]
        for k in range(_KNN):
            j = nn[:, k] if k < nn.shape[1] else np.arange(self.n_drones)
            obs[:, 16 + 2*k] = np.clip((self.pos[j, 0] - r) / g, -1, 1)
            obs[:, 17 + 2*k] = np.clip((self.pos[j, 1] - c) / g, -1, 1)
        self._nn_dist = nn_d[:, 1] if nn_d.shape[1] > 1 else np.full(self.n_drones, g)
        # local heat patch
        pad = _PATCH // 2
        hp = np.pad(self.sim.heat_intensity / self.sim.max_heat_intensity, pad)
        ri, ci = np.round(r).astype(int).clip(0, g-1), np.round(c).astype(int).clip(0, g-1)
        base = 16 + 2 * _KNN
        for i in range(self.n_drones):
            obs[i, base:base + _PATCH*_PATCH] = hp[ri[i]:ri[i]+_PATCH, ci[i]:ci[i]+_PATCH].ravel()
        return obs

    def greedy_actions(self):
        return self._fire_vec.astype(np.float32)

    # ------------------------------------------------------------- VecEnv
    def reset(self):
        self._new_episode(); return self._build_obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, np.float32).reshape(self.n_drones, 2)

    def step_wait(self):
        g = self.grid
        self.pos = np.clip(self.pos + np.clip(self._actions, -1, 1) * self.max_move, 0, g - 1)
        ip = np.round(self.pos).astype(int).clip(0, g - 1)
        thr = self.sim.ignition_threshold * 0.3
        # drops on active fire
        on = (self.sim.heat_intensity[ip[:, 0], ip[:, 1]] > thr) & (self.water > 0)
        dropped = np.zeros(self.n_drones, np.float32)
        for i in np.where(on)[0]:
            amt = min(self.water[i], 5.0)
            self.sim.apply_water_drop(int(ip[i, 0]), int(ip[i, 1]), amt)
            self.water[i] -= amt; dropped[i] = 1.0
        # refuel at nearest truck (rate + reserve limited)
        dt_ = np.linalg.norm(self.pos - self.trucks[self._truck_idx], axis=1)
        for k in range(self.n_trucks):
            here = np.where((self._truck_idx == k) & (dt_ < self.refuel_dist) & (self.water < self.tank))[0]
            budget = min(self.dispense, self.reserve[k])
            for i in here:
                if budget <= 0:
                    break
                give = min(self.tank - self.water[i], budget)
                self.water[i] += give; budget -= give; self.reserve[k] -= give
        self.reserve = np.minimum(self.reserve0, self.reserve + self.dispense)
        # trucks retreat
        fa = self._active()
        if len(fa):
            ft = cKDTree(fa); cen = fa.mean(0)
            for k in range(self.n_trucks):
                if ft.query(self.trucks[k])[0] < self.retreat:
                    a = self.trucks[k] - cen; a /= (np.linalg.norm(a) + 1e-6)
                    self.trucks[k] = np.clip(self.trucks[k] + a * self.retreat * 1.5, 0, g - 1)
        self.sim.step()
        self.steps += 1
        burned = int(self.sim.burned_area.sum())
        new_b = max(0, burned - self._prev_burned); self._prev_burned = burned
        active_cells = int((self.sim.heat_intensity > thr).sum())
        obs = self._build_obs()
        on_fire = (self.sim.heat_intensity[ip[:, 0], ip[:, 1]] > thr).astype(np.float32)
        rewards = (-1.0 * new_b / self.n_drones - 0.01 * active_cells / self.n_drones
                   + (0.6 - self.water_cost) * dropped + 0.3 * on_fire
                   + 0.1 * np.exp(-5 * self._fire_dist)
                   - 0.1 * (self._nn_dist <= 1.0)).astype(np.float32)
        self._ep_returns += rewards
        ext = active_cells == 0 and self.steps > 1
        trunc = self.steps >= self.max_steps
        done = ext or trunc
        if ext:
            rewards += 30.0 * (1 - burned / (g * g)) / self.n_drones
        infos = [{} for _ in range(self.n_drones)]
        dones = np.full(self.n_drones, done)
        if done:
            for i in range(self.n_drones):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["episode"] = {"r": float(self._ep_returns[i]), "l": self.steps}
                infos[i]["burned_cells"] = burned
                infos[i]["fire_extinguished"] = ext
                if trunc and not ext:
                    infos[i]["TimeLimit.truncated"] = True
            self._new_episode(); obs = self._build_obs()
        return obs, rewards, dones, infos

    def close(self): self.sim = None
    def seed(self, seed=None): self._rng = np.random.default_rng(seed); return [seed] * self.n_drones
    def get_attr(self, n, indices=None): return [getattr(self, n)] * self.n_drones
    def set_attr(self, n, v, indices=None): setattr(self, n, v)
    def env_method(self, m, *a, indices=None, **k): return [getattr(self, m)(*a, **k)] * self.n_drones
    def env_is_wrapped(self, w, indices=None): return [False] * self.n_drones
