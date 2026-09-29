#!/usr/bin/env python3
"""
Recover and evaluate the exploratory/front-aware per-drone policy from its last
checkpoint (the 2M-frame run died ~96% done with no final model saved).

Loads agents/checkpoints/ckpt/perdrone_explore_1440000_steps.zip and runs a
paired eval (random / greedy / trained) on the SAME env config it trained on:
difficulty=hard, continuous=True, wind_obs=True, spread_penalty=2.5.
"""
import sys
from pathlib import Path
import logging

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from train_per_drone import env_kwargs, paired_eval  # noqa: E402

logging.disable(logging.CRITICAL)
from stable_baselines3 import PPO  # noqa: E402

CKPT = project_root / "agents/checkpoints/ckpt/perdrone_explore_1440000_steps.zip"
EPISODES = 20

kwargs = env_kwargs("hard")
kwargs["continuous"] = True
kwargs["wind_obs"] = True
kwargs["spread_penalty"] = 2.5

tot = kwargs["grid_size"] ** 2
print("=" * 66)
print("Recovered explore policy eval  (checkpoint @ 1.44M / 2.0M frames)")
print("=" * 66)
print(f"grid={kwargs['grid_size']} drones={kwargs['n_drones']} "
      f"continuous=True wind_obs=True spread_penalty=2.5")

model = PPO.load(str(CKPT))

base, _ = paired_eval(("random", "greedy"), kwargs, EPISODES, 9000)
trained, _ = paired_eval(("trained",), kwargs, EPISODES, 9000, model=model)

rb = base["random"][0]
print("\nRESULT (identical fires, paired, %d episodes)" % EPISODES)
print(f"{'controller':<12}{'burned':>9}{'% grid':>9}{'extinguish':>12}{'vs random':>11}")
rows = [("random", *base["random"]), ("greedy", *base["greedy"]),
        ("explore", *trained["trained"])]
for name, b, x in rows:
    print(f"{name:<12}{b:>9.0f}{b/tot:>8.0%}{x:>11.0%}{(rb-b)/rb:>+10.0%}")
gb = base["greedy"][0]
print(f"\nexplore vs greedy: {(gb-trained['trained'][0])/gb:+.0%}")
