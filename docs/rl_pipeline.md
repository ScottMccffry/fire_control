# RL Drone-Swarm Wildfire Suppression — Pipeline & Findings

This document records the reinforcement-learning pipeline added on top of the
wildfire/drone simulator, the experiments run, and the conclusions.

## Components

| File | Purpose |
|------|---------|
| `envs/drone_fire_env.py` | Gymnasium env wrapping `MockFireSimulator` with agent-controlled drones. Two observation modes: `flat` (vector, for MLP) and `spatial` (Dict: multi-channel map + per-drone vector, for CNN). |
| `agents/cnn_policy.py` | `FireSwarmExtractor` — CNN over the map + MLP over the per-drone vector; adaptive pooling so one architecture fits any grid size. |
| `scripts/train_drone_swarm.py` | Flat-MLP PPO training (difficulty presets, VecNormalize, paired-ish eval). |
| `scripts/train_drone_swarm_cnn.py` | CNN policy: collect greedy demos → behavior-clone → critic warmup → PPO fine-tune. Saves a `_bc` checkpoint (BC-only) and the final checkpoint. |
| `scripts/evaluate_policy.py` | **Paired** evaluation: random/greedy/trained on *identical* seeded fires, with a significance check. |
| `envs/per_drone_env.py` | **Decentralized per-drone env** (`PerDroneSwarmVecEnv`): each drone is one SB3 VecEnv stream with an egocentric observation; one shared policy, per-drone rewards. |
| `scripts/train_per_drone.py` | Per-drone shared-policy PPO training + paired evaluation. |

## Difficulty presets (`DIFFICULTY_PRESETS`)

| preset | grid | drones | notes |
|--------|------|--------|-------|
| easy | 20×20 | 6 | mild fire — mostly self-extinguishes (little to learn) |
| medium | 32×32 | 16 | sustained spreading fire; main test bed |
| hard | 40×40 | 24 | drier/windier |
| large | 100×100 | 40 | WRF-SFIRE-scale probe |

## Methodology note: evaluate on identical fires

Fire spread is stochastic with large episode-to-episode variance (~10% of the
grid). Comparing controllers on *different* random fires is dominated by that
noise and produced misleading results early on. All conclusions below use
**paired** evaluation (same seeded fires for every controller). The fire RNG is
seeded in `DroneFireEnv.reset` for reproducibility.

## Results (paired, lower burned = better)

Greedy = hand-coded "each drone steps toward its nearest active fire cell".

### medium (32×32, 16 drones)
| controller | % grid burned | vs random |
|------------|---------------|-----------|
| random | ~60% | — |
| greedy | ~19% | +70% |
| flat-MLP PPO (250k–400k) | ~64% | ~0% (no learning) |
| **CNN + BC (cloned)** | **~20%** | **+68%** (matches greedy) |
| CNN + BC + PPO fine-tune | ~25% | +60% (PPO mildly degrades BC) |

### large (100×100, 40 drones)
| controller | % grid burned | vs random |
|------------|---------------|-----------|
| random | ~27% | — |
| greedy | ~7% | +73% |
| **CNN + BC (cloned)** | **~11%** | **+58%** (scales; gap closes with more BC) |

### Per-drone shared policy (decentralized) — BEATS GREEDY

One small policy (43-dim egocentric obs → Discrete(5)), run independently by
every drone. Observation: own position/water, nearest-fire bearing+distance,
fire centroid + size, 5×5 local heat patch, 3 nearest-drone offsets,
fleet-centroid offset, and fleet rank (fraction of drones closer to the fire).
Per-drone rewards (own drops, on-fire, anti-clumping spacing penalty + shared
fire terms). Trained with PPO from scratch — no behavior cloning needed —
via `PerDroneSwarmVecEnv` (num_envs = n_drones, so each sim step yields
n_drones samples).

medium (32×32, 16 drones), 2M frames, paired over 20 fires:

| controller | % grid burned | extinguish | vs random |
|------------|---------------|------------|-----------|
| random | 61% | 85% | — |
| greedy | 22% | 90% | +64% |
| **per-drone** | **17%** | **100%** | **+72%** |

**Scale transfer** — large (100×100, 40 drones), paired over the same 12 fires:

| controller | % grid burned | extinguish | vs random |
|------------|---------------|------------|-----------|
| random | 25% | 0% | — |
| greedy | 9% | 33% | +62% |
| per-drone trained natively at 100×100 (2M frames) | 7% | 50% | +73% |
| **per-drone trained at 32×32, zero-shot transfer** | **6%** | **58%** | **+74%** |

The egocentric observation is grid-normalized and fixed-size, so the policy is
effectively scale-free: weights trained at 32×32 with 16 drones match native
large-scale training on a 10× larger grid with 2.5× more drones — train cheap
at small scale, deploy at large scale. This is the architecture to take toward
WRF-SFIRE-scale swarms.

## Key findings

1. **A flat-MLP policy does not learn this task.** Flattening a large heat map
   into a vector discards spatial structure; PPO from scratch stayed at random
   even after 400k steps.
2. **A CNN policy + behavior cloning from greedy works and scales.** The
   decisive fix was giving each drone an **egocentric bearing** (unit dx, dy,
   distance) to the nearest active fire in its per-drone features — this solves
   the per-drone "binding" problem (relating a specific drone to its local fire
   context) that a centralized global-pooled policy cannot.
3. **PPO fine-tuning does not beat greedy here**, and naively even degrades the
   BC policy. Mitigated (not eliminated) by a **critic warmup** + gentle
   fine-tune (`ent_coef=0`, low LR, `target_kl`). Root cause: the reward credits
   exactly the greedy behaviors and the bearing feature biases the policy toward
   greedy, so greedy is ~optimal under this design — there is little headroom.
4. **Vectorizing `MockFireSimulator.step`** (NumPy) gave ~5.7× speedup at
   100×100, making large-grid training and paired evaluation practical.
5. **Decentralized beats centralized AND greedy.** A shared per-drone policy
   with per-drone rewards learns from scratch (no BC) what the centralized
   models never could: +72% vs random on medium (greedy: +64%) and +74% on
   large (greedy: +62%). Per-drone credit assignment and a fleet-relative
   observation (rank, centroid offset, neighbour spacing) are what unlock both
   learning and coordination beyond greedy.
6. **Zero-shot transfer matches native training at scale.** On 12 identical
   100×100 fires, the 32×32-trained policy (632 burned) equals the natively
   100×100-trained one (659) — scale-free observations mean you can train at
   small scale and deploy at large scale for free.

## To actually beat greedy (recommended next steps)

Greedy's blind spots are coordination and foresight. Give the policy headroom
and a reason to learn beyond greedy:

- **Reward coordination**: penalize over-assignment (many drones on one cell),
  reward covering more of the *perimeter* / distinct active cells.
- **Reward anticipation**: credit pre-positioning downwind of the front
  (containment) rather than only suppressing already-burning cells.
- **Stop spoon-feeding**: drop or weaken the per-drone nearest-fire bearing so
  the policy must learn allocation from the map.
- **Scale up**: more BC epochs at 100×100 (BC loss was still high there), then
  longer PPO with the coordination reward.
- **Architecture for very large swarms**: a shared per-drone policy with
  egocentric local + global-coarse observations (parameter sharing) is the
  design that scales to thousands of drones.

## Reproduce

```bash
# CNN + BC + fine-tune on the medium fire
python scripts/train_drone_swarm_cnn.py --difficulty medium \
    --bc-steps 40000 --bc-epochs 10 --value-warmup 20000 --timesteps 150000

# 100x100 scale
python scripts/train_drone_swarm_cnn.py --difficulty large \
    --bc-steps 40000 --bc-epochs 8 --value-warmup 20000 --timesteps 200000

# The CNN script prints a paired random/greedy/trained comparison at the end.
# scripts/evaluate_policy.py is the paired evaluator for the FLAT-MLP models:
python scripts/evaluate_policy.py \
    --model agents/checkpoints/drone_swarm_ppo_medium_g32_d16 \
    --difficulty medium --episodes 40
```
