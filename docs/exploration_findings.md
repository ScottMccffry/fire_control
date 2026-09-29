# Does a "more exploratory" reward beat nearest-fire on the macro view?

**Question (user).** On small grids the per-drone policy beats the
nearest-fire greedy heuristic, but the drones "mostly go to the fire point
nearest to them." On a *macro* view the fire expands massively *away* from a
clustered fleet, so maybe a **more exploratory reward** would teach the drones
to intercept the runaway front instead of dribbling on the near edge.

**Answer: no — more exploration consistently hurt.** On the regimes where the
macro failure actually occurs, the *standard* reward matches greedy and the
*exploratory* reward is much worse. Nearest-fire is a very strong baseline
because, recomputed every step, "nearest active cell" already tracks the live
front. The drones-go-to-nearest behaviour the user worried about is, in sim,
close to optimal.

## How the regimes were chosen

The original `hard` preset is **saturated**: random already burns ~90% of the
grid, so there is no headroom for any policy to demonstrate anything. The
recovered checkpoint of the first exploratory run (trained on `hard`, died at
~96%) scored **87% burned — worse than greedy's 81%**. That told us the regime,
not the reward, was the problem.

`probe_regime.py` / `probe_multifront.py` then mapped where headroom *and* a
genuine macro failure exist (grid 60, strong wind, 6 seeds):

| ignitions | drones | random | nearest-greedy | note |
|---|---|---|---|---|
| 1 | 24 | 85% | **20%** | one coherent front: nearest = the front, greedy near-optimal |
| 2 | 24 | 87% | **37%** | greedy still strong |
| 4 | 24 | 88% | **86%** | greedy collapses, but 24 drones are *under-resourced* |
| 4 | 40 | 87% | **40%** | greedy weak **and** fire suppressible → real test bed |
| 4 | 60 | 87% | 2% | enough drones to crush 4 fronts |

So two preset regimes were added (`scripts/train_drone_swarm.py`):
- **`front`** — 60×60, 1 ignition, 24 drones (single runaway front).
- **`multifront`** — 60×60, 4 ignitions, 40 drones (fragmented fire; the regime
  that matches the user's macro failure: greedy can't allocate a fixed fleet
  across several fronts).

## Results (paired, identical seeded fires, 20 episodes)

`front` (single front), continuous + wind-obs:

| controller | burned | extinguish | vs random |
|---|---|---|---|
| random | 84% | 0% | — |
| nearest-greedy | 16% | 75% | +82% |
| per-drone (exploratory: spread-pen 2.5, ent 0.03) | 54% | 30% | +36% |

`multifront` (4 fronts, 40 drones), continuous + wind-obs:

| controller | burned | extinguish | vs random |
|---|---|---|---|
| random | 88% | 10% | — |
| nearest-greedy | 28% | 65% | +68% |
| per-drone (**exploratory**: spread-pen 2.5, ent 0.02) | 64% | 20% | +28% |
| per-drone (**standard**: spread-pen 1.0, ent 0.01) | **30%** | **75%** | **+66%** |

The standard-reward policy **matches greedy** (30% vs 28% — a tie) and
**extinguishes more often** (75% vs 65%); its episode reward went positive
(+7.9) while the exploratory run stayed deeply negative (−136) and its critic,
though it converged (explained_variance ~0.86), was fitting a diffuse,
low-return policy.

## Why exploration hurt

The heavy `spread_penalty` + high `ent_coef` push the fleet to *spread out and
hedge*. Against a fire you want the opposite: **commit drones decisively to a
front and stay on it**. Extra entropy keeps the policy wandering off the front
it should be holding; the new-burned penalty rewards "be near where it might
spread" over "kill what is burning now." Greedy gets this for free by
recomputing the nearest live cell every step.

## Takeaways

1. **Don't add exploration to fight a runaway front.** Lower entropy and a plain
   "drops on fire" reward win. The macro fix is *capacity and allocation*, not
   *stochasticity*.
2. **Nearest-fire is a strong baseline** wherever the fleet is adequately
   resourced (1–2 fronts, or enough drones for N fronts). A learned policy can
   *match* it and extinguish slightly more reliably, but not beat it on burned
   area in these regimes.
3. **The real macro failure is under-resourcing, not myopia.** Greedy collapses
   at 4 fronts with 24 drones because 24 drones cannot cover 4 fronts — no
   controller can. Add drones (or per-drop effectiveness; see
   `docs/drone_design.md` and the payload sweep) and the problem becomes
   tractable again.

## Reproduce

```bash
# map headroom / where greedy fails
python scripts/probe_regime.py
python scripts/probe_multifront.py

# the policy that matches greedy on the fragmented-fire regime
python scripts/train_per_drone.py --difficulty multifront --timesteps 3000000 \
  --continuous --wind-obs --spread-penalty 1.0 --ent-coef 0.01 --tag multifront_std

# recover/eval the original (dead, saturated-regime) exploratory checkpoint
python scripts/eval_explore_ckpt.py
```
