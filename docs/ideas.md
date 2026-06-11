# Future ideas / notes for later

Running list of enhancements identified but not yet built.

## 1. Learn *whether* and *where* to attack — fire-cell prioritization (NEW)

Today every drone just flies to (and drops on) its **nearest active fire cell**.
But not all fire cells are equally worth attacking. Add a learned **attack /
don't-attack decision** and a notion of **cell value/priority**, so the swarm
spends its (very limited) water where it matters most.

Why some cells matter more than others:
- **Leading edge vs. interior.** Suppressing the *advancing* front (downwind /
  upslope, high rate-of-spread) prevents far more future burning than dousing
  already-burned interior or a backing edge that's barely moving.
- **Spread potential.** A cell next to lots of dry, dense, unburned fuel (high
  ROS) ignites many neighbours if left — high value. A cell surrounded by burned
  ground or a natural break will self-limit — low value, don't waste water.
- **Values at risk.** Cells threatening assets (a town edge, a substation, a
  lake/road we want to hold) deserve priority even if not the hottest.
- **Water economy.** Withholding water on low-value cells = more water for the
  front = fewer refuel trips. Directly compounds with the `vol` (water-cost)
  efficiency idea.

Ways to implement (cheapest first):
- **Action gate:** add a discrete "drop / hold" output alongside the movement
  vector, so the policy can choose *not* to drop on a low-value cell.
- **Value-weighted reward:** weight the suppression reward by the cell's spread
  potential (e.g. count of unburned flammable neighbours, or local ROS), so
  attacking high-ROS front cells pays more than interior cells. The policy then
  *learns* prioritization without an explicit value map.
- **Priority/value map in the observation:** feed a per-cell "spread-risk" layer
  (from wind + slope + fuel) and/or an asset/value layer; drones target
  high-value front cells. (Pairs with the wind/spread-direction obs already
  noted for beating greedy.)
- **Stretch — learned value head:** predict each cell's marginal "burned-area
  prevented if suppressed now" and dispatch to the top-K.

This connects to two earlier threads: **anticipation** (attack ahead of the
front, not where it currently is) and **water efficiency** (`vol`).

## 2. Other deferred ideas (from earlier in the project)

- **Beat greedy properly:** continuous-vector policy (done: `vec`) + **wind /
  fire-spread-direction observations** + **per-litre reward** so efficiency is an
  explicit objective. (See docs/rl_pipeline.md.)
- **Train the trucks / hierarchical dispatch:** currently trucks are heuristic
  (retreat only). Learn truck repositioning, or a central allocator assigning
  drones to fire sectors / trucks.
- **Finer WRF-SFIRE coupling** (12-20 s intervals instead of 60 s) and
  **calibrated water -> suppression** physics, on the live WRF closed loop.
- **Geolocated WRF-SFIRE domain** (real California/Australia terrain + fuels) —
  needs the `geog` static dataset + GRIB met data (blocked in the sandbox).
- **Spatial hashing / KD-tree everywhere** for 5,000-10,000-drone speed (KD-tree
  already added to the per-drone obs).
- **Response-lag realism** (done in truck_sim via `--deploy-delay`) — extend to
  detection uncertainty and staggered truck arrival.
