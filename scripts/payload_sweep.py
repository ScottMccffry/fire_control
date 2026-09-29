#!/usr/bin/env python3
"""
Sweep drone water payload (10..100 L) on the forest fire and plot how much
burned-area reduction each payload buys -- to find the suppression threshold.
"""
import sys
from pathlib import Path
import logging
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import truck_sim as T  # noqa: E402

logging.disable(logging.CRITICAL)
GRID, DPT, TICKS, SEED = 400, 200, 250, 3
T.DROP_EFF = 0.2                      # forest fuel: each litre ~5x less effective
tot = GRID * GRID

base = T.run(GRID, 10, DPT, TICKS, SEED, use_drones=False, forest=True)[0]
print(f"forest baseline burned: {base} ({base/tot:.0%})")

payloads = [10, 20, 30, 40, 50, 75, 100]
rows = []
for tk in payloads:
    T.TANK = float(tk)
    burned, _, _, water, _ = T.run(GRID, 10, DPT, TICKS, SEED, use_drones=True,
                                   balance=True, forest=True)
    red = (base - burned) / base
    rows.append((tk, burned, red, water))
    print(f"SWEEP tank={tk:3d}L  burned={burned:6d}  reduction={red:+.0%}  water={water:,.0f} L")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
tks = [r[0] for r in rows]; reds = [r[2] * 100 for r in rows]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(tks, reds, "o-", lw=2, color="tab:blue")
for tk, _, r, _ in rows:
    ax.annotate(f"{r:+.0%}", (tk, r * 100), textcoords="offset points", xytext=(0, 7), fontsize=8)
ax.set_xlabel("drone water payload (L)")
ax.set_ylabel("burned-area reduction vs baseline (%)")
ax.set_title("Forest fire: reduction vs drone payload\n(2000 drones, balanced refuel; forest fuel)")
ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
fig.savefig("docs/figures/payload_sweep.png", dpi=120, bbox_inches="tight")
print("wrote docs/figures/payload_sweep.png")
