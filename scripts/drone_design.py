#!/usr/bin/env python3
"""
Sizing + schematic for a delta-wing firefighting drone that carries its 100 L
water payload inside the wing ("wet wing"), runs the trained per-drone policy on
a small onboard computer, and is launched from the trucks.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

g = 9.81
rho = 1.225
# --- weight budget (kg) ---
W = dict(water=100, airframe=35, battery=25, propulsion=12, avionics=4, vtol=14)
MTOW = sum(W.values())
# --- wing sizing from stall ---
vstall, CLmax = 20.0, 1.1            # m/s, delta with LE flaps
S = MTOW * g / (0.5 * rho * vstall**2 * CLmax)
# --- delta planform (triangle: S = 0.5 * span * root_chord) ---
span = 4.4
root = 2 * S / span
le_sweep = np.degrees(np.arctan2(root, span / 2))
# --- cruise ---
CLcr = 0.35
vcr = np.sqrt(MTOW * g / (0.5 * rho * S * CLcr))
LD = 9.0
P_cr = MTOW * g / LD * vcr / 1000          # kW
batt_kWh = 25 * 0.20                         # 25 kg @ 200 Wh/kg
endurance_min = batt_kWh / (P_cr * 1.4) * 60  # 1.4x cruise for climb/margin
wing_vol = S * 0.17                          # m^3 available (mean t/c ~0.17)

print(f"MTOW={MTOW:.0f} kg  S={S:.1f} m^2  span={span} m  root={root:.1f} m  "
      f"LE sweep={le_sweep:.0f} deg")
print(f"stall={vstall:.0f} m/s  cruise={vcr:.0f} m/s ({vcr*3.6:.0f} km/h)  "
      f"cruise power~{P_cr:.1f} kW  endurance~{endurance_min:.0f} min")
print(f"wing internal volume ~{wing_vol*1000:.0f} L (need 100 L) -> wet-wing OK")

# ---------------- schematic ----------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
# top view: delta planform (apex forward)
apex = (0, root); l = (-span/2, 0); r = (span/2, 0)
ax.add_patch(Polygon([apex, l, r], closed=True, fc="#cfe0f0", ec="k", lw=1.5))
# wet-wing water tanks (shaded band across the wing near CG)
for sgn in (-1, 1):
    ax.add_patch(Polygon([(sgn*0.15, root*0.78), (sgn*span*0.40, root*0.18),
                          (sgn*span*0.40, root*0.05), (sgn*0.15, root*0.30)],
                         closed=True, fc="#3a7bd5", ec="navy", alpha=0.8))
ax.text(0, root*0.42, "100 L water\n(wet wing)", ha="center", va="center",
        color="white", fontsize=9, fontweight="bold")
# CG, payload bay, compute, pusher prop, dump doors
cg = (0, root*0.42)
ax.plot(*cg, "o", ms=10, mfc="yellow", mec="k"); ax.text(cg[0]+0.1, cg[1], " CG", fontsize=8)
ax.text(0, root*0.86, "avionics +\nedge computer\n(policy ~22k params)", ha="center", fontsize=7)
ax.add_patch(plt.Circle((0, -0.05), 0.18, fc="gray", ec="k")); ax.text(0.3, -0.05, "pusher prop", fontsize=7, va="center")
ax.plot([-span*0.30, span*0.30], [0.05, 0.05], "r-", lw=4)
ax.text(0, 0.25, "trailing-edge dump doors", ha="center", color="r", fontsize=7)
# VTOL lift rotors (hybrid for truck launch)
for x, y in [(-span*0.28, root*0.6), (span*0.28, root*0.6), (-span*0.40, root*0.2), (span*0.40, root*0.2)]:
    ax.add_patch(plt.Circle((x, y), 0.28, fc="none", ec="green", ls="--", lw=1.2))
ax.text(span*0.40, root*0.2+0.4, "VTOL rotor", color="green", fontsize=7, ha="center")
ax.annotate("", (-span/2, -0.6), (span/2, -0.6), arrowprops=dict(arrowstyle="<->"))
ax.text(0, -0.85, f"span {span:.1f} m", ha="center", fontsize=8)
ax.annotate("", (span/2+0.5, 0), (span/2+0.5, root), arrowprops=dict(arrowstyle="<->"))
ax.text(span/2+0.7, root/2, f"root chord {root:.1f} m", rotation=90, va="center", fontsize=8)
ax.set_xlim(-span/2-1.5, span/2+1.8); ax.set_ylim(-1.2, root+0.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"Delta-wing water drone — top view\nMTOW {MTOW:.0f} kg, {S:.1f} m², LE sweep {le_sweep:.0f}°")

# spec text panel
ax2.axis("off")
spec = (f"DELTA-WING WATER DRONE\n\n"
        f"Payload:        100 L water (in wet wing)\n"
        f"MTOW:           {MTOW:.0f} kg\n"
        f"Wing area:      {S:.1f} m²   (loading {MTOW/S:.0f} kg/m²)\n"
        f"Span/root:      {span:.1f} m / {root:.1f} m\n"
        f"Stall / cruise: {vstall:.0f} / {vcr:.0f} m/s ({vcr*3.6:.0f} km/h)\n"
        f"Cruise power:   ~{P_cr:.1f} kW\n"
        f"Endurance:      ~{endurance_min:.0f} min/sortie\n"
        f"Wing volume:    ~{wing_vol*1000:.0f} L available (100 L used)\n\n"
        f"Launch/recover: hybrid VTOL (4 lift rotors)\n"
        f"                from the truck; transition to\n"
        f"                wing-borne cruise.\n\n"
        f"Drop:           trailing-edge dump (~100 L in\n"
        f"                ~1-2 s = 35-70 m fireline pass)\n\n"
        f"Compute:        edge SoC; per-drone policy is a\n"
        f"                ~22k-param MLP -> microseconds,\n"
        f"                <100 KB. Perception (fire detect)\n"
        f"                on a small AI SoC (~5-15 W).\n")
ax2.text(0.0, 0.98, spec, va="top", family="monospace", fontsize=9)
fig.savefig("docs/figures/delta_drone.png", dpi=120, bbox_inches="tight")
print("wrote docs/figures/delta_drone.png")
