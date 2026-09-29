# Autonomous firefighting drone swarm — market gap, sizing, and doctrine

*Order-of-magnitude business brief built from the simulation results in this repo.
All figures are estimates with the stated caveats — directional, not audited.*

## 1. The gap: the pieces exist, the swarm doesn't

| Capability | Maturity | Representative players (early 2026) |
|---|---|---|
| Fire detection / monitoring | Mature | Pano AI, Dryad Networks, OroraTech |
| Ignition drones (prescribed / back-burns) | Operational | Drone Amplified (IGNIS), Kestrel |
| Single autonomous aircraft suppression | Early | Rain (rain.aero), Lockheed/Sikorsky MATRIX Black Hawk |
| Heavy-lift logistics drones | Emerging | Parallel Flight Technologies |
| Manned air tankers (incumbent) | Mature | Coulson, Bridger Aerospace, Cal Fire fleet |

**No one operates a coordinated swarm of hundreds–thousands of drones that lays
retardant lines, compartmentalizes a fire, and attacks the trapped pockets.**
That doctrine — the one these sims show is decisive against high-intensity fires —
is whitespace. Blockers are regulatory (BVLOS swarms sharing airspace with manned
tankers), perception in smoke, and per-drone payload economics — not the concept.

## 2. TAM / SAM / SOM

These markets are real and large; the autonomous-swarm slice is greenfield.

- **Reference pools (annual):**
  - US federal wildfire **suppression**: ~$3–4.5 B (USFS + DOI, highly variable by year).
  - Cal Fire budget: ~$3–4 B.
  - Global **aerial firefighting** services: ~$3–4 B, ~6% CAGR.
  - US **retardant** spend: ~$0.2 B+ (tens of millions of gallons/yr).
  - **Economic losses** from US wildfires: ~$50–350 B/yr (2018 ≈ $148 B incl. indirect)
    — this is the *willingness-to-pay ceiling*, not a sales line.

- **TAM (~$10–15 B/yr):** global wildfire **suppression + aerial ops + retardant
  services**, the budget an autonomous suppression provider could displace or expand.
- **SAM (~$4–6 B/yr):** aerial suppression + retardant in fire-prone OECD regions
  with budgets and airspace to adopt first (US, Canada, Australia, S. Europe).
- **SOM (~$0.2–0.6 B/yr in 5–10 yr):** a few % of SAM via pilots with one or two
  agencies/utilities, expanding as regulation and trust catch up.
- **Adjacent expansion:** the same swarm-logistics + autonomy stack is dual-use
  (area survey, contested-logistics, counter-UAS). Large but separate; not counted above.

**Why the willingness-to-pay is favorable:** a fleet costs ~$0.2–0.5 B (below);
a single bad fire season costs tens of billions. Buyers compare against *damages
avoided*, not against tanker hourly rates.

## 3. Sizing — from the held "hardcore" crown-fire run

Scenario: 56 km² domain (7.5×7.5 km), ~18 km² active containment zone, crown fire
(peak ~1800 kW), wind 28 m/s, contained to **5% burned / 0% assets lost**.

| Resource | In the run | Notes |
|---|---|---|
| Drones (total) | ~10,000 (5,600 defender + 4,200 attack) | **over-provisioned** — +0% gain past this; binding constraint is *lay-rate vs fire-rate*, not count |
| Retardant line laid | ~96 km | ~13% of the land treated as firebreak |
| Retardant **product** | **~3–11 ML** | at 0.4–2.4 L/m² coverage; ~30 m real line band (vs 75 m modeled) trims footprint ~2–3× |
| Per-drone payload | 60 L | the lever from the payload sweep |
| Fleet capex | ~$0.2–0.5 B | ~10k heavy-payload drones @ $20–50k |

**Caveats:** 25 m grid makes the modeled line ~75 m wide (coarser than a real
~30 m aerial line); "retardant units" in the sim are abstract — the litre figures
are a real-world conversion of the treated footprint. This is a *medium* fire;
real events span <1 km² to >4,000 km², so continental events need sector triage,
not whole-front coverage. For context, the US already drops ~100+ ML of retardant
nationally per year — the swarm's edge is **speed and precision of delivery**
(≈96 km of line in hours vs. days of dozer + tanker work), not total product.

## 4. The doctrine ladder (what each fire severity requires)

From the breaking-point sweeps (`scripts/combined_sim.py`):

| Fire | What holds it |
|---|---|
| **Grass / brush** (≤ ~1000 kW) | a single containment **perimeter**, evenly-placed trucks |
| **Timber** (~1300 kW) | **compartmentalization** (lattice ≈ 750 m cells) + balanced attack to kill the trapped pockets |
| **Crown fire** (~1800 kW) | all the above **+ a heavy line** that can withstand the fire's intensity (wider firebreak / heavier retardant) — smaller cells and more drones do *nothing* until the line itself holds |

Two priority rules that fall out of the sims:
1. **Placement before count** — trucks must *surround* the fire; doubling drones on
   one side is wasted. Even 2 well-placed defender trucks beat a random 6.
2. **Line strength before subdivision** — against a crown fire, a line that can't
   survive the fire's kW is paper; subdividing into more pockets is pointless until
   each wall actually holds.

The defense is strong but **not invincible** — it is overcome by *outburning* it
(intensity > line strength), *outrunning* it (fire reaches the line before it
closes), and (not yet modeled) **ember spotting** over the line.
