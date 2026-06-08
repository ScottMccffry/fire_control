# Running real WRF-SFIRE and bridging it to the drone swarm

This documents an actual WRF-SFIRE (WSF) run performed for this project and how
its output feeds the drone-swarm `FireState` pipeline.

![Real WRF-SFIRE fire spread](figures/wrf_sfire_fire.png)

## What was run

WRF-SFIRE (`github.com/openwfm/WRF-SFIRE`) was **compiled from source** (serial
gfortran build) and the bundled **`hill_simple` ideal `em_fire` case** was run —
a coupled fire–atmosphere simulation that needs no external geog/met data:

- atmosphere grid 103×103×51 @ 50 m, with a **4× refined 412×412 fire mesh**
- single ignition line, fire-on-a-hill terrain, 10 minutes simulated
- result: ignition at t=120 s, flame front climbs the hill; by t=600 s the
  **fire area ≈ 56,000 m²** with **~320 MW** total heat output and **~16 kW/m²**
  peak ground heat flux.

Real WRF-SFIRE output variables (on the fire mesh) include `FIRE_AREA`,
`FGRNHFX` (ground heat flux), `LFN` (level-set front), `FUEL_FRAC`, fire-grid
coordinates `FXLAT`/`FXLONG`, and fire winds `UF`/`VF`.

## Bridging WSF output to the swarm: `sim/wrf_replay.py`

A live WRF-SFIRE run cannot be stepped interactively from Python. The practical
bridge is **replay**: run WRF-SFIRE to produce `wrfout` history files, then step
through them with `WRFReplaySimulator`, which converts each frame into the same
`FireState`/`FlameFont` objects the drone swarm already consumes — so the
heuristic swarm (and, with an adapter, the RL env) can operate on genuine
WRF-SFIRE physics instead of the mock cellular automaton.

```bash
# Reproduce (requires a compiled WRF-SFIRE + a run; see notes below)
python scripts/wrf_sfire_demo.py --wrfout-dir /opt/wrf_run --out fire.png
```

Mapping used: `FGRNHFX → heat_intensity`, `FIRE_AREA>0 → burned_area`,
`FXLAT/FXLONG → bounds/perimeter`. For ideal cases the fire-grid coordinates are
not geographic, so pass `fire_mesh_res` explicitly (here 50 m / sr 4 = 12.5 m).

## Build notes (this environment)

- Containerized WRF-SFIRE was **not** possible here: Docker ran (nested daemon),
  but image pulls were blocked — Docker Hub anonymous rate limit + ghcr.io/quay.io
  returning 403 under the environment's network policy.
- From source worked: Ubuntu **main** apt archive is reachable (PPAs are not),
  GitHub is reachable. Build deps: `gfortran`, `gcc`, `libnetcdf-dev`,
  `libnetcdff-dev`, `libhdf5-dev`, `m4`, `csh`. A unified `$NETCDF` prefix was
  created because Ubuntu splits NetCDF headers/libs across paths WRF's
  `configure` doesn't expect.

## Closed-loop suppression caveat

Replay is one-way: drones reading these frames cannot change a precomputed fire.
True closed-loop suppression (drone water drops altering WRF-SFIRE spread)
requires running WRF-SFIRE live with fuel/moisture modification fed back each
coupling step — out of scope here, but the `FireState` bridge is the first step.
