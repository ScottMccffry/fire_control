# WRF-SFIRE Integration Guide

## Overview

This document outlines the path from the current mock simulator to full WRF-SFIRE integration for realistic wildfire simulation.

## Phase 1: Current State (Mock Simulator) ✅

**Status**: Ready for RL training

### What You Have Now
```
MockFireSimulator
├── Cellular automata fire spread
├── Wind and slope effects
├── Fuel moisture modeling
├── Water suppression
└── Fast execution (~1000 steps/second)
```

### Use For
- Validating RL algorithm design
- Hyperparameter tuning
- Quick experimentation
- Baseline policy development

### Limitations
- Simplified physics (not Rothermel)
- No atmosphere-fire coupling
- No spotting (ember transport)
- No real terrain/fuel data

---

## Phase 2: Rothermel Physics (Intermediate)

**Status**: Code exists (`sim/fire_physics.py`), needs testing

### What It Adds
```
ImprovedFireSimulator (Rothermel-based)
├── Physics-based spread rate equations
├── 13 standard fuel models (Anderson)
├── Fireline intensity calculation
├── Flame length estimation
└── More realistic fire behavior
```

### How to Use
```python
from sim.fire_physics import ImprovedFireSimulator, FuelModel

sim = ImprovedFireSimulator(
    grid_size=(100, 100),
    use_rothermel=True
)
sim.fuel_model[:] = FuelModel.CHAPARRAL  # Set fuel type
sim.set_ignition_points([(50, 50)])
sim.set_weather(wind_speed=10.0, wind_direction=45.0)

for _ in range(100):
    state = sim.step()
```

### When to Use
- Need more realistic fire behavior
- Training policies that should transfer to real conditions
- Before investing in full WRF-SFIRE

---

## Phase 3: WRF-SFIRE Integration (Production)

**Status**: Interface exists, requires WRF installation

### What WRF-SFIRE Provides
```
WRF-SFIRE
├── Coupled atmosphere-fire simulation
├── Real weather data integration
├── Terrain-following coordinates
├── Fire-induced winds (pyroconvection)
├── Spotting and crown fire
└── Validated against real fires
```

### Installation Requirements

#### 1. System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    gfortran \
    cpp \
    gcc \
    g++ \
    libnetcdf-dev \
    libnetcdff-dev \
    libpng-dev \
    libjasper-dev \
    libhdf5-dev \
    csh \
    m4
```

#### 2. Install WRF
```bash
# Download WRF
git clone https://github.com/wrf-model/WRF.git
cd WRF

# Configure (choose gfortran/gcc option)
./configure  # Select option 34 for gfortran/gcc

# Compile (takes 1-2 hours)
./compile em_real 2>&1 | tee compile.log
```

#### 3. Install WRF-SFIRE
```bash
# WRF-SFIRE is a branch of WRF
git clone https://github.com/openwfm/WRF-SFIRE.git
cd WRF-SFIRE
./configure
./compile em_fire 2>&1 | tee compile.log
```

#### 4. Install wrfxpy (Python wrapper)
```bash
git clone https://github.com/openwfm/wrfxpy.git
cd wrfxpy
pip install -e .

# Configure paths
cp etc/conf.json.initial etc/conf.json
# Edit conf.json with paths to WRF executables
```

#### 5. Download Geographic Data
```bash
# WPS Geographic Data (~30GB)
wget https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html

# LANDFIRE fuel data for USA
# https://landfire.gov/fuel.php
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your RL Training Loop                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐      ┌─────────────┐      ┌────────────┐ │
│   │   Agent     │─────▶│  Environment │─────▶│  Simulator │ │
│   │  (Policy)   │◀─────│  (Gym Env)  │◀─────│            │ │
│   └─────────────┘      └─────────────┘      └─────┬──────┘ │
│                                                    │        │
│         ┌──────────────────────────────────────────┘        │
│         ▼                                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Simulator Backend (swappable)           │  │
│   ├─────────────────────────────────────────────────────┤  │
│   │  Phase 1: MockFireSimulator (current)               │  │
│   │  Phase 2: ImprovedFireSimulator (Rothermel)         │  │
│   │  Phase 3: WRFFireSimulator (full physics)           │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Modifying the Environment for WRF

```python
# In envs/fire_control_env.py, the simulator is swappable:

class FireControlEnv(gym.Env):
    def __init__(self, simulator_type='mock', ...):
        if simulator_type == 'mock':
            self.fire_sim = MockFireSimulator(...)
        elif simulator_type == 'rothermel':
            self.fire_sim = ImprovedFireSimulator(...)
        elif simulator_type == 'wrf':
            self.fire_sim = WRFFireSimulator(...)
```

### WRF-SFIRE Running Time

| Grid Size | Drones | Real-Time Ratio | Training Feasibility |
|-----------|--------|-----------------|---------------------|
| 100x100   | 100    | ~10x slower     | Possible (slow)     |
| 500x500   | 1000   | ~100x slower    | Difficult           |
| 1000x1000 | 5000   | ~500x slower    | Impractical         |

**Recommendation**: Use WRF-SFIRE for:
- Final validation of trained policies
- Generating realistic scenarios
- Transfer learning datasets
- NOT for training loop (too slow)

---

## Recommended Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  DEVELOPMENT PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PROTOTYPE (Current)                                      │
│     └─▶ MockFireSimulator                                   │
│     └─▶ Train initial policies                              │
│     └─▶ Validate approach works                             │
│                                                              │
│  2. REFINE                                                   │
│     └─▶ ImprovedFireSimulator (Rothermel)                   │
│     └─▶ Fine-tune policies                                  │
│     └─▶ Test edge cases                                     │
│                                                              │
│  3. VALIDATE                                                 │
│     └─▶ WRF-SFIRE for specific scenarios                    │
│     └─▶ Compare policy behavior                             │
│     └─▶ Generate realistic test cases                       │
│                                                              │
│  4. DEPLOY                                                   │
│     └─▶ Run trained policy against WRF scenarios            │
│     └─▶ Or use hybrid (fast mock + WRF validation)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Resources

### WRF-SFIRE Documentation
- [WRF-SFIRE Users Guide](https://wiki.openwfm.org/wiki/WRF-SFIRE)
- [wrfxpy Documentation](https://github.com/openwfm/wrfxpy)
- [Fire Behavior Tutorial](https://wiki.openwfm.org/wiki/Fire_behavior)

### Fire Science References
- Rothermel, R.C. (1972) "A Mathematical Model for Predicting Fire Spread in Wildland Fuels"
- Andrews, P.L. (2018) "The Rothermel Surface Fire Spread Model and Associated Developments"

### Data Sources
- [LANDFIRE](https://landfire.gov/) - Fuel and vegetation data for USA
- [CAL FIRE](https://www.fire.ca.gov/incidents) - California fire history
- [NOAA HRRR](https://rapidrefresh.noaa.gov/hrrr/) - High-resolution weather data

---

## Next Steps for You

1. **Now**: Train with MockFireSimulator, validate RL approach
2. **Next**: Test ImprovedFireSimulator, compare policy performance
3. **Later**: Install WRF-SFIRE for final validation (if needed)

The current mock simulator is sufficient for developing and testing drone coordination strategies. WRF-SFIRE is primarily valuable for final validation against realistic physics.
