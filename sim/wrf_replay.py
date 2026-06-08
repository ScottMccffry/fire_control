"""
Replay real WRF-SFIRE output as the project's FireState stream.

A live WRF-SFIRE run cannot be stepped interactively from Python (it is a
standalone Fortran MPI/serial model). The practical bridge is *replay*: run
WRF-SFIRE to produce ``wrfout`` history files, then step through those frames
here, converting each into the same :class:`FireState`/:class:`FlameFont`
objects the drone swarm already consumes (see ``sim/mock_fire.py``).

This lets the heuristic swarm — and, with an adapter, the RL env — operate on
genuine WRF-SFIRE fire physics instead of the mock cellular automaton.

Fire fields used (on the refined fire mesh):
  * ``FGRNHFX``        ground heat flux  [W/m^2]  -> heat_intensity
  * ``FIRE_AREA``      burned fraction per cell    -> burned_area (>0)
  * ``FXLAT``/``FXLONG`` fire-grid coordinates     -> bounds / perimeter
"""

from typing import List, Optional, Tuple
from glob import glob
from datetime import datetime, timedelta
import os

import numpy as np

from .fire_state import FireState, FlameFont


class WRFReplaySimulator:
    """Step through WRF-SFIRE ``wrfout`` frames as a FireState stream."""

    def __init__(self, wrfout_dir: str, pattern: str = "wrfout_d01_*",
                 coarsen: int = 1, domain_name: str = "wrf_sfire",
                 fire_mesh_res: Optional[float] = None):
        """
        Args:
            wrfout_dir: Directory containing wrfout history files.
            pattern: Glob for the history files (sorted = time order).
            coarsen: Optional integer factor to block-average the fire mesh
                     down (e.g. 4 turns a 412x412 mesh into 103x103).
            domain_name: Label for the FireState.
            fire_mesh_res: Fire-cell size in metres. If None, inferred from the
                     fire-grid coordinates (with a sanity fallback). Ideal cases
                     have non-geographic coordinates, so pass it explicitly
                     (e.g. atmos dx / sr_x = 50/4 = 12.5).
        """
        self.files = sorted(glob(os.path.join(wrfout_dir, pattern)))
        if not self.files:
            raise FileNotFoundError(f"no wrfout files matching {pattern} in {wrfout_dir}")
        self.coarsen = max(1, int(coarsen))
        self.domain_name = domain_name
        self.fire_mesh_res = fire_mesh_res
        self.current_step = 0
        self.start_time = datetime.now()
        self._history: List[FlameFont] = []
        # Peak heat flux across the run, for normalization by consumers.
        self.max_heat_intensity = 1.0
        self.ignition_threshold = 100.0  # W/m^2 considered "active"

    def __len__(self) -> int:
        return len(self.files)

    def _coarsen(self, arr: np.ndarray) -> np.ndarray:
        c = self.coarsen
        if c == 1:
            return arr
        h, w = arr.shape
        h2, w2 = (h // c) * c, (w // c) * c
        return arr[:h2, :w2].reshape(h2 // c, c, w2 // c, c).mean(axis=(1, 3))

    def _read_frame(self, idx: int) -> FlameFont:
        import xarray as xr
        with xr.open_dataset(self.files[idx], decode_times=False) as ds:
            heat = ds["FGRNHFX"].isel(Time=0).values.astype(float)      # W/m^2
            area = ds["FIRE_AREA"].isel(Time=0).values.astype(float)    # burned frac
            lat = ds["FXLAT"].isel(Time=0).values.astype(float)
            lon = ds["FXLONG"].isel(Time=0).values.astype(float)

        heat = self._coarsen(heat)
        burned = (self._coarsen((area > 0).astype(float)) > 0).astype(int)
        self.max_heat_intensity = max(self.max_heat_intensity, float(heat.max()))

        bounds = (float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max()))
        perimeter = self._perimeter(burned, bounds)

        # Fire-mesh resolution in metres: explicit override, else inferred.
        res = (self.fire_mesh_res * self.coarsen if self.fire_mesh_res
               else self._mesh_resolution(lat, lon, heat.shape))

        return FlameFont(
            burned_area=burned,
            heat_intensity=heat,
            fire_perimeter=perimeter,
            timestamp=self.start_time + timedelta(seconds=idx),
            time_step=idx,
            grid_resolution=res,
            bounds=bounds,
        )

    @staticmethod
    def _perimeter(burned: np.ndarray, bounds) -> List[Tuple[float, float]]:
        from scipy import ndimage
        if not burned.any():
            return []
        edge = ndimage.binary_dilation(burned) & ~burned.astype(bool)
        rr, cc = np.where(edge)
        min_lat, min_lon, max_lat, max_lon = bounds
        h, w = burned.shape
        out = []
        for r, c in zip(rr, cc):
            out.append((min_lat + (r / h) * (max_lat - min_lat),
                        min_lon + (c / w) * (max_lon - min_lon)))
        return out

    @staticmethod
    def _mesh_resolution(lat, lon, shape) -> float:
        # Infer metres/cell from latitude spacing; ideal cases have
        # non-geographic coords, so clamp implausible values to a nominal 12.5 m.
        span_lat = float(lat.max() - lat.min())
        meters = (span_lat * 111000.0) / max(1, shape[0]) if span_lat > 1e-9 else 0.0
        return meters if 0.1 <= meters <= 5000.0 else 12.5

    def step(self) -> Optional[FireState]:
        """Advance to the next frame; returns its FireState (None when done)."""
        if self.current_step >= len(self.files):
            return None
        ff = self._read_frame(self.current_step)
        self._history.append(ff)
        self.current_step += 1
        return FireState(
            current_flame_front=ff,
            flame_front_history=list(self._history),
            weather_data={}, terrain_data={}, fuel_data={},
            simulation_id=os.path.basename(self.files[self.current_step - 1]),
            domain_name=self.domain_name,
            start_time=self.start_time,
            current_time=ff.timestamp,
        )

    def reset(self):
        self.current_step = 0
        self._history = []
