"""
Interface to WRF-SFIRE wildfire simulation via wrfxpy.

This module provides a Python interface to run WRF-SFIRE simulations
and extract flame front data for drone coordination algorithms.
"""

import os
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
from pathlib import Path

from .fire_state import FireState, FlameFont


class WRFFireSimulator:
    """Interface to WRF-SFIRE wildfire simulation system."""
    
    def __init__(self, 
                 wrfxpy_path: str,
                 domain_config: Dict[str, Any],
                 output_dir: str = "./wrf_output"):
        """
        Initialize WRF-SFIRE simulator.
        
        Args:
            wrfxpy_path: Path to wrfxpy installation
            domain_config: Domain configuration (bounds, resolution, etc.)
            output_dir: Directory for simulation output files
        """
        self.wrfxpy_path = Path(wrfxpy_path)
        self.domain_config = domain_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Simulation state
        self.current_simulation_id: Optional[str] = None
        self.simulation_running = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Validate wrfxpy installation
        self._validate_wrfxpy()
    
    def _validate_wrfxpy(self):
        """Validate that wrfxpy is properly installed."""
        if not self.wrfxpy_path.exists():
            raise FileNotFoundError(f"wrfxpy not found at {self.wrfxpy_path}")
        
        required_files = ["src/wrf", "src/wps"]
        for req_file in required_files:
            if not (self.wrfxpy_path / req_file).exists():
                self.logger.warning(f"Expected WRF component not found: {req_file}")
    
    def setup_domain(self, domain_name: str, config: Dict[str, Any]) -> str:
        """
        Set up a simulation domain for fire modeling.
        
        Args:
            domain_name: Name identifier for the domain
            config: Domain configuration including:
                - bounds: (min_lat, min_lon, max_lat, max_lon)
                - resolution: Grid resolution in meters
                - ignition_points: List of (lat, lon, start_time) tuples
                - terrain_data: Path to elevation data
                - fuel_data: Path to fuel type data
                - weather_data: Path to weather forcing data
        
        Returns:
            Domain ID for reference
        """
        domain_id = f"{domain_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        domain_dir = self.output_dir / domain_id
        domain_dir.mkdir(exist_ok=True)
        
        # Generate WPS namelist
        self._generate_wps_namelist(domain_id, config)
        
        # Generate WRF namelist
        self._generate_wrf_namelist(domain_id, config)
        
        # Set up fire domain
        self._setup_fire_domain(domain_id, config)
        
        self.logger.info(f"Domain {domain_id} set up successfully")
        return domain_id
    
    def _generate_wps_namelist(self, domain_id: str, config: Dict[str, Any]):
        """Generate WPS namelist.wps file for domain preprocessing."""
        bounds = config['bounds']
        min_lat, min_lon, max_lat, max_lon = bounds
        
        namelist_wps = f"""&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '{config.get('start_date', '2023-01-01_00:00:00')}',
 end_date   = '{config.get('end_date', '2023-01-02_00:00:00')}',
 interval_seconds = 3600,
 io_form_geogrid = 2,
/

&geogrid
 parent_id         =   1,
 parent_grid_ratio =   1,
 i_parent_start    =   1,
 j_parent_start    =   1,
 e_we              =  {config.get('grid_x', 100)},
 e_sn              =  {config.get('grid_y', 100)},
 geog_data_res     = '30s',
 dx = {config.get('resolution', 1000)},
 dy = {config.get('resolution', 1000)},
 map_proj = 'lambert',
 ref_lat   = {(min_lat + max_lat) / 2},
 ref_lon   = {(min_lon + max_lon) / 2},
 truelat1  = {min_lat},
 truelat2  = {max_lat},
 stand_lon = {(min_lon + max_lon) / 2},
 geog_data_path = '{self.wrfxpy_path}/data/geog/'
/

&ungrib
 out_format = 'WPS',
 prefix = 'FILE',
/

&metgrid
 fg_name = 'FILE'
 io_form_metgrid = 2,
/
"""
        
        namelist_path = self.output_dir / domain_id / "namelist.wps"
        with open(namelist_path, 'w') as f:
            f.write(namelist_wps)
    
    def _generate_wrf_namelist(self, domain_id: str, config: Dict[str, Any]):
        """Generate WRF namelist.input file for simulation."""
        # This is a simplified version - full WRF namelists are quite complex
        namelist_input = f"""&time_control
 run_days                = 0,
 run_hours               = {config.get('run_hours', 24)},
 run_minutes             = 0,
 run_seconds             = 0,
 start_year              = 2023,
 start_month             = 01,
 start_day               = 01,
 start_hour              = 00,
 start_minute            = 00,
 start_second            = 00,
 end_year                = 2023,
 end_month               = 01,
 end_day                 = 02,
 end_hour                = 00,
 end_minute              = 00,
 end_second              = 00,
 interval_seconds        = 3600,
 input_from_file         = .true.,
 history_interval        = 60,
 frames_per_outfile      = 1,
 restart                 = .false.,
 io_form_history         = 2,
 io_form_restart         = 2,
 io_form_input           = 2,
 io_form_boundary        = 2,
/

&domains
 time_step               = {config.get('time_step', 60)},
 time_step_fract_num     = 0,
 time_step_fract_den     = 1,
 max_dom                 = 1,
 e_we                    = {config.get('grid_x', 100)},
 e_sn                    = {config.get('grid_y', 100)},
 e_vert                  = 35,
 dx                      = {config.get('resolution', 1000)},
 dy                      = {config.get('resolution', 1000)},
 grid_id                 = 1,
 parent_id               = 0,
 i_parent_start          = 1,
 j_parent_start          = 1,
 parent_grid_ratio       = 1,
 parent_time_step_ratio  = 1,
 feedback                = 1,
 smooth_option           = 0,
/

&physics
 mp_physics              = 3,
 ra_lw_physics           = 1,
 ra_sw_physics           = 1,
 radt                    = 30,
 sf_sfclay_physics       = 1,
 sf_surface_physics      = 2,
 bl_pbl_physics          = 1,
 bldt                    = 0,
 cu_physics              = 1,
 cudt                    = 5,
 isfflx                  = 1,
 ifsnow                  = 1,
 icloud                  = 1,
 surface_input_source    = 1,
 num_soil_layers         = 4,
 sf_urban_physics        = 0,
/

&fire
 ifire                   = 2,
 fire_fuel_read          = 1,
 fire_fuel_cat           = 1,
 fire_num_ignitions      = {len(config.get('ignition_points', []))},
 fire_ignition_ros1      = 0.1,
 fire_ignition_start_x1  = {','.join([str(p[1]) for p in config.get('ignition_points', [])])},
 fire_ignition_start_y1  = {','.join([str(p[0]) for p in config.get('ignition_points', [])])},
 fire_ignition_end_x1    = {','.join([str(p[1]) for p in config.get('ignition_points', [])])},
 fire_ignition_end_y1    = {','.join([str(p[0]) for p in config.get('ignition_points', [])])},
 fire_ignition_radius1   = {','.join(['50' for _ in config.get('ignition_points', [])])},
 fire_ignition_start_time1 = {','.join(['0' for _ in config.get('ignition_points', [])])},
 fire_ignition_end_time1   = {','.join(['1' for _ in config.get('ignition_points', [])])},
/
"""
        
        namelist_path = self.output_dir / domain_id / "namelist.input"
        with open(namelist_path, 'w') as f:
            f.write(namelist_input)
    
    def _setup_fire_domain(self, domain_id: str, config: Dict[str, Any]):
        """Set up fire-specific domain files."""
        # Create fire input files (fuel maps, ignition, etc.)
        # This would involve processing terrain and fuel data
        pass
    
    def run_simulation(self, domain_id: str, 
                      duration_hours: int = 24,
                      output_interval_minutes: int = 60) -> str:
        """
        Run WRF-SFIRE simulation for specified domain.
        
        Args:
            domain_id: Domain identifier from setup_domain()
            duration_hours: Simulation duration in hours
            output_interval_minutes: Output frequency in minutes
            
        Returns:
            Simulation run ID
        """
        simulation_id = f"{domain_id}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / simulation_id
        run_dir.mkdir(exist_ok=True)
        
        self.current_simulation_id = simulation_id
        self.simulation_running = True
        
        try:
            # Run WPS preprocessing
            self._run_wps(domain_id, run_dir)
            
            # Run WRF simulation
            self._run_wrf(domain_id, run_dir, duration_hours)
            
            self.simulation_running = False
            self.logger.info(f"Simulation {simulation_id} completed successfully")
            
        except Exception as e:
            self.simulation_running = False
            self.logger.error(f"Simulation {simulation_id} failed: {str(e)}")
            raise
        
        return simulation_id
    
    def _run_wps(self, domain_id: str, run_dir: Path):
        """Run WRF Preprocessing System."""
        domain_dir = self.output_dir / domain_id
        
        # Copy namelist to run directory
        subprocess.run([
            "cp", str(domain_dir / "namelist.wps"), str(run_dir)
        ], check=True)
        
        # Run geogrid
        cmd = [str(self.wrfxpy_path / "WPS" / "geogrid.exe")]
        subprocess.run(cmd, cwd=run_dir, check=True)
        
        # Run ungrib (requires GRIB data)
        # cmd = [str(self.wrfxpy_path / "WPS" / "ungrib.exe")]
        # subprocess.run(cmd, cwd=run_dir, check=True)
        
        # Run metgrid
        # cmd = [str(self.wrfxpy_path / "WPS" / "metgrid.exe")]
        # subprocess.run(cmd, cwd=run_dir, check=True)
    
    def _run_wrf(self, domain_id: str, run_dir: Path, duration_hours: int):
        """Run WRF simulation with fire module."""
        domain_dir = self.output_dir / domain_id
        
        # Copy namelist to run directory
        subprocess.run([
            "cp", str(domain_dir / "namelist.input"), str(run_dir)
        ], check=True)
        
        # Run real.exe for initialization
        cmd = [str(self.wrfxpy_path / "WRFV3" / "main" / "real.exe")]
        subprocess.run(cmd, cwd=run_dir, check=True)
        
        # Run wrf.exe for simulation
        cmd = [str(self.wrfxpy_path / "WRFV3" / "main" / "wrf.exe")]
        subprocess.run(cmd, cwd=run_dir, check=True)
    
    def get_fire_state(self, simulation_id: str, time_step: int) -> FireState:
        """
        Extract fire state from WRF output at specified time step.
        
        Args:
            simulation_id: Simulation identifier
            time_step: Time step index to extract
            
        Returns:
            FireState object with flame front data
        """
        output_file = self.output_dir / simulation_id / f"wrfout_d01_{time_step:04d}.nc"
        
        if not output_file.exists():
            raise FileNotFoundError(f"WRF output file not found: {output_file}")
        
        # Read WRF-SFIRE output
        with xr.open_dataset(output_file) as ds:
            # Extract fire variables
            fire_area = ds['FIRE_AREA'].values if 'FIRE_AREA' in ds else None
            grnhfx = ds['GRNHFX'].values if 'GRNHFX' in ds else None  # Heat flux
            
            # Get grid info
            xlat = ds['XLAT'].values
            xlon = ds['XLONG'].values
            
            # Create flame front
            if fire_area is not None:
                burned_area = (fire_area > 0).astype(int)
                heat_intensity = grnhfx if grnhfx is not None else np.zeros_like(burned_area)
                
                # Extract fire perimeter
                fire_perimeter = self._extract_fire_perimeter(burned_area, xlat, xlon)
                
                # Create FlameFont object
                flame_front = FlameFont(
                    burned_area=burned_area,
                    heat_intensity=heat_intensity,
                    fire_perimeter=fire_perimeter,
                    timestamp=datetime.now(),  # Should extract from file
                    time_step=time_step,
                    grid_resolution=float(self.domain_config.get('resolution', 1000)),
                    bounds=(float(xlat.min()), float(xlon.min()), 
                           float(xlat.max()), float(xlon.max()))
                )
                
                # Create FireState
                fire_state = FireState(
                    current_flame_front=flame_front,
                    flame_front_history=[],
                    weather_data={},
                    terrain_data={},
                    fuel_data={},
                    simulation_id=simulation_id,
                    domain_name=self.domain_config.get('domain_name', 'unknown'),
                    start_time=datetime.now(),
                    current_time=datetime.now()
                )
                
                return fire_state
        
        raise ValueError(f"Could not extract fire state from {output_file}")
    
    def _extract_fire_perimeter(self, burned_area: np.ndarray, 
                               xlat: np.ndarray, xlon: np.ndarray) -> List[Tuple[float, float]]:
        """Extract fire perimeter coordinates from burned area grid."""
        from scipy import ndimage
        
        # Find fire perimeter using edge detection
        edges = ndimage.binary_dilation(burned_area) & ~burned_area
        perimeter_coords = np.where(edges)
        
        # Convert to lat/lon coordinates
        perimeter_latlon = []
        for i, j in zip(perimeter_coords[0], perimeter_coords[1]):
            if i < xlat.shape[0] and j < xlat.shape[1]:
                lat = float(xlat[i, j])
                lon = float(xlon[i, j])
                perimeter_latlon.append((lat, lon))
        
        return perimeter_latlon
    
    def is_simulation_running(self) -> bool:
        """Check if a simulation is currently running."""
        return self.simulation_running
    
    def stop_simulation(self):
        """Stop the currently running simulation."""
        if self.simulation_running:
            # This would involve killing WRF processes
            self.simulation_running = False
            self.logger.info("Simulation stopped by user request") 