"""
Real-time visualization for wildfire and drone swarm simulation.

This module provides interactive visualization of fire spread patterns
and drone swarm movements for monitoring and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Tuple, List, Dict, Any, Optional
import logging
from datetime import datetime

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable


class RealtimeVisualizer:
    """
    Real-time visualization of wildfire simulation and drone swarm.
    
    Displays:
    - Fire heat intensity as a heatmap
    - Drone positions and states
    - Water refill locations
    - Performance metrics
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (100, 100),
                 bounds: Tuple[float, float, float, float] = (34.0, -119.0, 35.0, -118.0),
                 update_interval: int = 200):
        """
        Initialize real-time visualizer.
        
        Args:
            grid_size: (rows, cols) of the simulation grid
            bounds: (min_lat, min_lon, max_lat, max_lon) of simulation area
            update_interval: Visualization update interval in milliseconds
        """
        self.grid_size = grid_size
        self.bounds = bounds
        self.update_interval = update_interval
        
        # Setup matplotlib
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Wildfire & Drone Swarm Simulation', fontsize=16)
        
        # Main fire/drone visualization
        self.ax_main = self.axes[0, 0]
        self.ax_main.set_title('Fire Intensity & Drone Positions')
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        
        # Fire metrics over time
        self.ax_fire = self.axes[0, 1]
        self.ax_fire.set_title('Fire Metrics Over Time')
        self.ax_fire.set_xlabel('Time Step')
        self.ax_fire.set_ylabel('Burned Area (km²)')
        
        # Drone metrics over time
        self.ax_drones = self.axes[1, 0]
        self.ax_drones.set_title('Drone Metrics Over Time')
        self.ax_drones.set_xlabel('Time Step')
        self.ax_drones.set_ylabel('Count')
        
        # Performance dashboard
        self.ax_perf = self.axes[1, 1]
        self.ax_perf.set_title('Performance Dashboard')
        self.ax_perf.axis('off')
        
        # Initialize fire heatmap
        self.fire_heatmap = None
        self.drone_scatter = None
        self.refill_scatter = None
        
        # Data storage for time series plots
        self.time_steps = []
        self.fire_data = {'burned_area': [], 'active_fires': []}
        self.drone_data = {'operational': [], 'water_drops': [], 'missions': []}
        
        # Color schemes
        self.fire_colors = plt.cm.hot
        self.drone_colors = {
            'idle': 'blue',
            'flying_to_fire': 'orange',
            'dropping_water': 'red',
            'flying_to_refill': 'yellow',
            'refilling': 'green',
            'out_of_battery': 'gray'
        }
        
        self.logger = logging.getLogger("RealtimeVisualizer")
        self.logger.info("Real-time visualizer initialized")
    
    def update(self, fire_state: Any, drone_swarm: Any, step: int):
        """
        Update visualization with current simulation state.
        
        Args:
            fire_state: Current fire simulation state
            drone_swarm: Current drone swarm state
            step: Current simulation step
        """
        try:
            # Clear previous plots
            self.ax_main.clear()
            
            # Update fire visualization
            self._update_fire_display(fire_state)
            
            # Update drone visualization
            self._update_drone_display(drone_swarm)
            
            # Update time series data
            self._update_time_series(fire_state, drone_swarm, step)
            
            # Update performance dashboard
            self._update_performance_dashboard(drone_swarm)
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.logger.error(f"Visualization update error: {e}")
    
    def _update_fire_display(self, fire_state: Any):
        """Update fire intensity heatmap."""
        heat_intensity = fire_state.current_flame_front.heat_intensity
        
        # Create coordinate grids for proper geographic display
        min_lat, min_lon, max_lat, max_lon = self.bounds
        
        # Display fire intensity
        im = self.ax_main.imshow(
            heat_intensity, 
            extent=[min_lon, max_lon, min_lat, max_lat],
            cmap=self.fire_colors,
            alpha=0.7,
            aspect='auto',
            origin='lower'
        )
        
        # Add colorbar for fire intensity
        if not hasattr(self, '_fire_colorbar'):
            self._fire_colorbar = self.fig.colorbar(im, ax=self.ax_main, label='Heat Intensity (kW/m²)')
        
        # Draw fire perimeter if available
        if fire_state.current_flame_front.fire_perimeter:
            perimeter = fire_state.current_flame_front.fire_perimeter
            if len(perimeter) > 0:
                lats, lons = zip(*perimeter)
                self.ax_main.plot(lons, lats, 'r-', linewidth=2, label='Fire Perimeter')
        
        self.ax_main.set_title('Fire Intensity & Drone Positions')
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
    
    def _update_drone_display(self, drone_swarm: Any):
        """Update drone positions and states."""
        # Group drones by state
        drone_positions = {}
        for state in self.drone_colors.keys():
            drone_positions[state] = {'lats': [], 'lons': []}
        
        # Sample a subset of drones for visualization (performance)
        sample_size = min(200, len(drone_swarm.drones))
        drone_items = list(drone_swarm.drones.items())
        step = max(1, len(drone_items) // sample_size)
        sampled_drones = drone_items[::step]
        
        for drone_id, drone in sampled_drones:
            state = drone.state.value
            if state in drone_positions:
                lat, lon = drone.position
                drone_positions[state]['lats'].append(lat)
                drone_positions[state]['lons'].append(lon)
        
        # Plot drones by state
        for state, color in self.drone_colors.items():
            lats = drone_positions[state]['lats']
            lons = drone_positions[state]['lons']
            
            if lats and lons:
                self.ax_main.scatter(
                    lons, lats, 
                    c=color, 
                    s=10, 
                    alpha=0.6, 
                    label=f'{state.replace("_", " ").title()} ({len(lats)})'
                )
        
        # Plot refill locations
        if drone_swarm.refill_locations:
            refill_lats, refill_lons = zip(*drone_swarm.refill_locations)
            self.ax_main.scatter(
                refill_lons, refill_lats,
                c='cyan',
                s=100,
                marker='s',
                edgecolors='black',
                label='Refill Stations'
            )
        
        # Add legend
        self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _update_time_series(self, fire_state: Any, drone_swarm: Any, step: int):
        """Update time series plots."""
        self.time_steps.append(step)
        
        # Fire metrics
        burned_area_km2 = fire_state.current_flame_front.get_fire_size() / 1e6
        active_fires = len(fire_state.current_flame_front.get_active_fire_cells())
        
        self.fire_data['burned_area'].append(burned_area_km2)
        self.fire_data['active_fires'].append(active_fires)
        
        # Drone metrics
        swarm_status = drone_swarm.get_swarm_status()
        metrics = swarm_status['metrics']
        
        self.drone_data['operational'].append(metrics['operational_drones'])
        self.drone_data['water_drops'].append(metrics['total_water_dropped_liters'])
        self.drone_data['missions'].append(metrics['total_missions_completed'])
        
        # Plot fire metrics
        self.ax_fire.clear()
        self.ax_fire.plot(self.time_steps, self.fire_data['burned_area'], 'r-', label='Burned Area (km²)')
        ax_fire_twin = self.ax_fire.twinx()
        ax_fire_twin.plot(self.time_steps, self.fire_data['active_fires'], 'orange', label='Active Fire Cells')
        
        self.ax_fire.set_xlabel('Time Step')
        self.ax_fire.set_ylabel('Burned Area (km²)', color='red')
        ax_fire_twin.set_ylabel('Active Fire Cells', color='orange')
        self.ax_fire.set_title('Fire Progression')
        
        # Plot drone metrics
        self.ax_drones.clear()
        self.ax_drones.plot(self.time_steps, self.drone_data['operational'], 'b-', label='Operational Drones')
        self.ax_drones.plot(self.time_steps, [w/100 for w in self.drone_data['water_drops']], 'g-', label='Water Dropped (x100L)')
        self.ax_drones.plot(self.time_steps, [m/10 for m in self.drone_data['missions']], 'purple', label='Missions (x10)')
        
        self.ax_drones.set_xlabel('Time Step')
        self.ax_drones.set_ylabel('Count')
        self.ax_drones.set_title('Drone Performance')
        self.ax_drones.legend()
        
        # Keep only recent data for performance
        max_points = 100
        if len(self.time_steps) > max_points:
            self.time_steps = self.time_steps[-max_points:]
            for key in self.fire_data:
                self.fire_data[key] = self.fire_data[key][-max_points:]
            for key in self.drone_data:
                self.drone_data[key] = self.drone_data[key][-max_points:]
    
    def _update_performance_dashboard(self, drone_swarm: Any):
        """Update performance dashboard with key metrics."""
        self.ax_perf.clear()
        self.ax_perf.axis('off')
        
        # Get current metrics
        swarm_status = drone_swarm.get_swarm_status()
        metrics = swarm_status['metrics']
        
        # Create performance text
        dashboard_text = [
            f"Simulation Time: {swarm_status['runtime_minutes']:.1f} min",
            "",
            f"SWARM STATUS",
            f"Total Drones: {metrics['swarm_size']:,}",
            f"Operational: {metrics['operational_drones']:,} ({metrics['operational_percentage']:.1f}%)",
            f"Avg Battery: {metrics['average_battery']:.1%}",
            f"Avg Water: {metrics['average_water']:.1%}",
            "",
            f"MISSION PERFORMANCE",
            f"Water Dropped: {metrics['total_water_dropped_liters']:.0f} L",
            f"Fires Suppressed: {metrics['total_fires_suppressed']}",
            f"Missions Completed: {metrics['total_missions_completed']:,}",
            f"Flight Time: {metrics['total_flight_time_hours']:.1f} hrs",
            "",
            f"COORDINATION",
            f"Active Assignments: {metrics['active_fire_assignments']}",
            f"Messages Sent: {metrics['coordination_messages']:,}",
            f"Network Density: {metrics['communication_network_density']:.3f}",
        ]
        
        # Display text
        y_pos = 0.95
        for line in dashboard_text:
            if line == "":
                y_pos -= 0.03
            else:
                weight = 'bold' if line.isupper() and not line.startswith('  ') else 'normal'
                size = 10 if weight == 'bold' else 9
                self.ax_perf.text(0.05, y_pos, line, fontsize=size, weight=weight, 
                                transform=self.ax_perf.transAxes)
                y_pos -= 0.05
        
        self.ax_perf.set_title('Performance Dashboard')
    
    def show(self):
        """Keep visualization window open."""
        plt.show(block=True)
    
    def save_frame(self, filename: str):
        """Save current visualization frame."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved visualization frame to {filename}")
    
    def close(self):
        """Close visualization window."""
        plt.close(self.fig)


def create_animation(fire_states: List[Any], drone_swarm_states: List[Any], 
                    save_path: str = None) -> animation.FuncAnimation:
    """
    Create an animation from recorded simulation states.
    
    Args:
        fire_states: List of fire states over time
        drone_swarm_states: List of drone swarm states over time
        save_path: Optional path to save animation as video
        
    Returns:
        Matplotlib animation object
    """
    if len(fire_states) != len(drone_swarm_states):
        raise ValueError("Fire states and drone states must have same length")
    
    # Initialize visualizer
    viz = RealtimeVisualizer()
    
    def animate(frame):
        viz.update(fire_states[frame], drone_swarm_states[frame], frame)
        return []
    
    # Create animation
    anim = animation.FuncAnimation(
        viz.fig, animate, frames=len(fire_states),
        interval=200, blit=False, repeat=True
    )
    
    # Save if requested
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        logging.info(f"Animation saved to {save_path}")
    
    return anim 