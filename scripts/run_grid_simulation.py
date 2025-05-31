#!/usr/bin/env python3
"""
Run a simplified grid-based wildfire and drone swarm simulation.

This script demonstrates the integrated system using the MockFireSimulator
for fast prototyping without requiring full WRF-SFIRE setup.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim import MockFireSimulator, FireState
from drones import DroneSwarm, RandomPolicy, GreedyFirePolicy, CoordinatedPolicy
from visualization.realtime_viz import RealtimeVisualizer


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )


def create_sample_scenario(scenario_name: str = "california_wildfire"):
    """Create a sample simulation scenario."""
    if scenario_name == "california_wildfire":
        # California scenario
        grid_size = (100, 100)
        cell_size = 100.0  # 100m per cell = 10km x 10km area
        bounds = (34.0, -119.0, 35.0, -118.0)  # Rough California coordinates
        
        # Initial ignition points (multiple spot fires)
        ignition_points = [
            (30, 40),  # Main fire
            (60, 70),  # Secondary fire
            (20, 80)   # Small fire
        ]
        
        # Refill locations (simulated lakes)
        refill_locations = [
            (34.2, -118.8),
            (34.7, -118.3),
            (34.5, -118.6)
        ]
        
        return {
            'grid_size': grid_size,
            'cell_size': cell_size,
            'bounds': bounds,
            'ignition_points': ignition_points,
            'refill_locations': refill_locations,
            'wind_speed': 8.0,  # m/s
            'wind_direction': 45.0,  # degrees
            'temperature': 30.0,  # Celsius
            'humidity': 0.2  # 20%
        }
    
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")


def run_simulation(
    scenario_config: dict,
    swarm_size: int = 1000,
    simulation_hours: int = 6,
    time_step_minutes: int = 5,
    policy_type: str = "greedy",
    visualize: bool = True,
    output_dir: str = "./results"
):
    """
    Run the complete wildfire and drone swarm simulation.
    
    Args:
        scenario_config: Scenario configuration dictionary
        swarm_size: Number of drones in the swarm
        simulation_hours: Total simulation time in hours
        time_step_minutes: Simulation time step in minutes
        policy_type: Drone policy ("random", "greedy", "coordinated")
        visualize: Whether to show real-time visualization
        output_dir: Directory to save results
    """
    logger = logging.getLogger("SimulationRunner")
    logger.info(f"Starting simulation with {swarm_size} drones for {simulation_hours} hours")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize fire simulator
    fire_sim = MockFireSimulator(
        grid_size=scenario_config['grid_size'],
        cell_size_meters=scenario_config['cell_size'],
        bounds=scenario_config['bounds']
    )
    
    # Set weather conditions
    fire_sim.set_weather(
        wind_speed=scenario_config['wind_speed'],
        wind_direction=scenario_config['wind_direction'],
        temperature=scenario_config['temperature'],
        humidity=scenario_config['humidity']
    )
    
    # Set ignition points
    fire_sim.set_ignition_points(scenario_config['ignition_points'])
    
    # Initialize drone swarm
    drone_swarm = DroneSwarm(
        swarm_size=swarm_size,
        deployment_area=scenario_config['bounds'],
        refill_locations=scenario_config['refill_locations']
    )
    
    # Set drone policy
    if policy_type == "random":
        drone_swarm.assignment_strategy = "random"
    elif policy_type == "greedy":
        drone_swarm.assignment_strategy = "greedy"
    elif policy_type == "coordinated":
        drone_swarm.assignment_strategy = "clusters"
    else:
        logger.warning(f"Unknown policy type: {policy_type}, using greedy")
        drone_swarm.assignment_strategy = "greedy"
    
    # Initialize visualizer if requested
    visualizer = None
    if visualize:
        try:
            visualizer = RealtimeVisualizer(
                grid_size=scenario_config['grid_size'],
                bounds=scenario_config['bounds']
            )
            logger.info("Real-time visualization enabled")
        except Exception as e:
            logger.warning(f"Could not initialize visualizer: {e}")
            visualizer = None
    
    # Simulation parameters
    dt_seconds = time_step_minutes * 60
    total_steps = int((simulation_hours * 60) / time_step_minutes)
    
    # Results tracking
    results = {
        'timestamps': [],
        'fire_metrics': [],
        'swarm_metrics': [],
        'step_times': []
    }
    
    logger.info(f"Running {total_steps} simulation steps")
    
    # Main simulation loop
    for step in range(total_steps):
        step_start_time = time.time()
        
        # Current simulation time
        sim_time = fire_sim.start_time + timedelta(minutes=step * time_step_minutes)
        
        # Step fire simulation
        fire_state = fire_sim.step()
        
        # Apply drone water drops to fire simulation
        drone_water_drops = []
        for drone_id, drone in drone_swarm.drones.items():
            if drone.state.value == 'dropping_water':
                # Get drone position in grid coordinates
                lat, lon = drone.position
                fire_row, fire_col = fire_sim._latlon_to_grid(lat, lon)
                
                # Apply water drop to fire simulation
                water_amount = min(drone.water_level * drone.capabilities.water_capacity_liters, 2.0)
                if water_amount > 0:
                    fire_sim.apply_water_drop(fire_row, fire_col, water_amount)
                    drone_water_drops.append({
                        'drone_id': drone_id,
                        'position': (fire_row, fire_col),
                        'water_amount': water_amount
                    })
        
        # Update drone swarm
        swarm_update = drone_swarm.update_swarm(dt_seconds, fire_state)
        
        # Collect metrics
        fire_metrics = {
            'total_burned_area_m2': fire_sim.get_total_burned_area(),
            'active_fire_cells': len(fire_sim.get_active_fire_cells()),
            'max_heat_intensity': float(fire_state.current_flame_front.heat_intensity.max()),
            'fire_perimeter_length': len(fire_state.current_flame_front.fire_perimeter)
        }
        
        swarm_metrics = swarm_update['metrics']
        swarm_metrics['water_drops_this_step'] = len(drone_water_drops)
        
        # Record results
        results['timestamps'].append(sim_time)
        results['fire_metrics'].append(fire_metrics)
        results['swarm_metrics'].append(swarm_metrics)
        results['step_times'].append(time.time() - step_start_time)
        
        # Update visualization
        if visualizer:
            try:
                visualizer.update(fire_state, drone_swarm, step)
            except Exception as e:
                logger.warning(f"Visualization update failed: {e}")
        
        # Progress logging
        if step % 10 == 0 or step == total_steps - 1:
            logger.info(
                f"Step {step+1}/{total_steps} - "
                f"Burned: {fire_metrics['total_burned_area_m2']/1e6:.2f} km² - "
                f"Active fires: {fire_metrics['active_fire_cells']} - "
                f"Operational drones: {swarm_metrics['operational_drones']}/{swarm_size} - "
                f"Water dropped: {swarm_metrics['total_water_dropped_liters']:.1f}L"
            )
        
        # Emergency stop if all drones are down
        if swarm_metrics['operational_drones'] == 0:
            logger.warning("All drones are non-operational, stopping simulation")
            break
    
    # Simulation complete
    total_sim_time = time.time() - (time.time() - sum(results['step_times']))
    logger.info(f"Simulation completed in {total_sim_time:.2f} seconds")
    
    # Final summary
    final_fire = results['fire_metrics'][-1]
    final_swarm = results['swarm_metrics'][-1]
    
    logger.info("=== SIMULATION SUMMARY ===")
    logger.info(f"Total burned area: {final_fire['total_burned_area_m2']/1e6:.2f} km²")
    logger.info(f"Total water dropped: {final_swarm['total_water_dropped_liters']:.1f} L")
    logger.info(f"Total fires suppressed: {final_swarm['total_fires_suppressed']}")
    logger.info(f"Average operational drones: {sum(m['operational_drones'] for m in results['swarm_metrics'])/len(results['swarm_metrics']):.1f}")
    logger.info(f"Total missions completed: {final_swarm['total_missions_completed']}")
    logger.info(f"Total flight time: {final_swarm['total_flight_time_hours']:.1f} hours")
    
    # Save results
    import json
    results_file = output_path / f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj
    
    # Clean results for JSON
    json_results = {
        'config': scenario_config,
        'simulation_params': {
            'swarm_size': swarm_size,
            'simulation_hours': simulation_hours,
            'time_step_minutes': time_step_minutes,
            'policy_type': policy_type
        },
        'timestamps': [t.isoformat() for t in results['timestamps']],
        'fire_metrics': [{k: convert_numpy(v) for k, v in m.items()} for m in results['fire_metrics']],
        'swarm_metrics': [{k: convert_numpy(v) for k, v in m.items()} for m in results['swarm_metrics']],
        'summary': {
            'total_burned_area_km2': final_fire['total_burned_area_m2']/1e6,
            'total_water_dropped_liters': final_swarm['total_water_dropped_liters'],
            'total_fires_suppressed': final_swarm['total_fires_suppressed'],
            'avg_operational_drones': sum(m['operational_drones'] for m in results['swarm_metrics'])/len(results['swarm_metrics']),
            'total_missions_completed': final_swarm['total_missions_completed']
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Keep visualization open if used
    if visualizer:
        logger.info("Keeping visualization open. Close window to exit.")
        visualizer.show()
    
    return results


def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description="Run wildfire and drone swarm simulation")
    
    parser.add_argument("--scenario", default="california_wildfire",
                       help="Simulation scenario name")
    parser.add_argument("--swarm-size", type=int, default=1000,
                       help="Number of drones in the swarm")
    parser.add_argument("--hours", type=int, default=6,
                       help="Simulation duration in hours")
    parser.add_argument("--time-step", type=int, default=5,
                       help="Time step in minutes")
    parser.add_argument("--policy", choices=["random", "greedy", "coordinated"], 
                       default="greedy", help="Drone behavior policy")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable real-time visualization")
    parser.add_argument("--output", default="./results",
                       help="Output directory for results")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create scenario configuration
    scenario_config = create_sample_scenario(args.scenario)
    
    # Run simulation
    try:
        results = run_simulation(
            scenario_config=scenario_config,
            swarm_size=args.swarm_size,
            simulation_hours=args.hours,
            time_step_minutes=args.time_step,
            policy_type=args.policy,
            visualize=not args.no_viz,
            output_dir=args.output
        )
        
        print("\nSimulation completed successfully!")
        print(f"Check {args.output} for detailed results.")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        logging.exception("Simulation error:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 