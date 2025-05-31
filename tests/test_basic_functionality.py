#!/usr/bin/env python3
"""
Basic functionality tests for the wildfire and drone swarm system.

This script tests core components to ensure they work correctly.
"""

import sys
import os
from pathlib import Path
import unittest
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim import MockFireSimulator, FireState, FlameFont
from drones import Drone, DroneSwarm, DroneCapabilities
from drones.behaviors import RandomPolicy, GreedyFirePolicy, CoordinatedPolicy


class TestFireSimulator(unittest.TestCase):
    """Test the fire simulation components."""
    
    def setUp(self):
        """Set up test fire simulator."""
        self.fire_sim = MockFireSimulator(
            grid_size=(50, 50),
            cell_size_meters=100.0,
            bounds=(34.0, -119.0, 35.0, -118.0)
        )
        
    def test_fire_simulator_initialization(self):
        """Test fire simulator initializes correctly."""
        self.assertEqual(self.fire_sim.grid_size, (50, 50))
        self.assertEqual(self.fire_sim.cell_size, 100.0)
        self.assertEqual(self.fire_sim.current_step, 0)
        
    def test_ignition_points(self):
        """Test setting ignition points."""
        ignition_points = [(10, 10), (20, 20)]
        self.fire_sim.set_ignition_points(ignition_points)
        
        # Check that ignition points are set
        self.assertEqual(self.fire_sim.burned_area[10, 10], 1)
        self.assertEqual(self.fire_sim.burned_area[20, 20], 1)
        self.assertGreater(self.fire_sim.heat_intensity[10, 10], 0)
        
    def test_fire_step(self):
        """Test fire simulation step."""
        # Set ignition point
        self.fire_sim.set_ignition_points([(25, 25)])
        
        # Run a few simulation steps
        initial_burned_area = self.fire_sim.get_total_burned_area()
        
        for _ in range(5):
            fire_state = self.fire_sim.step()
            self.assertIsInstance(fire_state, FireState)
            self.assertIsInstance(fire_state.current_flame_front, FlameFont)
        
        # Fire should have spread
        final_burned_area = self.fire_sim.get_total_burned_area()
        self.assertGreaterEqual(final_burned_area, initial_burned_area)
        
    def test_water_drop_effect(self):
        """Test water drop suppression effect."""
        # Set ignition and let fire grow
        self.fire_sim.set_ignition_points([(25, 25)])
        for _ in range(3):
            self.fire_sim.step()
        
        # Record heat before water drop
        heat_before = self.fire_sim.heat_intensity[25, 25]
        
        # Apply water drop
        self.fire_sim.apply_water_drop(25, 25, 10.0)  # 10 liters
        
        # Heat should be reduced
        heat_after = self.fire_sim.heat_intensity[25, 25]
        self.assertLess(heat_after, heat_before)


class TestDroneSystem(unittest.TestCase):
    """Test the drone and swarm components."""
    
    def setUp(self):
        """Set up test drone system."""
        self.drone = Drone(
            drone_id="test_drone_001",
            initial_position=(34.5, -118.5)
        )
        
        self.refill_locations = [(34.2, -118.8), (34.7, -118.3)]
        self.fire_map = np.zeros((50, 50))
        self.fire_map[20:25, 20:25] = 500.0  # Active fire region
        
    def test_drone_initialization(self):
        """Test drone initializes correctly."""
        self.assertEqual(self.drone.drone_id, "test_drone_001")
        self.assertEqual(self.drone.position, (34.5, -118.5))
        self.assertEqual(self.drone.battery_level, 1.0)
        self.assertEqual(self.drone.water_level, 1.0)
        self.assertTrue(self.drone.is_operational())
        
    def test_drone_update(self):
        """Test drone update cycle."""
        # Update drone for several time steps
        for _ in range(5):
            action = self.drone.update(
                dt_seconds=60.0,  # 1 minute
                fire_map=self.fire_map,
                refill_locations=self.refill_locations
            )
            self.assertIsInstance(action, dict)
            self.assertIn('type', action)
            
    def test_drone_water_drop(self):
        """Test drone water dropping behavior."""
        # Manually set drone to water dropping state
        self.drone.state = self.drone.state.DROPPING_WATER
        
        initial_water = self.drone.water_level
        
        # Update drone (should drop water)
        action = self.drone.update(
            dt_seconds=2.0,  # 2 seconds
            fire_map=self.fire_map,
            refill_locations=self.refill_locations
        )
        
        # Water level should decrease
        self.assertLess(self.drone.water_level, initial_water)
        self.assertEqual(action['type'], 'dropping_water')
        
    def test_drone_swarm_initialization(self):
        """Test drone swarm initialization."""
        swarm = DroneSwarm(
            swarm_size=100,
            deployment_area=(34.0, -119.0, 35.0, -118.0),
            refill_locations=self.refill_locations
        )
        
        self.assertEqual(swarm.swarm_size, 100)
        self.assertEqual(len(swarm.drones), 100)
        self.assertGreater(len(swarm.refill_locations), 0)
        
    def test_drone_swarm_update(self):
        """Test drone swarm coordinated update."""
        swarm = DroneSwarm(
            swarm_size=50,  # Smaller for testing
            deployment_area=(34.0, -119.0, 35.0, -118.0),
            refill_locations=self.refill_locations
        )
        
        # Create mock fire state
        fire_state = self._create_mock_fire_state()
        
        # Update swarm
        swarm_update = swarm.update_swarm(60.0, fire_state)
        
        self.assertIsInstance(swarm_update, dict)
        self.assertIn('metrics', swarm_update)
        self.assertIn('swarm_actions', swarm_update)
        
    def _create_mock_fire_state(self):
        """Create a mock fire state for testing."""
        flame_front = FlameFont(
            burned_area=np.zeros((50, 50)),
            heat_intensity=self.fire_map,
            fire_perimeter=[(34.4, -118.4), (34.6, -118.6)],
            timestamp=datetime.now(),
            time_step=1,
            grid_resolution=100.0,
            bounds=(34.0, -119.0, 35.0, -118.0)
        )
        
        fire_state = FireState(
            current_flame_front=flame_front,
            flame_front_history=[],
            weather_data={},
            terrain_data={},
            fuel_data={},
            simulation_id="test_sim",
            domain_name="test_domain",
            start_time=datetime.now(),
            current_time=datetime.now()
        )
        
        return fire_state


class TestPolicies(unittest.TestCase):
    """Test drone behavior policies."""
    
    def setUp(self):
        """Set up test components."""
        self.drone = Drone("policy_test_drone", (34.5, -118.5))
        self.fire_map = np.zeros((50, 50))
        self.fire_map[20:25, 20:25] = 500.0  # Active fire
        self.other_drones = []
        self.refill_locations = [(34.2, -118.8)]
        
    def test_random_policy(self):
        """Test random policy."""
        policy = RandomPolicy()
        
        action = policy.select_action(
            self.drone, self.fire_map, self.other_drones, self.refill_locations
        )
        
        self.assertIsInstance(action, dict)
        self.assertIn('type', action)
        
    def test_greedy_policy(self):
        """Test greedy fire policy."""
        policy = GreedyFirePolicy(priority_mode="nearest")
        
        action = policy.select_action(
            self.drone, self.fire_map, self.other_drones, self.refill_locations
        )
        
        self.assertIsInstance(action, dict)
        self.assertIn('type', action)
        
    def test_coordinated_policy(self):
        """Test coordinated policy."""
        policy = CoordinatedPolicy()
        
        action = policy.select_action(
            self.drone, self.fire_map, self.other_drones, self.refill_locations
        )
        
        self.assertIsInstance(action, dict)
        self.assertIn('type', action)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_full_simulation_step(self):
        """Test a complete simulation step with all components."""
        # Initialize components
        fire_sim = MockFireSimulator(grid_size=(30, 30))
        fire_sim.set_ignition_points([(15, 15)])
        
        swarm = DroneSwarm(
            swarm_size=20,
            deployment_area=(34.0, -119.0, 35.0, -118.0)
        )
        
        # Run simulation step
        fire_state = fire_sim.step()
        swarm_update = swarm.update_swarm(60.0, fire_state)
        
        # Verify results
        self.assertIsInstance(fire_state, FireState)
        self.assertIsInstance(swarm_update, dict)
        self.assertIn('metrics', swarm_update)
        
        # Check that some basic metrics are reasonable
        metrics = swarm_update['metrics']
        self.assertEqual(metrics['swarm_size'], 20)
        self.assertGreaterEqual(metrics['operational_drones'], 0)
        self.assertLessEqual(metrics['operational_drones'], 20)


def run_basic_tests():
    """Run all basic functionality tests."""
    print("Running basic functionality tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFireSimulator))
    test_suite.addTest(unittest.makeSuite(TestDroneSystem))
    test_suite.addTest(unittest.makeSuite(TestPolicies))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\nAll tests passed! ({result.testsRun} tests)")
        return True
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1) 