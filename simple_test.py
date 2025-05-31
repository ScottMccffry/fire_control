#!/usr/bin/env python3
"""
Simple test to verify the project structure and basic imports.
This test works without requiring all heavy dependencies.
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """Test that all required directories and files exist."""
    print("Testing project structure...")
    
    required_dirs = [
        'sim', 'envs', 'agents', 'drones', 'data', 'scripts', 
        'visualization', 'tests'
    ]
    
    required_files = [
        'requirements.txt', 'readme.md', '__init__.py',
        'sim/__init__.py', 'drones/__init__.py', 'visualization/__init__.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required directories and files exist")
    return True


def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    print("Testing basic imports...")
    
    try:
        # Test individual modules
        import sim
        print("✅ sim module imports successfully")
        
        import drones
        print("✅ drones module imports successfully")
        
        import visualization
        print("✅ visualization module imports successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_file_contents():
    """Test that key files have expected content."""
    print("Testing file contents...")
    
    # Test README
    with open('readme.md', 'r') as f:
        readme_content = f.read()
        if 'Wildfire Spread and Drone Swarm Coordination Simulator' in readme_content:
            print("✅ README contains correct title")
        else:
            print("❌ README missing expected title")
            return False
    
    # Test requirements
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
        if 'numpy' in requirements and 'matplotlib' in requirements:
            print("✅ Requirements file contains expected dependencies")
        else:
            print("❌ Requirements file missing expected dependencies")
            return False
    
    return True


def show_system_info():
    """Show system information."""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current directory: {os.getcwd()}")
    
    print("\nProject structure:")
    for root, dirs, files in os.walk('.'):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                print(f"{subindent}{file}")


def show_usage_examples():
    """Show usage examples."""
    print("\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    print("After installing dependencies with 'pip install -r requirements.txt':")
    print("")
    print("1. Quick demo (100 drones, 2 hours):")
    print("   python scripts/run_grid_simulation.py --swarm-size 100 --hours 2")
    print("")
    print("2. Standard simulation (1000 drones, 6 hours):")
    print("   python scripts/run_grid_simulation.py --swarm-size 1000 --hours 6")
    print("")
    print("3. Large scale (5000 drones, 8 hours):")
    print("   python scripts/run_grid_simulation.py --swarm-size 5000 --hours 8")
    print("")
    print("4. Compare policies:")
    print("   python scripts/run_grid_simulation.py --policy random --no-viz")
    print("   python scripts/run_grid_simulation.py --policy greedy --no-viz")
    print("   python scripts/run_grid_simulation.py --policy coordinated --no-viz")
    print("")
    print("5. Run tests:")
    print("   python tests/test_basic_functionality.py")


def main():
    """Run all tests."""
    print("Wildfire Spread and Drone Swarm Coordination Simulator")
    print("Basic Structure Test")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    if not test_project_structure():
        all_passed = False
    
    if not test_file_contents():
        all_passed = False
    
    # Note about imports
    print("\nNote: Full import testing requires dependencies:")
    print("Run 'pip install -r requirements.txt' to install all dependencies")
    
    # Show system info
    show_system_info()
    
    # Show usage examples
    show_usage_examples()
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ PROJECT STRUCTURE TEST PASSED")
        print("The wildfire and drone swarm simulation system is properly set up!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run demo: python scripts/run_grid_simulation.py --swarm-size 100 --hours 2")
    else:
        print("❌ PROJECT STRUCTURE TEST FAILED")
        print("Some issues were found with the project structure.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main()) 