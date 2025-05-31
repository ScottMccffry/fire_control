# Wildfire Spread and Drone Swarm Coordination Simulator
# Makefile for easy project management

.PHONY: help test run-simulation demo install clean lint docs

# Default target
help:
	@echo "Wildfire & Drone Swarm Simulator"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@echo "  help            - Show this help message"
	@echo "  install         - Install dependencies"
	@echo "  test            - Run unit tests"
	@echo "  demo            - Run quick demo simulation"
	@echo "  run-simulation  - Run full simulation with visualization"
	@echo "  run-large       - Run large-scale simulation (5000 drones)"
	@echo "  clean           - Clean up generated files"
	@echo "  lint            - Run code linting"
	@echo ""
	@echo "Examples:"
	@echo "  make demo                          # Quick 100-drone demo"
	@echo "  make run-simulation SWARM=1000     # 1000 drones"
	@echo "  make run-large                     # 5000 drones"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run unit tests
test:
	@echo "Running unit tests..."
	python tests/test_basic_functionality.py

# Quick demo (small scale for fast testing)
demo:
	@echo "Running quick demo simulation..."
	python scripts/run_grid_simulation.py \
		--swarm-size 100 \
		--hours 2 \
		--time-step 5 \
		--policy greedy \
		--output ./demo_results

# Standard simulation run
run-simulation:
	@echo "Running wildfire and drone swarm simulation..."
	python scripts/run_grid_simulation.py \
		--swarm-size $(if $(SWARM),$(SWARM),1000) \
		--hours $(if $(HOURS),$(HOURS),6) \
		--time-step $(if $(TIMESTEP),$(TIMESTEP),5) \
		--policy $(if $(POLICY),$(POLICY),greedy) \
		--output ./results

# Large scale simulation (5000 drones)
run-large:
	@echo "Running large-scale simulation with 5000 drones..."
	python scripts/run_grid_simulation.py \
		--swarm-size 5000 \
		--hours 8 \
		--time-step 10 \
		--policy coordinated \
		--output ./large_scale_results

# Run without visualization (faster)
run-headless:
	@echo "Running simulation without visualization..."
	python scripts/run_grid_simulation.py \
		--swarm-size $(if $(SWARM),$(SWARM),1000) \
		--hours $(if $(HOURS),$(HOURS),6) \
		--no-viz \
		--output ./headless_results

# Compare different policies
compare-policies:
	@echo "Comparing different drone policies..."
	@mkdir -p policy_comparison
	@echo "Running random policy..."
	python scripts/run_grid_simulation.py --swarm-size 500 --hours 4 --policy random --no-viz --output policy_comparison/random
	@echo "Running greedy policy..."
	python scripts/run_grid_simulation.py --swarm-size 500 --hours 4 --policy greedy --no-viz --output policy_comparison/greedy
	@echo "Running coordinated policy..."
	python scripts/run_grid_simulation.py --swarm-size 500 --hours 4 --policy coordinated --no-viz --output policy_comparison/coordinated
	@echo "Policy comparison complete! Check policy_comparison/ directory"

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf *.pyc
	rm -rf */*.pyc
	rm -rf */*/*.pyc
	rm -rf .pytest_cache/
	rm -rf *.log
	rm -rf results/
	rm -rf demo_results/
	rm -rf large_scale_results/
	rm -rf headless_results/
	rm -rf policy_comparison/
	@echo "Cleanup complete!"

# Code linting (requires flake8)
lint:
	@echo "Running code linting..."
	-python -m flake8 --max-line-length=100 --ignore=E501,W503 sim/ drones/ scripts/ tests/ visualization/
	@echo "Linting complete!"

# Generate documentation (requires pdoc)
docs:
	@echo "Generating documentation..."
	-python -m pdoc --html --output-dir docs sim drones visualization
	@echo "Documentation generated in docs/ directory"

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	-pip install flake8 pdoc3 pytest
	@echo "Development environment ready!"

# Quick system check
check:
	@echo "Performing system check..."
	@python -c "import numpy, matplotlib, scipy; print('✓ Core dependencies available')"
	@python -c "import sim, drones, visualization; print('✓ All modules importable')"
	@echo "System check complete!"

# Show system info
info:
	@echo "System Information:"
	@echo "==================="
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	@python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
	@echo ""
	@echo "Project Structure:"
	@find . -name "*.py" -type f | head -20 | sed 's/^/  /'
	@echo "  ..." 