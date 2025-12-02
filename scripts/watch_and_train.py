#!/usr/bin/env python3
"""
Local Git Watcher - Trains automatically when new commits are detected.

This is a simpler alternative to GitHub Actions self-hosted runners.
Just run this script on your GPU machine and it will:
1. Watch for new commits on the remote
2. Pull changes
3. Run training
4. Push results back

Usage:
    python scripts/watch_and_train.py
    python scripts/watch_and_train.py --branch main --interval 60

    # Run in background
    nohup python scripts/watch_and_train.py > training.log 2>&1 &
"""

import os
import sys
import time
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('watch_train.log')
        ]
    )
    return logging.getLogger(__name__)


def run_command(cmd, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def get_local_commit():
    """Get current local commit hash."""
    return run_command("git rev-parse HEAD")


def get_remote_commit(branch):
    """Get latest remote commit hash."""
    run_command(f"git fetch origin {branch}", check=False)
    return run_command(f"git rev-parse origin/{branch}")


def pull_changes(branch):
    """Pull latest changes from remote."""
    run_command(f"git pull origin {branch}")


def push_results():
    """Push training results back to remote."""
    try:
        run_command("git add results/ checkpoints/", check=False)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        run_command(f'git commit -m "Auto-training results {timestamp}"', check=False)
        run_command("git push", check=False)
    except Exception as e:
        logging.warning(f"Could not push results: {e}")


def run_training(args):
    """Run the training script."""
    cmd = f"""python scripts/auto_train.py \
        --timesteps {args.timesteps} \
        --algorithm {args.algorithm} \
        --swarm-size {args.swarm_size} \
        --grid-size {args.grid_size}"""

    subprocess.run(cmd, shell=True)


def watch_loop(args):
    """Main watch loop."""
    logger = setup_logging()

    logger.info("="*60)
    logger.info("GIT WATCHER - Auto-Training on Push")
    logger.info("="*60)
    logger.info(f"Watching branch: {args.branch}")
    logger.info(f"Check interval: {args.interval} seconds")
    logger.info(f"Training timesteps: {args.timesteps:,}")
    logger.info("="*60)
    logger.info("")
    logger.info("Waiting for new commits... (Ctrl+C to stop)")
    logger.info("")

    last_trained_commit = None

    while True:
        try:
            local_commit = get_local_commit()
            remote_commit = get_remote_commit(args.branch)

            if remote_commit != local_commit:
                logger.info(f"New commit detected: {remote_commit[:8]}")
                logger.info("Pulling changes...")

                pull_changes(args.branch)

                logger.info("")
                logger.info("Starting training...")
                logger.info("")

                run_training(args)

                if args.push_results:
                    logger.info("Pushing results...")
                    push_results()

                last_trained_commit = remote_commit
                logger.info("")
                logger.info(f"Training complete for commit {remote_commit[:8]}")
                logger.info("Waiting for next commit...")
                logger.info("")

            time.sleep(args.interval)

        except KeyboardInterrupt:
            logger.info("")
            logger.info("Watcher stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info(f"Retrying in {args.interval} seconds...")
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(
        description='Watch git repo and train on new commits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Watcher settings
    parser.add_argument('--branch', type=str, default='main',
                        help='Branch to watch')
    parser.add_argument('--interval', type=int, default=60,
                        help='Check interval in seconds')
    parser.add_argument('--push-results', action='store_true',
                        help='Push results back to remote')

    # Training parameters (passed to auto_train.py)
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Training timesteps per run')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'DQN', 'A2C'],
                        help='RL algorithm')
    parser.add_argument('--swarm-size', type=int, default=30,
                        help='Number of drones')
    parser.add_argument('--grid-size', type=int, default=40,
                        help='Fire grid size')

    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    watch_loop(args)


if __name__ == '__main__':
    main()
