#!/usr/bin/env python3
"""
Automated training script for CI/CD pipelines.

This script is designed to be triggered by:
- GitHub Actions (self-hosted runner)
- Local git hooks
- Cron jobs
- Manual execution

Usage:
    python scripts/auto_train.py --timesteps 100000 --algorithm PPO
    python scripts/auto_train.py --config config/training.yaml
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def setup_logging(log_dir: str):
    """Setup logging for training run."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_git_info():
    """Get current git commit info."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return {'commit': commit, 'branch': branch}
    except:
        return {'commit': 'unknown', 'branch': 'unknown'}


def train(args):
    """Run training with specified parameters."""
    from agents.trainer import FireControlTrainer
    from config import Config, load_config

    logger = setup_logging(args.log_dir)
    git_info = get_git_info()

    logger.info("="*60)
    logger.info("AUTOMATED TRAINING RUN")
    logger.info("="*60)
    logger.info(f"Git commit: {git_info['commit']}")
    logger.info(f"Git branch: {git_info['branch']}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU available - training will be slow!")

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = load_config()
        logger.info("Using default config")

    # Override with command line args
    logger.info("")
    logger.info("Training Parameters:")
    logger.info(f"  Algorithm: {args.algorithm}")
    logger.info(f"  Timesteps: {args.timesteps:,}")
    logger.info(f"  Swarm size: {args.swarm_size}")
    logger.info(f"  Grid size: {args.grid_size}x{args.grid_size}")
    logger.info(f"  Parallel envs: {args.n_envs}")
    logger.info(f"  Seed: {args.seed}")
    logger.info("")

    # Create trainer
    trainer = FireControlTrainer(
        config=config,
        algorithm=args.algorithm,
        n_envs=args.n_envs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        verbose=1,
    )

    # Create environment
    trainer.create_env(
        swarm_size=args.swarm_size,
        grid_size=(args.grid_size, args.grid_size),
        max_episode_steps=args.max_steps,
    )

    # Create model
    trainer.create_model(learning_rate=args.learning_rate)

    # Train
    logger.info("Starting training...")
    start_time = datetime.now()

    try:
        trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("")
    logger.info(f"Training completed in {duration}")

    # Evaluate
    logger.info("")
    logger.info("Running evaluation...")
    results = trainer.evaluate(n_episodes=10)

    logger.info(f"Evaluation Results:")
    logger.info(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"  Mean length: {results['mean_length']:.1f}")

    # Save final model
    trainer.save_model('final')

    # Save run summary
    summary = {
        'git': git_info,
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': duration.total_seconds(),
        'parameters': {
            'algorithm': args.algorithm,
            'timesteps': args.timesteps,
            'swarm_size': args.swarm_size,
            'grid_size': args.grid_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
        },
        'results': {
            'mean_reward': float(results['mean_reward']),
            'std_reward': float(results['std_reward']),
            'mean_length': float(results['mean_length']),
        },
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    summary_path = os.path.join(args.checkpoint_dir, 'final', 'run_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Run summary saved to: {summary_path}")
    logger.info("")
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)

    trainer.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Automated training script for fire control RL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'DQN', 'A2C', 'SAC'],
                        help='RL algorithm')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')

    # Environment parameters
    parser.add_argument('--swarm-size', type=int, default=30,
                        help='Number of drones')
    parser.add_argument('--grid-size', type=int, default=40,
                        help='Fire grid size (square)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Max steps per episode')

    # Training settings
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Parallel environments')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=25000,
                        help='Checkpoint save frequency')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Paths
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
