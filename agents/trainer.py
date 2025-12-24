"""
Training infrastructure for fire control RL agents.

Supports PPO, DQN, SAC, and A2C algorithms via Stable Baselines3.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Type, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO, DQN, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fire_control_env import FireControlEnv
from config import Config, load_config
from .callbacks import FireControlCallback


class FireControlTrainer:
    """
    Trainer class for fire control RL agents.

    Handles:
    - Environment creation and vectorization
    - Algorithm selection and configuration
    - Training with callbacks
    - Model saving and loading
    - Evaluation and metrics logging
    """

    ALGORITHMS = {
        'PPO': PPO,
        'DQN': DQN,
        'SAC': SAC,
        'A2C': A2C,
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        algorithm: str = 'PPO',
        n_envs: int = 4,
        device: str = 'auto',
        seed: int = 42,
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints',
        verbose: int = 1,
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration object
            algorithm: RL algorithm name (PPO, DQN, SAC, A2C)
            n_envs: Number of parallel environments
            device: Device for training ('auto', 'cpu', 'cuda')
            seed: Random seed
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for model checkpoints
            verbose: Verbosity level
        """
        self.config = config or load_config()
        self.algorithm_name = algorithm.upper()
        self.n_envs = n_envs
        self.device = device
        self.seed = seed
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set random seeds
        set_random_seed(seed)

        # Training state
        self.model = None
        self.env = None
        self.eval_env = None
        self.training_start_time = None
        self.total_timesteps_trained = 0

        # Metrics storage
        self.training_metrics = []

    def _make_env(
        self,
        swarm_size: int = 50,
        grid_size: tuple = (50, 50),
        max_episode_steps: int = 200,
        rank: int = 0,
        seed: int = 0,
    ):
        """Create a single environment instance."""
        def _init():
            env = FireControlEnv(
                config=self.config,
                swarm_size=swarm_size,
                grid_size=grid_size,
                max_episode_steps=max_episode_steps,
            )
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    def create_env(
        self,
        swarm_size: int = 50,
        grid_size: tuple = (50, 50),
        max_episode_steps: int = 200,
        use_subprocess: bool = False,
        normalize: bool = True,
    ):
        """
        Create vectorized training environment.

        Args:
            swarm_size: Number of drones
            grid_size: Fire grid dimensions
            max_episode_steps: Maximum steps per episode
            use_subprocess: Use SubprocVecEnv for true parallelism
            normalize: Apply VecNormalize wrapper
        """
        env_fns = [
            self._make_env(swarm_size, grid_size, max_episode_steps, i, self.seed)
            for i in range(self.n_envs)
        ]

        if use_subprocess and self.n_envs > 1:
            self.env = SubprocVecEnv(env_fns)
        else:
            self.env = DummyVecEnv(env_fns)

        if normalize:
            self.env = VecNormalize(
                self.env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )

        # Create eval environment (single, not normalized for fair comparison)
        eval_env_fn = self._make_env(swarm_size, grid_size, max_episode_steps, 0, self.seed + 1000)
        self.eval_env = DummyVecEnv([eval_env_fn])

        self.logger.info(f"Created {self.n_envs} training environments")

    def create_model(
        self,
        policy: str = 'MlpPolicy',
        learning_rate: float = None,
        **kwargs
    ):
        """
        Create RL model.

        Args:
            policy: Policy network type
            learning_rate: Learning rate (uses config default if None)
            **kwargs: Additional algorithm-specific arguments
        """
        if self.env is None:
            raise ValueError("Environment must be created before model")

        algorithm_class = self.ALGORITHMS.get(self.algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

        # Get hyperparameters from config
        if self.algorithm_name == 'PPO':
            default_params = {
                'learning_rate': self.config.training.learning_rate,
                'n_steps': self.config.training.n_steps,
                'batch_size': self.config.training.batch_size,
                'n_epochs': self.config.training.n_epochs,
                'gamma': self.config.training.gamma,
                'gae_lambda': self.config.training.gae_lambda,
                'clip_range': self.config.training.clip_range,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
            }
        elif self.algorithm_name == 'DQN':
            default_params = {
                'learning_rate': 0.0001,
                'buffer_size': 100000,
                'learning_starts': 1000,
                'batch_size': 32,
                'tau': 0.005,
                'gamma': self.config.training.gamma,
                'target_update_interval': 500,
                'exploration_fraction': 0.1,
                'exploration_final_eps': 0.05,
            }
        elif self.algorithm_name == 'SAC':
            default_params = {
                'learning_rate': 0.0003,
                'buffer_size': 100000,
                'learning_starts': 1000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': self.config.training.gamma,
            }
        else:  # A2C
            default_params = {
                'learning_rate': 0.0007,
                'n_steps': 5,
                'gamma': self.config.training.gamma,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.25,
                'max_grad_norm': 0.5,
            }

        # Override with provided arguments
        if learning_rate is not None:
            default_params['learning_rate'] = learning_rate
        default_params.update(kwargs)

        # Create model
        self.model = algorithm_class(
            policy,
            self.env,
            verbose=self.verbose,
            device=self.device,
            tensorboard_log=self.log_dir,
            seed=self.seed,
            **default_params
        )

        self.logger.info(f"Created {self.algorithm_name} model with policy {policy}")

    def train(
        self,
        total_timesteps: int = None,
        eval_freq: int = None,
        n_eval_episodes: int = 10,
        save_freq: int = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
    ):
        """
        Train the model.

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_freq: Checkpoint save frequency
            reset_num_timesteps: Reset timestep counter
            progress_bar: Show progress bar
        """
        if self.model is None:
            raise ValueError("Model must be created before training")

        # Use config defaults if not specified
        total_timesteps = total_timesteps or self.config.training.total_timesteps
        eval_freq = eval_freq or self.config.training.eval_freq
        save_freq = save_freq or self.config.training.save_freq

        # Create callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // self.n_envs,
            save_path=self.checkpoint_dir,
            name_prefix=f'{self.algorithm_name.lower()}_fire_control',
            save_replay_buffer=self.algorithm_name == 'DQN',
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=os.path.join(self.checkpoint_dir, 'best'),
                log_path=os.path.join(self.log_dir, 'eval'),
                eval_freq=eval_freq // self.n_envs,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        # Custom fire control callback
        fc_callback = FireControlCallback(
            verbose=self.verbose,
            log_interval=1000,
        )
        callbacks.append(fc_callback)

        callback_list = CallbackList(callbacks)

        # Train
        self.training_start_time = datetime.now()
        self.logger.info(f"Starting training for {total_timesteps} timesteps")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        self.total_timesteps_trained += total_timesteps

        # Save final model
        self.save_model('final')

        self.logger.info(f"Training completed. Total timesteps: {self.total_timesteps_trained}")

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            render: Render episodes

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be created/loaded before evaluation")

        # Use eval environment
        env = self.eval_env if self.eval_env is not None else self.env

        episode_rewards = []
        episode_lengths = []
        episode_metrics = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            ep_info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)

                episode_reward += reward[0]
                episode_length += 1

                if render:
                    env.render()

                if done[0]:
                    ep_info = info[0]

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_metrics.append(ep_info)

        # Compute statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'n_episodes': n_episodes,
            'episode_rewards': episode_rewards,
        }

        # Aggregate episode metrics
        if episode_metrics:
            for key in episode_metrics[0].keys():
                if isinstance(episode_metrics[0][key], (int, float)):
                    values = [m.get(key, 0) for m in episode_metrics]
                    results[f'mean_{key}'] = np.mean(values)

        self.logger.info(
            f"Evaluation: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f} "
            f"over {n_episodes} episodes"
        )

        return results

    def save_model(self, name: str = 'model'):
        """Save model and training state."""
        if self.model is None:
            raise ValueError("No model to save")

        # Create save directory
        save_dir = os.path.join(self.checkpoint_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(save_dir, 'model.zip')
        self.model.save(model_path)

        # Save VecNormalize stats if applicable
        if isinstance(self.env, VecNormalize):
            norm_path = os.path.join(save_dir, 'vec_normalize.pkl')
            self.env.save(norm_path)

        # Save training metadata
        metadata = {
            'algorithm': self.algorithm_name,
            'total_timesteps': self.total_timesteps_trained,
            'seed': self.seed,
            'training_start': str(self.training_start_time) if self.training_start_time else None,
            'config': self.config.to_dict(),
        }
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Model saved to {save_dir}")

    def load_model(self, path: str, load_env_stats: bool = True):
        """
        Load a saved model.

        Args:
            path: Path to saved model directory
            load_env_stats: Load VecNormalize statistics
        """
        # Load model
        model_path = os.path.join(path, 'model.zip')
        algorithm_class = self.ALGORITHMS.get(self.algorithm_name)

        self.model = algorithm_class.load(
            model_path,
            env=self.env,
            device=self.device,
        )

        # Load VecNormalize stats
        if load_env_stats and isinstance(self.env, VecNormalize):
            norm_path = os.path.join(path, 'vec_normalize.pkl')
            if os.path.exists(norm_path):
                self.env = VecNormalize.load(norm_path, self.env)
                self.env.training = False
                self.env.norm_reward = False

        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.total_timesteps_trained = metadata.get('total_timesteps', 0)

        self.logger.info(f"Model loaded from {path}")

    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()


def quick_train(
    algorithm: str = 'PPO',
    total_timesteps: int = 100000,
    swarm_size: int = 30,
    n_envs: int = 4,
    seed: int = 42,
) -> FireControlTrainer:
    """
    Quick training function for testing.

    Args:
        algorithm: RL algorithm
        total_timesteps: Training timesteps
        swarm_size: Number of drones
        n_envs: Parallel environments
        seed: Random seed

    Returns:
        Trained FireControlTrainer instance
    """
    trainer = FireControlTrainer(
        algorithm=algorithm,
        n_envs=n_envs,
        seed=seed,
    )

    trainer.create_env(swarm_size=swarm_size)
    trainer.create_model()
    trainer.train(total_timesteps=total_timesteps)

    return trainer


if __name__ == '__main__':
    # Test training
    logging.basicConfig(level=logging.INFO)

    trainer = quick_train(
        algorithm='PPO',
        total_timesteps=10000,
        swarm_size=20,
        n_envs=2,
    )

    # Evaluate
    results = trainer.evaluate(n_episodes=5)
    print(f"Evaluation results: {results}")

    trainer.close()
