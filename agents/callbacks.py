"""
Custom callbacks for fire control training.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback as SB3EvalCallback


class FireControlCallback(BaseCallback):
    """
    Custom callback for fire control training.

    Logs domain-specific metrics:
    - Fire suppression rate
    - Drone operational status
    - Water efficiency
    - Episode outcomes
    """

    def __init__(
        self,
        verbose: int = 0,
        log_interval: int = 1000,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval

        # Metrics tracking
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.verbose > 0:
            print("Starting Fire Control training...")
            print(f"  Total timesteps: {self.locals.get('total_timesteps', 'N/A')}")

    def _on_step(self) -> bool:
        """Called after each step."""
        # Get info from environments
        infos = self.locals.get('infos', [])

        for info in infos:
            if info is None:
                continue

            # Track step-level metrics
            if 'active_fires' in info:
                self.step_metrics['active_fires'].append(info['active_fires'])
            if 'operational_drones' in info:
                self.step_metrics['operational_drones'].append(info['operational_drones'])
            if 'burned_cells' in info:
                self.step_metrics['burned_cells'].append(info['burned_cells'])

            # Check for episode end
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_count += 1
                self.episode_rewards.append(ep_info.get('r', 0))

                # Log episode metrics
                if 'episode_metrics' in info:
                    for key, value in info['episode_metrics'].items():
                        if isinstance(value, (int, float)):
                            self.episode_metrics[key].append(value)

        # Periodic logging
        if self.n_calls % self.log_interval == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Log aggregated metrics."""
        if self.verbose < 1:
            return

        # Step metrics
        if self.step_metrics['active_fires']:
            avg_fires = np.mean(self.step_metrics['active_fires'][-self.log_interval:])
            self.logger.record('fire_control/avg_active_fires', avg_fires)

        if self.step_metrics['operational_drones']:
            avg_drones = np.mean(self.step_metrics['operational_drones'][-self.log_interval:])
            self.logger.record('fire_control/avg_operational_drones', avg_drones)

        if self.step_metrics['burned_cells']:
            avg_burned = np.mean(self.step_metrics['burned_cells'][-self.log_interval:])
            self.logger.record('fire_control/avg_burned_cells', avg_burned)

        # Episode metrics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]
            self.logger.record('fire_control/mean_episode_reward', np.mean(recent_rewards))
            self.logger.record('fire_control/episode_count', self.episode_count)

        for key, values in self.episode_metrics.items():
            if values:
                self.logger.record(f'fire_control/{key}', np.mean(values[-10:]))

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print("\nFire Control Training Summary:")
            print(f"  Total episodes: {self.episode_count}")
            if self.episode_rewards:
                print(f"  Mean reward: {np.mean(self.episode_rewards):.2f}")
                print(f"  Final 10 episodes: {np.mean(self.episode_rewards[-10:]):.2f}")


class EvalCallback(SB3EvalCallback):
    """
    Extended evaluation callback with fire control metrics.
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 1,
        **kwargs
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose,
            **kwargs
        )

        self.eval_metrics = defaultdict(list)

    def _on_step(self) -> bool:
        """Extended step callback with additional logging."""
        result = super()._on_step()

        # Log additional metrics if evaluation happened
        if self.n_calls % self.eval_freq == 0:
            self._log_eval_metrics()

        return result

    def _log_eval_metrics(self):
        """Log evaluation-specific metrics."""
        if self.evaluations_results is not None and len(self.evaluations_results) > 0:
            # Get latest evaluation results
            latest_mean = self.evaluations_results[-1].mean()
            latest_std = self.evaluations_results[-1].std()

            self.logger.record('eval/mean_reward', latest_mean)
            self.logger.record('eval/std_reward', latest_std)

            # Track best performance
            best_mean = max(r.mean() for r in self.evaluations_results)
            self.logger.record('eval/best_mean_reward', best_mean)


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording evaluation videos.
    """

    def __init__(
        self,
        eval_env,
        video_folder: str = 'videos',
        record_freq: int = 50000,
        video_length: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.record_freq = record_freq
        self.video_length = video_length

        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        """Record video periodically."""
        if self.n_calls % self.record_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        """Record a single evaluation video."""
        try:
            import imageio

            frames = []
            obs = self.eval_env.reset()

            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.eval_env.step(action)

                # Render frame
                frame = self.eval_env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)

                if done[0]:
                    break

            # Save video
            if frames:
                video_path = os.path.join(
                    self.video_folder,
                    f'eval_step_{self.n_calls}.mp4'
                )
                imageio.mimsave(video_path, frames, fps=10)

                if self.verbose > 0:
                    print(f"Saved video to {video_path}")

        except ImportError:
            if self.verbose > 0:
                print("imageio not installed, skipping video recording")
        except Exception as e:
            if self.verbose > 0:
                print(f"Error recording video: {e}")


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.

    Gradually increases task difficulty during training.
    """

    def __init__(
        self,
        schedule: Dict[int, Dict[str, Any]],
        verbose: int = 0,
    ):
        """
        Args:
            schedule: Dictionary mapping timestep -> environment parameters
                Example: {
                    0: {'num_ignitions': 1, 'swarm_size': 50},
                    100000: {'num_ignitions': 2, 'swarm_size': 40},
                    200000: {'num_ignitions': 3, 'swarm_size': 30},
                }
        """
        super().__init__(verbose)
        self.schedule = schedule
        self.current_level = 0
        self.schedule_steps = sorted(schedule.keys())

    def _on_step(self) -> bool:
        """Check if curriculum should advance."""
        # Find current curriculum level
        new_level = 0
        for i, step in enumerate(self.schedule_steps):
            if self.num_timesteps >= step:
                new_level = i + 1

        if new_level > self.current_level:
            self.current_level = new_level
            step = self.schedule_steps[new_level - 1]
            params = self.schedule[step]

            if self.verbose > 0:
                print(f"\nCurriculum: Advancing to level {new_level}")
                print(f"  New parameters: {params}")

            # Note: Actual environment update would need to be implemented
            # based on how environments are managed

        return True
