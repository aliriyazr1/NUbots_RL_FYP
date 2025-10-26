"""
Multi-Model Training System - Train multiple variants of each algorithm and pick the best
Author: Fixed by Claude for Ali Riyaz
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.environments.soccerenv import SoccerEnv, ActionSmoothingWrapper, HandCodedPolicy
from src.training.train_GUI import TrainGUI
import os, time, yaml, torch, datetime, json, copy
from collections import deque
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Tuple, Optional, Any, Union

#TODO: Need to see if I'm collecting metrics like ball possession time, goals scored
#TODO: Number of times ball went out of bounds, collision frequency, etc

#TODO: Need to actually use the create_enhanced_metrics whatever function
#TODO: Need to actually train the models as well



class MultiModelTrainingSystem:
    """System for training multiple variants of each algorithm and selecting the best performers"""
    
    def __init__(self, config_path="configs/field_config.yaml", base_output_dir="experiments/runs"):
        self.config_path = config_path
        self.training_start_time = None
        self.base_output_dir = base_output_dir

        # Track best models for each algorithm type
        self.best_ppo_models = []  # List of (model_path, score, config) tuples
        self.best_ddpg_models = []  # List of (model_path, score, config) tuples
        self.best_ppo_score = -np.inf
        self.best_ddpg_score = -np.inf
        self.best_ppo_path = None
        self.best_ddpg_path = None

        # Create output directory structure
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.create_experiment_directory(self.timestamp, self.base_output_dir)

        # Setup logging
        self._setup_logging()
        
    def create_experiment_directory(self, timestamp=None, base_dir="experiments/runs") -> str:
        """Create organised experiment directory structure with tensorboard support"""
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create main experiment directory
        experiment_name = f"multi_model_training_{timestamp}"
        experiment_dir = os.path.join(base_dir, experiment_name)

        # Define all subdirectories needed
        # Note: logs, tensorboard_logs, and evaluations not created here
        # - TensorBoard logs go to algorithm-specific directories (ppo_logs/PPO_1, etc.)
        # - evaluations.npz goes to algorithm-specific directories (ppo_logs/, ddpg_logs/)
        # - Training log goes to experiment root
        directories = [
            experiment_dir,
            os.path.join(experiment_dir, "models", "ppo"),
            os.path.join(experiment_dir, "models", "ddpg"),
            os.path.join(experiment_dir, "plots"),
            os.path.join(experiment_dir, "hyperparameters"),
            os.path.join(experiment_dir, "checkpoints")
        ]

        # Create all directories
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)

        # Note: Logging not available yet as this runs before _setup_logging()
        # Directory creation message will be logged by caller if needed
        return experiment_dir

    def _create_directories(self):
        """Legacy method for backward compatibility"""
        # This method is now handled by create_experiment_directory
        pass
            
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_file = f"{self.output_dir}/training_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created experiment directory: {self.output_dir}")
        self.logger.info(f"Logging to: {log_file}")


class ModelTracker(BaseCallback):
    """Track model performance and save best performers"""
    
    def __init__(self, algorithm_name, training_system, model_variant, verbose=0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.model_variant = model_variant
        self.training_system = training_system
        self.episode_rewards = deque(maxlen=100)
        self.best_avg_reward = -np.inf
        self.evaluation_counter = 0
        self.models_saved = 0
        self.current_model = None
    
    def set_model(self, model):
        """Set model reference for evaluation"""
        self.current_model = model
        
    def _on_step(self) -> bool:
        """Called at every step during training"""
        # Track episode completion
        infos = self.locals.get('infos', [])
        if isinstance(infos, dict):
            infos = [infos]
            
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
        # Evaluate and save model every 100k steps
        if self.num_timesteps % 100000 == 0 and len(self.episode_rewards) >= 20:
            self._evaluate_and_save_model();
        
        # To pass current timestep to env if needed
        if hasattr(self, 'train_env'):
            # Calculate relative timesteps (how much we've trained in THIS session)
            initial = getattr(self.train_env, 'initial_timesteps', 0)
            relative_timesteps = self.num_timesteps - initial
            self.train_env.env.total_timesteps_trained = self.num_timesteps
            self.train_env.env.relative_timesteps = relative_timesteps

        return True
    
    def _evaluate_and_save_model(self):
        """Evaluate current model and save if it's the best so far"""
        if self.current_model is None:
            self.training_system.logger.error("Model reference not set!")
            return
        
        try:
            current_avg_reward = np.mean(list(self.episode_rewards))
            
            # Comprehensive evaluation across difficulties
            eval_score = self._comprehensive_evaluation()
            
            self.training_system.logger.info(
                f"{self.algorithm_name}-{self.model_variant} Step {self.num_timesteps}: "
                f"Recent Avg = {current_avg_reward:.2f}, Eval Score = {eval_score:.2f}"
            )
            
            # Save model if it's the best so far
            if eval_score > self.best_avg_reward:
                self.best_avg_reward = eval_score
                save_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"{self.training_system.output_dir}/models/{self.algorithm_name.lower()}/{self.algorithm_name.lower()}_{self.model_variant}_{save_timestamp}_step{self.num_timesteps}"
                
                # Create a copy of the model to save
                # temp_model = copy.deepcopy(self.current_model)
                # temp_model.save(model_path)

                self.current_model.save(model_path) # No idea why this replaces the above 2 lines
                
                # Update tracking
                if self.algorithm_name == "PPO":
                    # Update global best PPO if this is better
                    if eval_score > self.training_system.best_ppo_score:
                        self.training_system.best_ppo_score = eval_score
                        self.training_system.best_ppo_path = model_path
                    
                    # Add to PPO models list
                    self.training_system.best_ppo_models.append((model_path, eval_score, self.model_variant))
                    
                elif self.algorithm_name == "DDPG":
                    # Update global best DDPG if this is better
                    if eval_score > self.training_system.best_ddpg_score:
                        self.training_system.best_ddpg_score = eval_score
                        self.training_system.best_ddpg_path = model_path
                    
                    # Add to DDPG models list
                    self.training_system.best_ddpg_models.append((model_path, eval_score, self.model_variant))
                
                self.training_system.logger.info(
                    f"NEW BEST {self.algorithm_name}-{self.model_variant} MODEL! "
                    f"Score: {eval_score:.2f} (saved to {model_path})"
                )
                
                # Save detailed evaluation results
                self._save_evaluation_details(eval_score, model_path)
                
            self.models_saved += 1
            
        except Exception as e:
            self.training_system.logger.error(f"Error in model evaluation: {e}")
    
    def _comprehensive_evaluation(self):
        """Evaluate model across all difficulties"""
        total_score = 0
        weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}  # Weight harder difficulties more
        
        for difficulty, weight in weights.items():
            try:
                eval_env = Monitor(SoccerEnv(difficulty=difficulty, config_path=self.training_system.config_path))
                mean_reward, _ = evaluate_policy(self.current_model, eval_env, n_eval_episodes=5, deterministic=True)
                eval_env.close()
                
                weighted_score = mean_reward * weight
                total_score += weighted_score
                
            except Exception as e:
                self.training_system.logger.error(f"Error evaluating {difficulty}: {e}")
                total_score += -50.0 * weight  # Penalty for failed evaluation
                
        return total_score
    
    def _save_evaluation_details(self, eval_score, model_path):
        """Save detailed evaluation results"""
        eval_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'algorithm': self.algorithm_name,
            'model_variant': self.model_variant,
            'timesteps': self.num_timesteps,
            'eval_score': eval_score,
            'model_path': model_path,
            'recent_rewards': list(self.episode_rewards)[-20:]
        }
        
        eval_file = f"{self.training_system.output_dir}/evaluations/{self.algorithm_name.lower()}_variant_{self.model_variant}_evaluations.json"
        
        # Load existing data or create new
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        existing_data.append(eval_data)
        
        with open(eval_file, 'w') as f:
            json.dump(existing_data, f, indent=2)

class EnhancedMetricsCallback(BaseCallback):
    """
    Callback to track enhanced soccer-specific metrics during training.

    Collects goals, possession, collisions, and out-of-bounds metrics at regular intervals
    throughout the training process, not just at the end.
    """

    def __init__(self, eval_env: Monitor, eval_freq: int = 15000, n_eval_episodes: int = 10,
                 log_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path

        # History storage for enhanced metrics
        self.timesteps_history = []
        self.goals_scored_history = []
        self.possession_rate_history = []
        self.collision_rate_history = []
        self.out_of_bounds_rate_history = []
        self.mean_reward_history = []
        self.episode_length_history = []

    def _on_step(self) -> bool:
        """Called at every step during training"""
        # Evaluate at specified frequency
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_enhanced_metrics()
        return True

    def _evaluate_enhanced_metrics(self):
        """Run evaluation episodes and collect enhanced metrics"""
        unwrapped_env = self.eval_env.unwrapped
        POSSESSION_DISTANCE_THRESHOLD = unwrapped_env.field_config.meters_to_pixels(0.3)
        COLLISION_DISTANCE_THRESHOLD = unwrapped_env.collision_distance

        episode_rewards = []
        episode_lengths = []
        total_goals = 0
        total_possession_steps = 0
        total_collisions = 0
        total_out_of_bounds = 0
        total_steps = 0

        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            episode_possession_steps = 0
            collision_occurred = False
            out_of_bounds_occurred = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Track possession
                robot_ball_distance = np.linalg.norm(
                    unwrapped_env.robot_pos - unwrapped_env.ball_pos
                )
                if robot_ball_distance < POSSESSION_DISTANCE_THRESHOLD:
                    episode_possession_steps += 1

                # Track collisions (binary per episode)
                robot_opponent_distance = np.linalg.norm(
                    unwrapped_env.robot_pos - unwrapped_env.opponent_pos
                )
                if not collision_occurred and robot_opponent_distance < COLLISION_DISTANCE_THRESHOLD:
                    total_collisions += 1
                    collision_occurred = True

                # Track out of bounds (binary per episode)
                if not out_of_bounds_occurred and unwrapped_env._check_ball_out_of_play():
                    total_out_of_bounds += 1
                    out_of_bounds_occurred = True

                # Track goals
                if unwrapped_env._check_goal():
                    total_goals += 1

                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            total_possession_steps += episode_possession_steps
            total_steps += episode_steps

        # Calculate aggregate metrics
        mean_reward = np.mean(episode_rewards)
        mean_episode_length = np.mean(episode_lengths)
        possession_rate = (total_possession_steps / total_steps) * 100 if total_steps > 0 else 0
        collision_rate = (total_collisions / self.n_eval_episodes) * 100
        out_of_bounds_rate = (total_out_of_bounds / self.n_eval_episodes) * 100

        # Store in history
        self.timesteps_history.append(self.num_timesteps)
        self.goals_scored_history.append(total_goals)
        self.possession_rate_history.append(possession_rate)
        self.collision_rate_history.append(collision_rate)
        self.out_of_bounds_rate_history.append(out_of_bounds_rate)
        self.mean_reward_history.append(mean_reward)
        self.episode_length_history.append(mean_episode_length)

        if self.verbose > 0:
            print(f"Enhanced Metrics @ {self.num_timesteps} steps: "
                  f"Goals={total_goals}, Possession={possession_rate:.1f}%, "
                  f"Collisions={collision_rate:.1f}%, OOB={out_of_bounds_rate:.1f}%")

    def get_metrics_history(self) -> Dict[str, list]:
        """Return all collected metrics as a dictionary"""
        return {
            'timesteps': self.timesteps_history,
            'goals_scored': self.goals_scored_history,
            'possession_rate': self.possession_rate_history,
            'collision_rate': self.collision_rate_history,
            'out_of_bounds_rate': self.out_of_bounds_rate_history,
            'mean_reward': self.mean_reward_history,
            'episode_length': self.episode_length_history
        }

    def save_metrics(self):
        """Save metrics history to JSON file"""
        if self.log_path is not None:
            metrics_path = os.path.join(self.log_path, "enhanced_metrics_history.json")
            metrics_data = self.get_metrics_history()
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            if self.verbose > 0:
                print(f"Enhanced metrics saved to {metrics_path}")


def create_ppo_model(env: Monitor, hyperparams: Dict[str, Any],
                    tensorboard_log: str) -> PPO:
    """
    Create PPO model with standardized configuration.

    Args:
        env: Training environment
        hyperparams: Hyperparameter dictionary
        tensorboard_log: TensorBoard log directory

    Returns:
        Configured PPO model
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        clip_range=hyperparams["clip_range"],
        ent_coef=hyperparams["ent_coef"],
        vf_coef=hyperparams["vf_coef"],
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=hyperparams["net_arch"],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=tensorboard_log
    )

def create_ddpg_model(env: Monitor, hyperparams: Dict[str, Any],
                     tensorboard_log: str) -> DDPG:
    """
    Create DDPG model with standardized configuration.

    Args:
        env: Training environment
        hyperparams: Hyperparameter dictionary
        tensorboard_log: TensorBoard log directory

    Returns:
        Configured DDPG model
    """
    # Action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=hyperparams["noise_sigma"] * np.ones(n_actions)
    )

    return DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        batch_size=hyperparams["batch_size"],
        tau=hyperparams["tau"],
        gamma=hyperparams["gamma"],
        action_noise=action_noise,
        train_freq=hyperparams["train_freq"],
        gradient_steps=hyperparams["gradient_steps"],
        policy_kwargs=dict(
            net_arch=hyperparams["net_arch"],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=tensorboard_log
    )

def create_callbacks_and_tracker(
    algorithm_name: str,
    training_system,
    variant_name: str,
    eval_env: Monitor,
    train_env: Optional[Monitor] = None,
    eval_freq: int = 15000,
    n_eval_episodes: int = 5,
    checkpoint_freq: int = 50000,
    verbose: int = 1
) -> Tuple[ModelTracker, EvalCallback, CheckpointCallback, EnhancedMetricsCallback]:
    """
    Create ModelTracker, EvalCallback, CheckpointCallback, and EnhancedMetricsCallback for training.

    Args:
        algorithm_name: Name of algorithm ("PPO", "DDPG", etc.)
        training_system: Training system instance with output_dir, logger, etc.
        variant_name: Name of the variant being trained
        eval_env: Environment for evaluation
        train_env: Training environment (optional, for timestep tracking)
        eval_freq: Frequency of evaluation (steps)
        n_eval_episodes: Number of episodes per evaluation
        checkpoint_freq: Frequency of checkpointing (steps)
        verbose: Verbosity level

    Returns:
        Tuple of (ModelTracker, EvalCallback, CheckpointCallback, EnhancedMetricsCallback)
    """
    # Create ModelTracker
    model_tracker = ModelTracker(algorithm_name, training_system, variant_name, verbose=verbose)

    # Set train_env if provided for timestep tracking
    if train_env is not None:
        model_tracker.train_env = train_env

    # Create EvalCallback with algorithm-specific log path
    # This ensures evaluations.npz is saved with the algorithm's logs
    algo_log_path = f"{training_system.output_dir}/{algorithm_name.lower()}_logs/"
    best_model_path = f"{algo_log_path}best_model/"

    # Ensure directories exist for EvalCallback
    os.makedirs(algo_log_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=algo_log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose
    )

    # Create CheckpointCallback to save model periodically
    checkpoint_path = f"{training_system.output_dir}/checkpoints/{algorithm_name.lower()}/"
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix=f"{algorithm_name.lower()}_checkpoint",
        save_replay_buffer=False,  # Don't save replay buffer to save space
        save_vecnormalize=False,
        verbose=verbose
    )

    # Create EnhancedMetricsCallback to track soccer-specific metrics throughout training
    enhanced_metrics_callback = EnhancedMetricsCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=algo_log_path,
        verbose=verbose
    )

    return model_tracker, eval_callback, checkpoint_callback, enhanced_metrics_callback

def evaluate_model_comprehensive(model: Union[PPO, DDPG], env: Monitor,
                                n_episodes: int = 50, algorithm: str = "") -> Dict[str, Any]:
    """
    Comprehensive model evaluation with detailed metrics including soccer-specific metrics.
    Tracks the same metrics as test_trained_model.py watch_trained_robot function.

    Args:
        model: Trained model to evaluate
        env: Environment for evaluation
        n_episodes: Number of evaluation episodes
        algorithm: Algorithm name for logging

    Returns:
        Dictionary with evaluation results including enhanced metrics
    """
    # Basic metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    # Enhanced metrics (matching test_trained_model.py exactly)
    goals_scored_list = []
    ball_possession_timesteps_list = []
    ball_out_of_bounds_count_list = []
    robot_collisions_count_list = []
    final_ball_distance_list = []
    goal_approaches_list = []

    # Thresholds for metrics calculation
    unwrapped_env = env.unwrapped
    POSSESSION_DISTANCE_THRESHOLD = unwrapped_env.field_config.meters_to_pixels(0.3)
    COLLISION_DISTANCE_THRESHOLD = unwrapped_env.collision_distance
    ATTACKING_THIRD_START = unwrapped_env.field_width * 0.66

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        # Episode-specific enhanced metrics
        episode_goals = 0
        episode_possession_steps = 0
        episode_out_of_bounds = 0  # Binary: 0 or 1
        episode_collisions = 0  # Binary: 0 or 1
        episode_goal_approaches = 0

        # Tracking flags
        collision_occurred = False
        out_of_bounds_occurred = False
        in_attacking_third = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # 1. Ball possession tracking (distance < 0.3m)
            robot_ball_distance = np.linalg.norm(unwrapped_env.robot_pos - unwrapped_env.ball_pos)
            if robot_ball_distance < POSSESSION_DISTANCE_THRESHOLD:
                episode_possession_steps += 1

            # 2. Robot collision tracking (detect first occurrence only)
            robot_opponent_distance = np.linalg.norm(unwrapped_env.robot_pos - unwrapped_env.opponent_pos)
            if not collision_occurred and robot_opponent_distance < COLLISION_DISTANCE_THRESHOLD:
                episode_collisions = 1
                collision_occurred = True

            # 3. Goal approaches tracking (ball enters attacking third)
            ball_in_attacking_third = unwrapped_env.ball_pos[0] > ATTACKING_THIRD_START
            if ball_in_attacking_third and not in_attacking_third:
                episode_goal_approaches += 1
            in_attacking_third = ball_in_attacking_third

            # 4. Ball out of bounds tracking (detect first occurrence only)
            if not out_of_bounds_occurred and unwrapped_env._check_ball_out_of_play():
                episode_out_of_bounds = 1
                out_of_bounds_occurred = True

            # 5. Goals scored tracking
            if unwrapped_env._check_goal():
                episode_goals += 1

            done = terminated or truncated

            # Check for success
            if terminated and hasattr(unwrapped_env, '_check_goal'):
                if unwrapped_env._check_goal():
                    success_count += 1

        # Calculate final ball distance (in meters)
        final_ball_distance_px = np.linalg.norm(unwrapped_env.robot_pos - unwrapped_env.ball_pos)
        final_ball_distance_m = unwrapped_env.field_config.pixels_to_meters(final_ball_distance_px)

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        goals_scored_list.append(episode_goals)
        ball_possession_timesteps_list.append(episode_possession_steps)
        ball_out_of_bounds_count_list.append(episode_out_of_bounds)
        robot_collisions_count_list.append(episode_collisions)
        final_ball_distance_list.append(final_ball_distance_m)
        goal_approaches_list.append(episode_goal_approaches)

    # Calculate possession percentage for enhanced metrics (for plotting)
    possession_pct_list = [(steps / length * 100) if length > 0 else 0
                           for steps, length in zip(ball_possession_timesteps_list, episode_lengths)]

    return {
        'algorithm': algorithm,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'success_rate': (success_count / n_episodes) * 100.0,
        'total_episodes': n_episodes,
        # Enhanced metrics (matching test_trained_model.py exactly)
        'enhanced_metrics': {
            'goals_scored': goals_scored_list,
            'ball_possession_timesteps': ball_possession_timesteps_list,
            'possession_time_pct': possession_pct_list,  # For plotting
            'ball_out_of_bounds_count': ball_out_of_bounds_count_list,
            'robot_collisions_count': robot_collisions_count_list,
            'collision_count': robot_collisions_count_list,  # Alias for plotting
            'out_of_bounds_count': ball_out_of_bounds_count_list,  # Alias for plotting
            'final_ball_distance': final_ball_distance_list,
            'goal_approaches': goal_approaches_list
        }
    }




def comprehensive_final_evaluation(training_system, ppo_results, ddpg_results):
    """Comprehensive evaluation of all best models"""
    training_system.logger.info("="*60)
    training_system.logger.info("FINAL COMPREHENSIVE EVALUATION")
    training_system.logger.info("="*60)
    
    results = {"PPO": {}, "DDPG": {}}
    
    # Evaluate best PPO model
    if training_system.best_ppo_path:
        training_system.logger.info(f"Evaluating best PPO model: {training_system.best_ppo_path}")
        
        try:
            model = PPO.load(training_system.best_ppo_path)
            
            for difficulty in ["easy", "medium", "hard"]:
                training_system.logger.info(f"  Testing PPO on {difficulty}...")
                
                test_env = Monitor(SoccerEnv(difficulty=difficulty, config_path=training_system.config_path))
                
                # Extended evaluation with more episodes
                episode_rewards = []
                episode_lengths = []
                goals_scored = 0
                
                for episode in range(50):  # More episodes for robust evaluation
                    obs, _ = test_env.reset()
                    episode_reward = 0
                    steps = 0
                    
                    for step in range(test_env.max_steps):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = test_env.step(action)
                        episode_reward += reward
                        steps += 1
                        
                        if terminated:
                            if test_env._check_goal():
                                goals_scored += 1
                            break
                        elif truncated:
                            break
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(steps)
                
                test_env.close()
                
                # Calculate statistics
                results["PPO"][difficulty] = {
                    'mean_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'mean_length': np.mean(episode_lengths),
                    'goals_scored': goals_scored,
                    'success_rate': (goals_scored / 50) * 100,
                    'all_rewards': episode_rewards
                }
                
        except Exception as e:
            training_system.logger.error(f"Error evaluating best PPO: {e}")
            results["PPO"] = None
    
    # Evaluate best DDPG model
    if training_system.best_ddpg_path:
        training_system.logger.info(f"Evaluating best DDPG model: {training_system.best_ddpg_path}")
        
        try:
            model = DDPG.load(training_system.best_ddpg_path)
            
            for difficulty in ["easy", "medium", "hard"]:
                training_system.logger.info(f"  Testing DDPG on {difficulty}...")
                
                test_env = Monitor(SoccerEnv(difficulty=difficulty, config_path=training_system.config_path))
                
                # Extended evaluation with more episodes
                episode_rewards = []
                episode_lengths = []
                goals_scored = 0
                
                for episode in range(50):
                    obs, _ = test_env.reset()
                    episode_reward = 0
                    steps = 0
                    
                    for step in range(test_env.max_steps):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = test_env.step(action)
                        episode_reward += reward
                        steps += 1
                        
                        if terminated:
                            if test_env._check_goal():
                                goals_scored += 1
                            break
                        elif truncated:
                            break
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(steps)
                
                test_env.close()
                
                # Calculate statistics
                results["DDPG"][difficulty] = {
                    'mean_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'mean_length': np.mean(episode_lengths),
                    'goals_scored': goals_scored,
                    'success_rate': (goals_scored / 50) * 100,
                    'all_rewards': episode_rewards
                }
                
        except Exception as e:
            training_system.logger.error(f"Error evaluating best DDPG: {e}")
            results["DDPG"] = None
        
    # Save comprehensive results
    results_file = f"{training_system.output_dir}/final_evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg, alg_results in results.items():
            if alg_results:
                json_results[alg] = {}
                for diff, diff_results in alg_results.items():
                    json_results[alg][diff] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                            for k, v in diff_results.items()}
            else:
                json_results[alg] = None
                
        json.dump(json_results, f, indent=2)
    
    return results


def print_policy_detailed_metrics(policy_name: str, n_episodes: int, total_steps: int,
                                  goals_scored_list, episode_rewards, episode_lengths,
                                  possession_timesteps_list, goal_approaches_list,
                                  final_ball_distance_list, collision_count_list,
                                  out_of_bounds_count_list):
    """
    Print detailed metrics for a single policy (matching watch_trained_robot format).
    Reused for both DDPG and PPO in compare mode.
    """
    def calc_stats(data):
        """Calculate mean, std, median, min, max for a list of values"""
        arr = np.array(data)
        return {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'median': np.median(arr),
            'min': np.min(arr),
            'max': np.max(arr)
        }

    # Create metrics dictionary
    all_metrics = {
        'goals_scored': goals_scored_list,
        'ball_possession_timesteps': possession_timesteps_list,
        'ball_out_of_bounds_count': out_of_bounds_count_list,
        'robot_collisions_count': collision_count_list,
        'episode_length': episode_lengths,
        'cumulative_reward': episode_rewards,
        'final_ball_distance': final_ball_distance_list,
        'goal_approaches': goal_approaches_list
    }

    metrics_stats = {key: calc_stats(values) for key, values in all_metrics.items()}

    # Calculate summary statistics
    total_goals = sum(goals_scored_list)
    avg_reward = metrics_stats['cumulative_reward']['mean']
    std_reward = metrics_stats['cumulative_reward']['std']
    avg_length = metrics_stats['episode_length']['mean']
    success_rate = (total_goals / n_episodes) * 100 if n_episodes > 0 else 0

    # Display comprehensive results
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {policy_name}")
    print(f"{'='*70}")
    print(f"Episodes: {n_episodes}")
    print(f"Total Steps: {total_steps}")
    print(f"\nPrimary Metrics:")
    print(f"  Goals Scored: {total_goals}/{n_episodes}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average Episode Length: {avg_length:.1f} steps")

    # Detailed metrics table
    print(f"\nDetailed Performance Metrics:")
    print(f"{'  Metric':<37} {'Mean ± Std':>18} {'[Min, Max]':>15}")
    print(f"  {'-'*68}")

    # Ball possession
    s = metrics_stats['ball_possession_timesteps']
    poss_rate_mean = (s['mean'] / avg_length) * 100 if avg_length > 0 else 0
    print(f"  {'Ball Possession (steps)':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
    print(f"  {'Ball Possession Rate (%)':<35} {poss_rate_mean:>8.2f}%")

    # Goal approaches
    s = metrics_stats['goal_approaches']
    print(f"  {'Goal Approaches':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")

    # Final ball distance
    s = metrics_stats['final_ball_distance']
    print(f"  {'Final Ball Distance (m)':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.2f}, {s['max']:.2f}]")

    # Collisions
    s = metrics_stats['robot_collisions_count']
    collision_rate_pct = (s['mean']) * 100
    print(f"  {'Robot Collision Occurrences':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
    print(f"  {'Episodes with Collisions (%)':<35} {collision_rate_pct:>8.2f}%")

    # Out of bounds
    s = metrics_stats['ball_out_of_bounds_count']
    out_of_bounds_pct = (s['mean']) * 100
    print(f"  {'Ball Out of Bounds Occurrences':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
    print(f"  {'Episodes with Out-of-Bounds (%)':<35} {out_of_bounds_pct:>8.2f}%")

    # Performance assessment
    print(f"\nPERFORMANCE ASSESSMENT:")
    if success_rate >= 70:
        print(f"  Status: EXCELLENT - Success rate {success_rate:.1f}% is very high")
    elif success_rate >= 40:
        print(f"  Status: GOOD - Success rate {success_rate:.1f}% is solid")
    elif success_rate >= 20:
        print(f"  Status: DECENT - Success rate {success_rate:.1f}% shows learning")
    else:
        print(f"  Status: NEEDS IMPROVEMENT - Success rate {success_rate:.1f}% is low")

    if avg_reward > 1000:
        print(f"  Reward Level: EXCELLENT - Average reward {avg_reward:.1f} is very good")
    elif avg_reward > 0:
        print(f"  Reward Level: POSITIVE - Average reward {avg_reward:.1f}")
    else:
        print(f"  Reward Level: NEGATIVE - Average reward {avg_reward:.1f}, needs work")

    print(f"{'='*70}\n")


def compare_three_policies(ppo_model, ddpg_model, config_path: str, difficulty: str = "medium",
                          n_episodes: int = 50, output_dir: str = None, logger=None, debug_display: bool = False, testing_mode: bool = False, handcoded_policy=None):
    """
    Comprehensive 3-way comparison: DDPG vs PPO vs Hand-Coded.

    Evaluates all three policies on the same environment and collects detailed metrics.
    Generates comparison plots and thesis-ready summary report.

    Args:
        ppo_model: Trained PPO model
        ddpg_model: Trained DDPG model
        config_path: Path to field configuration
        difficulty: Environment difficulty level
        n_episodes: Number of evaluation episodes
        output_dir: Directory to save results
        logger: Logger instance
        debug_display: Show debug overlay during evaluation (default: False)
        testing_mode: Enable testing mode (default: False)
        handcoded_policy: Hand-coded policy instance (optional, if None runs 2-way comparison)

    Returns:
        Dictionary with all metrics and file paths
    """
    # Determine if running 2-way or 3-way comparison
    is_3way = handcoded_policy is not None
    comparison_type = "3-WAY" if is_3way else "2-WAY"
    comparison_desc = "DDPG vs PPO vs Hand-Coded" if is_3way else "DDPG vs PPO"

    if logger:
        logger.info("="*60)
        logger.info(f"{comparison_type} POLICY COMPARISON: {comparison_desc}")
        logger.info("="*60)

    # Create evaluation environment
    # Enable rendering if debug_display is requested
    render_mode = "human" if debug_display else None
    eval_env = SoccerEnv(config_path=config_path, difficulty=difficulty, reward_type="standard", render_mode=render_mode, testing_mode=testing_mode)

    # Speed up visualization (match watch_trained_robot behavior)
    eval_env.dt *= 4.0

    # Enable debug display if requested
    if debug_display:
        if hasattr(eval_env, 'enable_debug_display'):
            eval_env.enable_debug_display()
        if hasattr(eval_env, 'set_show_velocities'):
            eval_env.set_show_velocities(True)

    eval_env = Monitor(eval_env)

    # Storage for results
    results = {
        'ddpg': {},
        'ppo': {},
        'metadata': {
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_episodes': n_episodes,
            'difficulty': difficulty,
            'config_path': config_path,
            'comparison_type': comparison_type
        }
    }

    # Add hand-coded results storage if running 3-way comparison
    if is_3way:
        results['handcoded'] = {}

    # Wrap models with ActionSmoothingWrapper (match watch_trained_robot behavior)
    ddpg_smooth = ActionSmoothingWrapper(ddpg_model, smoothing_factor=0.6)
    ppo_smooth = ActionSmoothingWrapper(ppo_model, smoothing_factor=0.6)

    # Build policy list based on comparison type
    policies = [
        ('DDPG', ddpg_smooth, results['ddpg']),
        ('PPO', ppo_smooth, results['ppo'])
    ]

    # Add hand-coded policy if running 3-way comparison
    # Note: OpponentAsPolicy requires environment reference, created later
    handcoded_policy_template = None
    if is_3way:
        handcoded_policy_template = handcoded_policy  # Save for later instantiation
        policies.append(('Rule-Based', 'PLACEHOLDER', results['handcoded']))

    # Evaluate each policy

    for policy_name, policy_model, policy_results in policies:
        # Special handling for OpponentAsPolicy - needs environment reference
        if policy_name == 'Rule-Based' and policy_model == 'PLACEHOLDER':
            # Import OpponentAsPolicy
            from src.environments.soccerenv import OpponentAsPolicy
            # Create policy with direct environment access
            policy_model = OpponentAsPolicy(eval_env.unwrapped, behavior='balanced', debug=False)
        # Terminal output for real-time feedback
        print(f"\n{'='*70}")
        print(f"EVALUATING: {policy_name}")
        print(f"{'='*70}")

        if logger:
            logger.info(f"Evaluating {policy_name} policy...")

        # Collect comprehensive metrics
        episode_rewards = []
        episode_lengths = []
        goals_scored_list = []
        possession_timesteps_list = []
        collision_count_list = []
        out_of_bounds_count_list = []
        final_ball_distance_list = []
        goal_approaches_list = []

        unwrapped_env = eval_env.unwrapped
        POSSESSION_THRESHOLD = unwrapped_env.field_config.meters_to_pixels(0.3)
        COLLISION_THRESHOLD = unwrapped_env.collision_distance
        ATTACKING_THIRD_START = unwrapped_env.field_width * 0.66  # Right third of field

        for episode in range(n_episodes):
            # Terminal output: Episode progress
            print(f"\nEpisode {episode + 1}/{n_episodes}")

            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            # Episode metrics
            episode_goals = 0
            episode_possession_steps = 0
            episode_goal_approaches = 0
            collision_occurred = False
            out_of_bounds_occurred = False
            in_attacking_third = False

            while not done:
                action, _ = policy_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Track possession
                robot_ball_dist = np.linalg.norm(
                    unwrapped_env.robot_pos - unwrapped_env.ball_pos
                )
                if robot_ball_dist < POSSESSION_THRESHOLD:
                    episode_possession_steps += 1

                # Track collisions (binary)
                robot_opp_dist = np.linalg.norm(
                    unwrapped_env.robot_pos - unwrapped_env.opponent_pos
                )
                if not collision_occurred and robot_opp_dist < COLLISION_THRESHOLD:
                    collision_occurred = True

                # Track out of bounds (binary)
                if not out_of_bounds_occurred and unwrapped_env._check_ball_out_of_play():
                    out_of_bounds_occurred = True

                # Track goal approaches (ball enters attacking third)
                ball_in_attacking_third = unwrapped_env.ball_pos[0] > ATTACKING_THIRD_START
                if ball_in_attacking_third and not in_attacking_third:
                    episode_goal_approaches += 1
                in_attacking_third = ball_in_attacking_third

                # Track goals
                if unwrapped_env._check_goal():
                    episode_goals += 1

                # Render with debug overlay if requested
                if debug_display:
                    if hasattr(unwrapped_env, 'set_debug_info'):
                        unwrapped_env.set_debug_info({
                            'policy': policy_name,
                            'step': episode_steps,
                            'reward': reward,
                            'cumulative_reward': episode_reward,
                            'goals': episode_goals,
                            'possession_pct': (episode_possession_steps / (episode_steps)) * 100 if episode_steps > 0 else 0,
                            'robot_pos': unwrapped_env.robot_pos,
                            'ball_pos': unwrapped_env.ball_pos,
                            'action': action,
                            'has_possession': robot_ball_dist < POSSESSION_THRESHOLD,
                            'has_collision': robot_opp_dist < COLLISION_THRESHOLD,
                            'ball_out': unwrapped_env._check_ball_out_of_play()
                        })
                    if hasattr(unwrapped_env, 'render'):
                        unwrapped_env.render()
                    import time
                    # Match watch_trained_robot speed: 0.05s for normal viewing
                    time.sleep(0.05)  # Normal viewing speed (20 FPS, same as watch_trained_robot)

                # Check for termination and print reason
                if terminated:
                    # Check termination reason
                    if unwrapped_env._check_goal():
                        print(f"  SUCCESS! Robot scored a goal!")
                    elif collision_occurred:
                        print(f"  Episode ended: COLLISION with opponent")
                    elif out_of_bounds_occurred:
                        print(f"  Episode ended: Ball OUT OF BOUNDS")
                    elif hasattr(unwrapped_env, '_check_opponent_goal') and unwrapped_env._check_opponent_goal():
                        print(f"  Episode ended: Opponent scored")
                    else:
                        print(f"  Episode ended: Other termination reason")
                    break

                if truncated:
                    print(f"  Episode timeout (reached {unwrapped_env.max_steps} steps)")
                    break

                done = terminated or truncated

            # Calculate final ball distance
            final_ball_dist_px = np.linalg.norm(
                unwrapped_env.robot_pos - unwrapped_env.ball_pos
            )
            final_ball_dist_m = unwrapped_env.field_config.pixels_to_meters(final_ball_dist_px)

            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            goals_scored_list.append(episode_goals)
            possession_timesteps_list.append(episode_possession_steps)
            collision_count_list.append(1 if collision_occurred else 0)
            out_of_bounds_count_list.append(1 if out_of_bounds_occurred else 0)
            final_ball_distance_list.append(final_ball_dist_m)
            goal_approaches_list.append(episode_goal_approaches)

            # Terminal output: Episode statistics
            print(f"  Reward: {episode_reward:.2f} | Steps: {episode_steps} | Goals: {episode_goals} | Possession: {episode_possession_steps} steps\n")

        # Calculate aggregate statistics
        total_goals = sum(goals_scored_list)
        total_possession_steps = sum(possession_timesteps_list)
        total_steps = sum(episode_lengths)
        total_collisions = sum(collision_count_list)
        total_out_of_bounds = sum(out_of_bounds_count_list)

        policy_results['mean_reward'] = float(np.mean(episode_rewards))
        policy_results['std_reward'] = float(np.std(episode_rewards))
        policy_results['median_reward'] = float(np.median(episode_rewards))
        policy_results['min_reward'] = float(np.min(episode_rewards))
        policy_results['max_reward'] = float(np.max(episode_rewards))

        policy_results['total_goals'] = int(total_goals)
        policy_results['goals_per_episode'] = float(total_goals / n_episodes)
        policy_results['goals_std'] = float(np.std(goals_scored_list))

        policy_results['possession_rate'] = float((total_possession_steps / total_steps) * 100) if total_steps > 0 else 0
        policy_results['collision_rate'] = float((total_collisions / n_episodes) * 100)
        policy_results['out_of_bounds_rate'] = float((total_out_of_bounds / n_episodes) * 100)

        policy_results['mean_episode_length'] = float(np.mean(episode_lengths))
        policy_results['std_episode_length'] = float(np.std(episode_lengths))

        policy_results['mean_final_ball_distance'] = float(np.mean(final_ball_distance_list))

        # Store raw data for plotting
        policy_results['episode_rewards'] = episode_rewards
        policy_results['goals_scored_list'] = goals_scored_list

        # Print detailed metrics to terminal (matching watch_trained_robot format)
        print_policy_detailed_metrics(
            policy_name=policy_name,
            n_episodes=n_episodes,
            total_steps=total_steps,
            goals_scored_list=goals_scored_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            possession_timesteps_list=possession_timesteps_list,
            goal_approaches_list=goal_approaches_list,
            final_ball_distance_list=final_ball_distance_list,
            collision_count_list=collision_count_list,
            out_of_bounds_count_list=out_of_bounds_count_list
        )

        if logger:
            logger.info(f"{policy_name} Results:")
            logger.info(f"  Mean Reward: {policy_results['mean_reward']:.2f} ± {policy_results['std_reward']:.2f}")
            logger.info(f"  Total Goals: {policy_results['total_goals']} ({policy_results['goals_per_episode']:.2f} per episode)")
            logger.info(f"  Possession Rate: {policy_results['possession_rate']:.1f}%")
            logger.info(f"  Collision Rate: {policy_results['collision_rate']:.1f}%")

    eval_env.close()

    # Generate visualisations and reports
    if output_dir:
        results['plots'] = create_3way_comparison_plots(results, output_dir)
        results['summary_report'] = generate_thesis_summary(results, output_dir)
        results['json_path'] = save_3way_results_json(results, output_dir)
        results['markdown_path'] = save_3way_results_markdown(results, output_dir)

        # Print comparison summary to terminal (reusing file content)
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        # Read and print the generated summary file
        with open(results['summary_report'], 'r') as f:
            summary_content = f.read()
            print(summary_content)
    return results


def create_3way_comparison_plots(results: Dict, output_dir: str) -> Dict[str, str]:
    """
    Create bar charts and histograms comparing the two policies (DDPG vs PPO).

    Args:
        results: Results dictionary from compare_three_policies
        output_dir: Directory to save plots

    Returns:
        Dictionary with paths to generated plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Academic styling
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 11

    plot_paths = {}

    # Determine if 3-way comparison based on presence of handcoded results
    is_3way = 'handcoded' in results and len(results['handcoded']) > 0

    # Extract data - DDPG, PPO, and optionally Hand-Coded
    if is_3way:
        policies = ['DDPG', 'PPO', 'Hand-Coded']
        policy_keys = ['ddpg', 'ppo', 'handcoded']
        # Colors: Blue (DDPG), Red (PPO), Green (Hand-Coded)
        colors = ['#2E86C1', '#E74C3C', '#27AE60']
    else:
        policies = ['DDPG', 'PPO']
        policy_keys = ['ddpg', 'ppo']
        # Colors: Blue (DDPG), Red (PPO)
        colors = ['#2E86C1', '#E74C3C']

    # ============ PLOT 1: Bar Chart Comparison (6 subplots) ============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    comparison_title = 'Policy Comparison: ' + ' vs '.join(policies)
    fig.suptitle(comparison_title, fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Subplot 1: Mean Reward
    ax = axes[0, 0]
    means = [results[key]['mean_reward'] for key in policy_keys]
    stds = [results[key]['std_reward'] for key in policy_keys]
    bars = ax.bar(policies, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Mean Episode Reward', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f}', ha='center', va='bottom', fontsize=10)

    # Subplot 2: Total Goals
    ax = axes[0, 1]
    goals = [results[key]['total_goals'] for key in policy_keys]
    bars = ax.bar(policies, goals, color=colors, alpha=0.8)
    ax.set_ylabel('Total Goals Scored', fontsize=12)
    ax.set_title('Goal Scoring Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, goal in zip(bars, goals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{goal}', ha='center', va='bottom', fontsize=10)

    # Subplot 3: Ball Possession Rate
    ax = axes[0, 2]
    possession = [results[key]['possession_rate'] for key in policy_keys]
    bars = ax.bar(policies, possession, color=colors, alpha=0.8)
    ax.set_ylabel('Ball Possession Rate (%)', fontsize=12)
    ax.set_title('Ball Possession Efficiency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    for bar, poss in zip(bars, possession):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{poss:.1f}%', ha='center', va='bottom', fontsize=10)

    # Subplot 4: Collision Rate
    ax = axes[1, 0]
    collisions = [results[key]['collision_rate'] for key in policy_keys]
    bars = ax.bar(policies, collisions, color=colors, alpha=0.8)
    ax.set_ylabel('Collision Rate (%)', fontsize=12)
    ax.set_title('Collision Frequency (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, coll in zip(bars, collisions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{coll:.1f}%', ha='center', va='bottom', fontsize=10)

    # Subplot 5: Mean Episode Length
    ax = axes[1, 1]
    lengths = [results[key]['mean_episode_length'] for key in policy_keys]
    length_stds = [results[key]['std_episode_length'] for key in policy_keys]
    bars = ax.bar(policies, lengths, yerr=length_stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Episode Length (steps)', fontsize=12)
    ax.set_title('Episode Duration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{length:.0f}', ha='center', va='bottom', fontsize=10)

    # Subplot 6: Out of Bounds Rate
    ax = axes[1, 2]
    oob = [results[key]['out_of_bounds_rate'] for key in policy_keys]
    bars = ax.bar(policies, oob, color=colors, alpha=0.8)
    ax.set_ylabel('Out of Bounds Rate (%)', fontsize=12)
    ax.set_title('Out of Bounds Incidents (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, oob_val in zip(bars, oob):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{oob_val:.1f}%', ha='center', va='bottom', fontsize=10)

    bar_chart_path = os.path.join(output_dir, "2way_comparison_bar_charts.png")
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['bar_charts'] = bar_chart_path

    # ============ PLOT 2: Reward Distribution Histograms ============
    num_policies = len(policies)
    fig, axes = plt.subplots(1, num_policies, figsize=(6 * num_policies, 5))
    histogram_title = 'Reward Distribution: ' + ' vs '.join(policies)
    fig.suptitle(histogram_title, fontsize=16, fontweight='bold')
    plt.subplots_adjust(wspace=0.3)

    # Handle both single and multiple subplots
    if num_policies == 1:
        axes = [axes]

    for idx, (policy_name, policy_key, color) in enumerate(zip(policies, policy_keys, colors)):
        ax = axes[idx]
        rewards = results[policy_key]['episode_rewards']
        ax.hist(rewards, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.0f}')
        ax.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.0f}')
        ax.set_xlabel('Episode Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{policy_name} Policy', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    histogram_path = os.path.join(output_dir, "ddpg_vs_ppo_reward_histograms.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['histograms'] = histogram_path

    # ============ PLOT 3: Goals Over Episodes (Line Plot) ============
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Goals Scored Over Episodes', fontsize=16, fontweight='bold')

    for policy_name, policy_key, color in zip(policies, policy_keys, colors):
        goals_list = results[policy_key]['goals_scored_list']
        cumulative_goals = np.cumsum(goals_list)
        ax.plot(range(1, len(cumulative_goals) + 1), cumulative_goals,
                linewidth=2.5, color=color, label=policy_name, alpha=0.9, marker='o', markersize=3, markevery=5)

    ax.set_xlabel('Episode Number', fontsize=12)
    ax.set_ylabel('Cumulative Goals Scored', fontsize=12)
    ax.set_title('Goal Scoring Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    goals_line_path = os.path.join(output_dir, "2way_goals_over_time.png")
    plt.savefig(goals_line_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['goals_over_time'] = goals_line_path

    return plot_paths


def generate_thesis_summary(results: Dict, output_dir: str) -> str:
    """
    Generate thesis-ready text summary with clear numerical comparisons.

    Supports both 2-way (DDPG vs PPO) and 3-way (DDPG vs PPO vs Hand-Coded) comparisons.

    Args:
        results: Results dictionary from compare_three_policies
        output_dir: Directory to save summary

    Returns:
        Path to summary text file
    """
    # Extract data
    ddpg = results['ddpg']
    ppo = results['ppo']
    metadata = results['metadata']

    # Check if 3-way comparison (has hand-coded results)
    is_3way = 'handcoded' in results and len(results['handcoded']) > 0
    if is_3way:
        handcoded = results['handcoded']

    # Calculate percentage differences
    def pct_diff(val1, val2):
        if val2 == 0:
            return "N/A"
        return f"{((val1 - val2) / abs(val2)) * 100:.1f}%"

    # Determine winner for each metric (handle ties properly)
    def determine_winner(ddpg_val, ppo_val, higher_is_better=True):
        """Determine winner, handling ties correctly"""
        if ddpg_val == ppo_val:
            return "TIE"
        if higher_is_better:
            return "DDPG" if ddpg_val > ppo_val else "PPO"
        else:
            return "DDPG" if ddpg_val < ppo_val else "PPO"

    reward_winner = determine_winner(ddpg['mean_reward'], ppo['mean_reward'], higher_is_better=True)
    goals_winner = determine_winner(ddpg['total_goals'], ppo['total_goals'], higher_is_better=True)
    possession_winner = determine_winner(ddpg['possession_rate'], ppo['possession_rate'], higher_is_better=True)
    collision_winner = determine_winner(ddpg['collision_rate'], ppo['collision_rate'], higher_is_better=False)
    oob_winner = determine_winner(ddpg['out_of_bounds_rate'], ppo['out_of_bounds_rate'], higher_is_better=False)

    # Generate comparison text handling ties
    if reward_winner == "TIE":
        reward_comparison = "Both algorithms achieved identical mean rewards."
    else:
        reward_comparison = f"{reward_winner} achieved {pct_diff(ddpg['mean_reward'], ppo['mean_reward'])} {'higher' if reward_winner == 'DDPG' else 'lower'} mean reward than {'PPO' if reward_winner == 'DDPG' else 'DDPG'}."

    if goals_winner == "TIE":
        goals_comparison = "Both algorithms scored the same number of goals."
    else:
        goals_comparison = f"{goals_winner} scored {abs(ddpg['total_goals'] - ppo['total_goals'])} more goals than {'PPO' if goals_winner == 'DDPG' else 'DDPG'}."

    if possession_winner == "TIE":
        possession_comparison = "Both algorithms achieved identical ball possession rates."
    else:
        possession_comparison = f"{possession_winner} maintained higher ball possession, outperforming {'PPO' if possession_winner == 'DDPG' else 'DDPG'} by {abs(ddpg['possession_rate'] - ppo['possession_rate']):.1f} percentage points."

    if collision_winner == "TIE":
        collision_comparison = "Both algorithms had identical collision rates."
    else:
        collision_comparison = f"{collision_winner} demonstrated better collision avoidance."

    if oob_winner == "TIE":
        oob_comparison = "Both algorithms had identical out-of-bounds rates."
    else:
        oob_comparison = f"{oob_winner} maintained better ball control with fewer out-of-bounds incidents."

    # Generate title based on comparison type
    if is_3way:
        summary_title = "# DDPG vs PPO vs Hand-Coded Policy Comparison Summary"
    else:
        summary_title = "# DDPG vs PPO Policy Comparison Summary"

    summary = f"""
{summary_title}
Generated: {metadata['date']}
Evaluation: {metadata['n_episodes']} episodes on {metadata['difficulty']} difficulty

## Individual Policy Performance

### DDPG Detailed Results

**Primary Metrics:**
- Goals Scored: {ddpg['total_goals']}/{metadata['n_episodes']}
- Success Rate: {(ddpg['total_goals'] / metadata['n_episodes']) * 100:.1f}%
- Average Reward: {ddpg['mean_reward']:.2f} ± {ddpg['std_reward']:.2f}
- Average Episode Length: {ddpg['mean_episode_length']:.1f} steps

**Detailed Performance Metrics:**
- Ball Possession Rate: {ddpg['possession_rate']:.1f}%
- Robot Collision Rate: {ddpg['collision_rate']:.1f}%
- Ball Out-of-Bounds Rate: {ddpg['out_of_bounds_rate']:.1f}%
- Mean Final Ball Distance: {ddpg['mean_final_ball_distance']:.2f} meters

**Performance Assessment:**
{('EXCELLENT - Success rate ' + f"{(ddpg['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is very high') if (ddpg['total_goals'] / metadata['n_episodes']) * 100 >= 70 else ('GOOD - Success rate ' + f"{(ddpg['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is solid') if (ddpg['total_goals'] / metadata['n_episodes']) * 100 >= 40 else ('DECENT - Success rate ' + f"{(ddpg['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% shows learning') if (ddpg['total_goals'] / metadata['n_episodes']) * 100 >= 20 else ('NEEDS IMPROVEMENT - Success rate ' + f"{(ddpg['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is low')}
{('EXCELLENT - Average reward ' + f"{ddpg['mean_reward']:.1f}" + ' is very good') if ddpg['mean_reward'] > 1000 else ('POSITIVE - Average reward ' + f"{ddpg['mean_reward']:.1f}") if ddpg['mean_reward'] > 0 else ('NEGATIVE - Average reward ' + f"{ddpg['mean_reward']:.1f}" + ', needs work')}

### PPO Detailed Results

**Primary Metrics:**
- Goals Scored: {ppo['total_goals']}/{metadata['n_episodes']}
- Success Rate: {(ppo['total_goals'] / metadata['n_episodes']) * 100:.1f}%
- Average Reward: {ppo['mean_reward']:.2f} ± {ppo['std_reward']:.2f}
- Average Episode Length: {ppo['mean_episode_length']:.1f} steps

**Detailed Performance Metrics:**
- Ball Possession Rate: {ppo['possession_rate']:.1f}%
- Robot Collision Rate: {ppo['collision_rate']:.1f}%
- Ball Out-of-Bounds Rate: {ppo['out_of_bounds_rate']:.1f}%
- Mean Final Ball Distance: {ppo['mean_final_ball_distance']:.2f} meters

**Performance Assessment:**
{('EXCELLENT - Success rate ' + f"{(ppo['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is very high') if (ppo['total_goals'] / metadata['n_episodes']) * 100 >= 70 else ('GOOD - Success rate ' + f"{(ppo['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is solid') if (ppo['total_goals'] / metadata['n_episodes']) * 100 >= 40 else ('DECENT - Success rate ' + f"{(ppo['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% shows learning') if (ppo['total_goals'] / metadata['n_episodes']) * 100 >= 20 else ('NEEDS IMPROVEMENT - Success rate ' + f"{(ppo['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is low')}
{('EXCELLENT - Average reward ' + f"{ppo['mean_reward']:.1f}" + ' is very good') if ppo['mean_reward'] > 1000 else ('POSITIVE - Average reward ' + f"{ppo['mean_reward']:.1f}") if ppo['mean_reward'] > 0 else ('NEGATIVE - Average reward ' + f"{ppo['mean_reward']:.1f}" + ', needs work')}
"""

    # Add Hand-Coded section if 3-way comparison
    if is_3way:
        handcoded_section = f"""
### Hand-Coded Detailed Results

**Primary Metrics:**
- Goals Scored: {handcoded['total_goals']}/{metadata['n_episodes']}
- Success Rate: {(handcoded['total_goals'] / metadata['n_episodes']) * 100:.1f}%
- Average Reward: {handcoded['mean_reward']:.2f} ± {handcoded['std_reward']:.2f}
- Average Episode Length: {handcoded['mean_episode_length']:.1f} steps

**Detailed Performance Metrics:**
- Ball Possession Rate: {handcoded['possession_rate']:.1f}%
- Robot Collision Rate: {handcoded['collision_rate']:.1f}%
- Ball Out-of-Bounds Rate: {handcoded['out_of_bounds_rate']:.1f}%
- Mean Final Ball Distance: {handcoded['mean_final_ball_distance']:.2f} meters

**Performance Assessment:**
{('EXCELLENT - Success rate ' + f"{(handcoded['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is very high') if (handcoded['total_goals'] / metadata['n_episodes']) * 100 >= 70 else ('GOOD - Success rate ' + f"{(handcoded['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is solid') if (handcoded['total_goals'] / metadata['n_episodes']) * 100 >= 40 else ('DECENT - Success rate ' + f"{(handcoded['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% shows learning') if (handcoded['total_goals'] / metadata['n_episodes']) * 100 >= 20 else ('NEEDS IMPROVEMENT - Success rate ' + f"{(handcoded['total_goals'] / metadata['n_episodes']) * 100:.1f}" + '% is low')}
{('EXCELLENT - Average reward ' + f"{handcoded['mean_reward']:.1f}" + ' is very good') if handcoded['mean_reward'] > 1000 else ('POSITIVE - Average reward ' + f"{handcoded['mean_reward']:.1f}") if handcoded['mean_reward'] > 0 else ('NEGATIVE - Average reward ' + f"{handcoded['mean_reward']:.1f}" + ', needs work')}
"""
        summary += handcoded_section

    summary += """
## Comparative Analysis

### Mean Episode Reward
- DDPG: {ddpg['mean_reward']:.2f} ± {ddpg['std_reward']:.2f}
- PPO: {ppo['mean_reward']:.2f} ± {ppo['std_reward']:.2f}

{reward_comparison}

### Goal Scoring Performance
- DDPG: {ddpg['total_goals']} goals ({ddpg['goals_per_episode']:.2f} per episode, σ={ddpg['goals_std']:.2f})
- PPO: {ppo['total_goals']} goals ({ppo['goals_per_episode']:.2f} per episode, σ={ppo['goals_std']:.2f})

{goals_comparison}

### Ball Possession Rate
- DDPG: {ddpg['possession_rate']:.1f}%
- PPO: {ppo['possession_rate']:.1f}%

{possession_comparison}

### Collision Avoidance
- DDPG: {ddpg['collision_rate']:.1f}% collision rate
- PPO: {ppo['collision_rate']:.1f}% collision rate

{collision_comparison}

### Out of Bounds Control
- DDPG: {ddpg['out_of_bounds_rate']:.1f}% out of bounds rate
- PPO: {ppo['out_of_bounds_rate']:.1f}% out of bounds rate

{oob_comparison}

### Episode Length
- DDPG: {ddpg['mean_episode_length']:.1f} ± {ddpg['std_episode_length']:.1f} steps
- PPO: {ppo['mean_episode_length']:.1f} ± {ppo['std_episode_length']:.1f} steps

## Summary Table
"""

    # Add summary table with conditional Hand-Coded column
    if is_3way:
        summary += f"""
| Metric                    | DDPG           | PPO            | Hand-Coded     |
|---------------------------|----------------|----------------|----------------|
| Mean Reward               | {ddpg['mean_reward']:>7.1f} ± {ddpg['std_reward']:<6.1f} | {ppo['mean_reward']:>7.1f} ± {ppo['std_reward']:<6.1f} | {handcoded['mean_reward']:>7.1f} ± {handcoded['std_reward']:<6.1f} |
| Total Goals               | {ddpg['total_goals']:>14} | {ppo['total_goals']:>14} | {handcoded['total_goals']:>14} |
| Goals per Episode         | {ddpg['goals_per_episode']:>14.2f} | {ppo['goals_per_episode']:>14.2f} | {handcoded['goals_per_episode']:>14.2f} |
| Ball Possession (%)       | {ddpg['possession_rate']:>14.1f} | {ppo['possession_rate']:>14.1f} | {handcoded['possession_rate']:>14.1f} |
| Collision Rate (%)        | {ddpg['collision_rate']:>14.1f} | {ppo['collision_rate']:>14.1f} | {handcoded['collision_rate']:>14.1f} |
| Out of Bounds (%)         | {ddpg['out_of_bounds_rate']:>14.1f} | {ppo['out_of_bounds_rate']:>14.1f} | {handcoded['out_of_bounds_rate']:>14.1f} |
| Mean Episode Length       | {ddpg['mean_episode_length']:>14.1f} | {ppo['mean_episode_length']:>14.1f} | {handcoded['mean_episode_length']:>14.1f} |
| Median Reward             | {ddpg['median_reward']:>14.1f} | {ppo['median_reward']:>14.1f} | {handcoded['median_reward']:>14.1f} |
| Min Reward                | {ddpg['min_reward']:>14.1f} | {ppo['min_reward']:>14.1f} | {handcoded['min_reward']:>14.1f} |
| Max Reward                | {ddpg['max_reward']:>14.1f} | {ppo['max_reward']:>14.1f} | {handcoded['max_reward']:>14.1f} |
"""
    else:
        summary += f"""
| Metric                    | DDPG           | PPO            |
|---------------------------|----------------|----------------|
| Mean Reward               | {ddpg['mean_reward']:>7.1f} ± {ddpg['std_reward']:<6.1f} | {ppo['mean_reward']:>7.1f} ± {ppo['std_reward']:<6.1f} |
| Total Goals               | {ddpg['total_goals']:>14} | {ppo['total_goals']:>14} |
| Goals per Episode         | {ddpg['goals_per_episode']:>14.2f} | {ppo['goals_per_episode']:>14.2f} |
| Ball Possession (%)       | {ddpg['possession_rate']:>14.1f} | {ppo['possession_rate']:>14.1f} |
| Collision Rate (%)        | {ddpg['collision_rate']:>14.1f} | {ppo['collision_rate']:>14.1f} |
| Out of Bounds (%)         | {ddpg['out_of_bounds_rate']:>14.1f} | {ppo['out_of_bounds_rate']:>14.1f} |
| Mean Episode Length       | {ddpg['mean_episode_length']:>14.1f} | {ppo['mean_episode_length']:>14.1f} |
| Median Reward             | {ddpg['median_reward']:>14.1f} | {ppo['median_reward']:>14.1f} |
| Min Reward                | {ddpg['min_reward']:>14.1f} | {ppo['min_reward']:>14.1f} |
| Max Reward                | {ddpg['max_reward']:>14.1f} | {ppo['max_reward']:>14.1f} |
"""

    summary += """

## Key Findings

1. **Best Overall Performance**: {reward_winner + ' achieved the highest mean reward.' if reward_winner != 'TIE' else 'Both algorithms achieved identical mean rewards.'}

2. **Best Goal Scorer**: {goals_winner + ' scored the most goals.' if goals_winner != 'TIE' else 'Both algorithms scored the same number of goals.'}

3. **Best Ball Control**: {possession_winner + ' maintained the highest possession rate.' if possession_winner != 'TIE' else 'Both algorithms had identical possession rates.'}

4. **Best Collision Avoidance**: {collision_winner + ' had the lowest collision rate.' if collision_winner != 'TIE' else 'Both algorithms had identical collision rates.'}

5. **Best Out-of-Bounds Control**: {oob_winner + ' had the lowest out-of-bounds rate.' if oob_winner != 'TIE' else 'Both algorithms had identical out-of-bounds rates.'}

## Conclusion

This comparison demonstrates the relative performance of DDPG and PPO algorithms in the soccer environment.
Both algorithms show learned behavior, with differences in risk-taking, ball control, and goal-scoring strategies.
"""

    summary_path = os.path.join(output_dir, "ddpg_vs_ppo_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)

    return summary_path


def save_3way_results_json(results: Dict, output_dir: str) -> str:
    """Save complete results to JSON file"""
    json_path = os.path.join(output_dir, "2way_comparison_results.json")

    # Create clean copy without raw episode data for JSON
    clean_results = {
        'metadata': results['metadata'],
        'ddpg': {k: v for k, v in results['ddpg'].items() if k not in ['episode_rewards', 'goals_scored_list']},
        'ppo': {k: v for k, v in results['ppo'].items() if k not in ['episode_rewards', 'goals_scored_list']},
        # 'handcoded': {k: v for k, v in results['handcoded'].items() if k not in ['episode_rewards', 'goals_scored_list']}
    }

    with open(json_path, 'w') as f:
        json.dump(clean_results, f, indent=2)

    return json_path


def save_3way_results_markdown(results: Dict, output_dir: str) -> str:
    """Save results in markdown table format (supports both 2-way and 3-way)"""
    # Determine if 3-way comparison
    is_3way = 'handcoded' in results and len(results['handcoded']) > 0

    md_path = os.path.join(output_dir, "2way_comparison_table.md")

    ddpg = results['ddpg']
    ppo = results['ppo']

    # Calculate success rates for individual assessments
    ddpg_success_rate = (ddpg['total_goals'] / results['metadata']['n_episodes']) * 100
    ppo_success_rate = (ppo['total_goals'] / results['metadata']['n_episodes']) * 100

    # Add hand-coded if 3-way comparison
    if is_3way:
        handcoded = results['handcoded']
        handcoded_success_rate = (handcoded['total_goals'] / results['metadata']['n_episodes']) * 100

    # Performance assessments
    def get_status_assessment(success_rate):
        if success_rate >= 70:
            return f"EXCELLENT - Success rate {success_rate:.1f}% is very high"
        elif success_rate >= 40:
            return f"GOOD - Success rate {success_rate:.1f}% is solid"
        elif success_rate >= 20:
            return f"DECENT - Success rate {success_rate:.1f}% shows learning"
        else:
            return f"NEEDS IMPROVEMENT - Success rate {success_rate:.1f}% is low"

    def get_reward_assessment(mean_reward):
        if mean_reward > 1000:
            return f"EXCELLENT - Average reward {mean_reward:.1f} is very good"
        elif mean_reward > 0:
            return f"POSITIVE - Average reward {mean_reward:.1f}"
        else:
            return f"NEGATIVE - Average reward {mean_reward:.1f}, needs work"

    # Generate title based on comparison type
    comparison_title = "# 3-Way Policy Comparison" if is_3way else "# 2-Way Policy Comparison"

    markdown = f"""{comparison_title}

**Evaluation Date:** {results['metadata']['date']}
**Episodes:** {results['metadata']['n_episodes']}
**Difficulty:** {results['metadata']['difficulty']}

---

## Individual Policy Performance

### DDPG Detailed Results

**Primary Metrics:**
- Goals Scored: {ddpg['total_goals']}/{results['metadata']['n_episodes']}
- Success Rate: {ddpg_success_rate:.1f}%
- Average Reward: {ddpg['mean_reward']:.2f} ± {ddpg['std_reward']:.2f}
- Average Episode Length: {ddpg['mean_episode_length']:.1f} steps

**Detailed Performance Metrics:**
- Ball Possession Rate: {ddpg['possession_rate']:.1f}%
- Robot Collision Rate: {ddpg['collision_rate']:.1f}%
- Ball Out-of-Bounds Rate: {ddpg['out_of_bounds_rate']:.1f}%
- Mean Final Ball Distance: {ddpg['mean_final_ball_distance']:.2f} meters

**Performance Assessment:**
- Status: {get_status_assessment(ddpg_success_rate)}
- Reward: {get_reward_assessment(ddpg['mean_reward'])}

---

### PPO Detailed Results

**Primary Metrics:**
- Goals Scored: {ppo['total_goals']}/{results['metadata']['n_episodes']}
- Success Rate: {ppo_success_rate:.1f}%
- Average Reward: {ppo['mean_reward']:.2f} ± {ppo['std_reward']:.2f}
- Average Episode Length: {ppo['mean_episode_length']:.1f} steps

**Detailed Performance Metrics:**
- Ball Possession Rate: {ppo['possession_rate']:.1f}%
- Robot Collision Rate: {ppo['collision_rate']:.1f}%
- Ball Out-of-Bounds Rate: {ppo['out_of_bounds_rate']:.1f}%
- Mean Final Ball Distance: {ppo['mean_final_ball_distance']:.2f} meters

**Performance Assessment:**
- Status: {get_status_assessment(ppo_success_rate)}
- Reward: {get_reward_assessment(ppo['mean_reward'])}

---
"""

    # Add Hand-Coded section if 3-way comparison
    if is_3way:
        markdown += f"""
### Hand-Coded Detailed Results

**Primary Metrics:**
- Goals Scored: {handcoded['total_goals']}/{results['metadata']['n_episodes']}
- Success Rate: {handcoded_success_rate:.1f}%
- Average Reward: {handcoded['mean_reward']:.2f} ± {handcoded['std_reward']:.2f}
- Average Episode Length: {handcoded['mean_episode_length']:.1f} steps

**Detailed Performance Metrics:**
- Ball Possession Rate: {handcoded['possession_rate']:.1f}%
- Robot Collision Rate: {handcoded['collision_rate']:.1f}%
- Ball Out-of-Bounds Rate: {handcoded['out_of_bounds_rate']:.1f}%
- Mean Final Ball Distance: {handcoded['mean_final_ball_distance']:.2f} meters

**Performance Assessment:**
- Status: {get_status_assessment(handcoded_success_rate)}
- Reward: {get_reward_assessment(handcoded['mean_reward'])}

---
"""

    # Add comparison table with conditional columns
    markdown += "\n## Comparison Table\n\n"
    if is_3way:
        markdown += f"""| Metric | DDPG | PPO | Hand-Coded |
|--------|------|-----|------------|
| Mean Reward | {ddpg['mean_reward']:.2f} ± {ddpg['std_reward']:.2f} | {ppo['mean_reward']:.2f} ± {ppo['std_reward']:.2f} | {handcoded['mean_reward']:.2f} ± {handcoded['std_reward']:.2f} |
| Median Reward | {ddpg['median_reward']:.2f} | {ppo['median_reward']:.2f} | {handcoded['median_reward']:.2f} |
| Total Goals | {ddpg['total_goals']} | {ppo['total_goals']} | {handcoded['total_goals']} |
| Goals/Episode | {ddpg['goals_per_episode']:.3f} | {ppo['goals_per_episode']:.3f} | {handcoded['goals_per_episode']:.3f} |
| Possession (%) | {ddpg['possession_rate']:.1f}% | {ppo['possession_rate']:.1f}% | {handcoded['possession_rate']:.1f}% |
| Collision Rate (%) | {ddpg['collision_rate']:.1f}% | {ppo['collision_rate']:.1f}% | {handcoded['collision_rate']:.1f}% |
| Out of Bounds (%) | {ddpg['out_of_bounds_rate']:.1f}% | {ppo['out_of_bounds_rate']:.1f}% | {handcoded['out_of_bounds_rate']:.1f}% |
| Episode Length | {ddpg['mean_episode_length']:.1f} ± {ddpg['std_episode_length']:.1f} | {ppo['mean_episode_length']:.1f} ± {ppo['std_episode_length']:.1f} | {handcoded['mean_episode_length']:.1f} ± {handcoded['std_episode_length']:.1f} |

## Winner Summary

- **Best Reward:** {"DDPG" if ddpg['mean_reward'] > max(ppo['mean_reward'], handcoded['mean_reward']) else "PPO" if ppo['mean_reward'] > max(ddpg['mean_reward'], handcoded['mean_reward']) else "Hand-Coded" if handcoded['mean_reward'] > max(ddpg['mean_reward'], ppo['mean_reward']) else "Tie"}
- **Most Goals:** {"DDPG" if ddpg['total_goals'] > max(ppo['total_goals'], handcoded['total_goals']) else "PPO" if ppo['total_goals'] > max(ddpg['total_goals'], handcoded['total_goals']) else "Hand-Coded" if handcoded['total_goals'] > max(ddpg['total_goals'], ppo['total_goals']) else "Tie"}
- **Best Possession:** {"DDPG" if ddpg['possession_rate'] > max(ppo['possession_rate'], handcoded['possession_rate']) else "PPO" if ppo['possession_rate'] > max(ddpg['possession_rate'], handcoded['possession_rate']) else "Hand-Coded" if handcoded['possession_rate'] > max(ddpg['possession_rate'], ppo['possession_rate']) else "Tie"}
- **Fewest Collisions:** {"DDPG" if ddpg['collision_rate'] < min(ppo['collision_rate'], handcoded['collision_rate']) else "PPO" if ppo['collision_rate'] < min(ddpg['collision_rate'], handcoded['collision_rate']) else "Hand-Coded" if handcoded['collision_rate'] < min(ddpg['collision_rate'], ppo['collision_rate']) else "Tie"}
"""
    else:
        markdown += f"""| Metric | DDPG | PPO |
|--------|------|-----|
| Mean Reward | {ddpg['mean_reward']:.2f} ± {ddpg['std_reward']:.2f} | {ppo['mean_reward']:.2f} ± {ppo['std_reward']:.2f} |
| Median Reward | {ddpg['median_reward']:.2f} | {ppo['median_reward']:.2f} |
| Total Goals | {ddpg['total_goals']} | {ppo['total_goals']} |
| Goals/Episode | {ddpg['goals_per_episode']:.3f} | {ppo['goals_per_episode']:.3f} |
| Possession (%) | {ddpg['possession_rate']:.1f}% | {ppo['possession_rate']:.1f}% |
| Collision Rate (%) | {ddpg['collision_rate']:.1f}% | {ppo['collision_rate']:.1f}% |
| Out of Bounds (%) | {ddpg['out_of_bounds_rate']:.1f}% | {ppo['out_of_bounds_rate']:.1f}% |
| Episode Length | {ddpg['mean_episode_length']:.1f} ± {ddpg['std_episode_length']:.1f} | {ppo['mean_episode_length']:.1f} ± {ppo['std_episode_length']:.1f} |

## Winner Summary

- **Best Reward:** {"DDPG" if ddpg['mean_reward'] > ppo['mean_reward'] else "PPO" if ppo['mean_reward'] > ddpg['mean_reward'] else "Tie"}
- **Most Goals:** {"DDPG" if ddpg['total_goals'] > ppo['total_goals'] else "PPO" if ppo['total_goals'] > ddpg['total_goals'] else "Tie"}
- **Best Possession:** {"DDPG" if ddpg['possession_rate'] > ppo['possession_rate'] else "PPO" if ppo['possession_rate'] > ddpg['possession_rate'] else "Tie"}
- **Fewest Collisions:** {"DDPG" if ddpg['collision_rate'] < ppo['collision_rate'] else "PPO" if ppo['collision_rate'] < ddpg['collision_rate'] else "Tie"}
"""

    with open(md_path, 'w') as f:
        f.write(markdown)

    return md_path


# def create_comprehensive_comparison_plots(training_system, ppo_results, ddpg_results, final_evaluation):
#     """Create comprehensive plots comparing all model variants"""
#     training_system.logger.info("Creating comprehensive comparison plots...")
    
#     try:
#         # Create a large figure with multiple subplots - increased spacing for clarity
#         fig = plt.figure(figsize=(24, 20))
#         gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.4, top=0.88, bottom=0.08, left=0.08, right=0.95)
        
#         fig.suptitle(f'Multi-Model Training Results - {training_system.timestamp}', 
#                     fontsize=18, fontweight='bold', y=0.98)
        
#         # Plot 1: PPO Variants Performance Comparison
#         ax1 = fig.add_subplot(gs[0, 0])
#         ppo_names = [result[2] for result in ppo_results]
#         ppo_scores = [result[1] for result in ppo_results]
        
#         bars1 = ax1.bar(ppo_names, ppo_scores, color='skyblue', alpha=0.8)
#         ax1.set_title('PPO Variants Performance', fontweight='bold', fontsize=12)
#         ax1.set_ylabel('Evaluation Score', fontsize=10)
#         ax1.tick_params(axis='x', rotation=45, labelsize=8)
#         ax1.tick_params(axis='y', labelsize=8)
#         ax1.grid(True, alpha=0.3)
        
#         # Highlight best PPO
#         if ppo_scores:
#             best_idx = ppo_scores.index(max(ppo_scores))
#             bars1[best_idx].set_color('gold')
#             bars1[best_idx].set_edgecolor('orange')
#             bars1[best_idx].set_linewidth(2)
        
#         # Plot 2: DDPG Variants Performance Comparison
#         ax2 = fig.add_subplot(gs[0, 1])
#         ddpg_names = [result[2] for result in ddpg_results]
#         ddpg_scores = [result[1] for result in ddpg_results]
        
#         bars2 = ax2.bar(ddpg_names, ddpg_scores, color='lightcoral', alpha=0.8)
#         ax2.set_title('DDPG Variants Performance', fontweight='bold', fontsize=12)
#         ax2.set_ylabel('Evaluation Score', fontsize=10)
#         ax2.tick_params(axis='x', rotation=45, labelsize=8)
#         ax2.tick_params(axis='y', labelsize=8)
#         ax2.grid(True, alpha=0.3)
        
#         # Highlight best DDPG
#         if ddpg_scores:
#             best_idx = ddpg_scores.index(max(ddpg_scores))
#             bars2[best_idx].set_color('gold')
#             bars2[best_idx].set_edgecolor('orange')
#             bars2[best_idx].set_linewidth(2)
        
#         # Plot 3: Best Models Difficulty Comparison
#         if final_evaluation["PPO"] and final_evaluation["DDPG"]:
#             ax3 = fig.add_subplot(gs[0, 2])
#             difficulties = ['easy', 'medium', 'hard']
#             x_pos = np.arange(len(difficulties))
#             width = 0.35
            
#             ppo_rewards = [final_evaluation["PPO"][diff]['mean_reward'] for diff in difficulties]
#             ddpg_rewards = [final_evaluation["DDPG"][diff]['mean_reward'] for diff in difficulties]
            
#             ax3.bar(x_pos - width/2, ppo_rewards, width, label='Best PPO', color='skyblue', alpha=0.8)
#             ax3.bar(x_pos + width/2, ddpg_rewards, width, label='Best DDPG', color='lightcoral', alpha=0.8)
#             ax3.set_xlabel('Difficulty Level', fontsize=10)
#             ax3.set_ylabel('Average Reward', fontsize=10)
#             ax3.set_title('Best Models: Difficulty Performance', fontweight='bold', fontsize=12)
#             ax3.set_xticks(x_pos)
#             ax3.set_xticklabels(difficulties, fontsize=8)
#             ax3.tick_params(axis='y', labelsize=8)
#             ax3.legend(fontsize=9)
#             ax3.grid(True, alpha=0.3)
        
#         # Plot 4: Success Rates by Difficulty
#         if final_evaluation["PPO"] and final_evaluation["DDPG"]:
#             ax4 = fig.add_subplot(gs[1, 0])
#             ppo_success = [final_evaluation["PPO"][diff]['success_rate'] for diff in difficulties]
#             ddpg_success = [final_evaluation["DDPG"][diff]['success_rate'] for diff in difficulties]
            
#             ax4.bar(x_pos - width/2, ppo_success, width, label='Best PPO', color='skyblue', alpha=0.8)
#             ax4.bar(x_pos + width/2, ddpg_success, width, label='Best DDPG', color='lightcoral', alpha=0.8)
#             ax4.set_xlabel('Difficulty Level', fontsize=10)
#             ax4.set_ylabel('Success Rate (%)', fontsize=10)
#             ax4.set_title('Goal Scoring Success Rate', fontweight='bold', fontsize=12)
#             ax4.set_xticks(x_pos)
#             ax4.set_xticklabels(difficulties, fontsize=8)
#             ax4.tick_params(axis='y', labelsize=8)
#             ax4.legend(fontsize=9)
#             ax4.grid(True, alpha=0.3)
        
#         # Plot 5: Training Summary Text
#         ax5 = fig.add_subplot(gs[1, 1])
#         ax5.axis('off')
#         summary_text = "TRAINING SUMMARY:\n\n"
#         summary_text += f"PPO variants trained: {len(ppo_results)}\n"
#         summary_text += f"DDPG variants trained: {len(ddpg_results)}\n"
#         summary_text += f"Total training time: {(time.time() - training_system.training_start_time)/3600:.1f} hours\n"
#         summary_text += f"Best PPO score: {training_system.best_ppo_score:.2f}\n"
#         summary_text += f"Best DDPG score: {training_system.best_ddpg_score:.2f}\n"
        
#         if training_system.best_ppo_score > training_system.best_ddpg_score:
#             summary_text += f"\n OVERALL WINNER: PPO\n"
#         elif training_system.best_ddpg_score > training_system.best_ppo_score:
#             summary_text += f"\n OVERALL WINNER: DDPG\n"
#         else:
#             summary_text += f"\n TIE!\n"
        
#         ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
#                 verticalalignment='top', fontfamily='monospace',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
#         # Plot 6: Best Models Details
#         ax6 = fig.add_subplot(gs[1, 2])
#         ax6.axis('off')
        
#         models_text = "BEST MODELS FOUND:\n\n"
        
#         if ppo_results:
#             best_ppo = max(ppo_results, key=lambda x: x[1])
#             models_text += f"PPO Winner:\n"
#             models_text += f"  Variant: {best_ppo[2]}\n"
#             models_text += f"  Score: {best_ppo[1]:.2f}\n"
#             models_text += f"  Path: {os.path.basename(best_ppo[0]) if best_ppo[0] else 'None'}\n\n"
        
#         if ddpg_results:
#             best_ddpg = max(ddpg_results, key=lambda x: x[1])
#             models_text += f"DDPG Winner:\n"
#             models_text += f"  Variant: {best_ddpg[2]}\n"
#             models_text += f"  Score: {best_ddpg[1]:.2f}\n"
#             models_text += f"  Path: {os.path.basename(best_ddpg[0]) if best_ddpg[0] else 'None'}\n"
        
#         ax6.text(0.05, 0.95, models_text, transform=ax6.transAxes, fontsize=9,
#                 verticalalignment='top', fontfamily='monospace',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
#         # Plot 7-9: Individual variant performance distributions
#         if len(ppo_results) > 1:
#             ax7 = fig.add_subplot(gs[2, :])
#             ppo_variant_names = [r[2] for r in ppo_results]
#             ppo_variant_scores = [r[1] for r in ppo_results]
#             ddpg_variant_names = [r[2] for r in ddpg_results] 
#             ddpg_variant_scores = [r[1] for r in ddpg_results]
            
#             # Combined comparison of all variants
#             all_variants = [(f"PPO-{name}", score) for name, score in zip(ppo_variant_names, ppo_variant_scores)]
#             all_variants += [(f"DDPG-{name}", score) for name, score in zip(ddpg_variant_names, ddpg_variant_scores)]
            
#             # Sort by performance
#             all_variants.sort(key=lambda x: x[1], reverse=True)
            
#             names = [v[0] for v in all_variants]
#             scores = [v[1] for v in all_variants]
#             colors = ['skyblue' if 'PPO' in name else 'lightcoral' for name in names]
            
#             bars = ax7.barh(names, scores, color=colors, alpha=0.8)
#             ax7.set_xlabel('Evaluation Score', fontsize=10)
#             ax7.set_title('All Model Variants Performance Ranking', fontweight='bold', fontsize=12)
#             ax7.tick_params(axis='both', labelsize=8)
#             ax7.grid(True, alpha=0.3)
            
#             # Highlight top performer
#             if bars:
#                 bars[0].set_color('gold')
#                 bars[0].set_edgecolor('orange')
#                 bars[0].set_linewidth(2)
        
#         # Plot 10: Hyperparameter insights
#         ax8 = fig.add_subplot(gs[3, :2])
#         ax8.axis('off')
        
#         insights_text = "HYPERPARAMETER INSIGHTS:\n\n"
        
#         if ppo_results:
#             best_ppo = max(ppo_results, key=lambda x: x[1])
#             worst_ppo = min(ppo_results, key=lambda x: x[1])
            
#             insights_text += f"PPO Analysis:\n"
#             insights_text += f"  Best variant: {best_ppo[2]} (Score: {best_ppo[1]:.2f})\n"
#             insights_text += f"  Worst variant: {worst_ppo[2]} (Score: {worst_ppo[1]:.2f})\n"
#             insights_text += f"  Performance range: {best_ppo[1] - worst_ppo[1]:.2f}\n\n"
            
#         if ddpg_results:
#             best_ddpg = max(ddpg_results, key=lambda x: x[1])
#             worst_ddpg = min(ddpg_results, key=lambda x: x[1])
            
#             insights_text += f"DDPG Analysis:\n"
#             insights_text += f"  Best variant: {best_ddpg[2]} (Score: {best_ddpg[1]:.2f})\n"
#             insights_text += f"  Worst variant: {worst_ddpg[2]} (Score: {worst_ddpg[1]:.2f})\n"
#             insights_text += f"  Performance range: {best_ddpg[1] - worst_ddpg[1]:.2f}\n"
        
#         ax8.text(0.05, 0.95, insights_text, transform=ax8.transAxes, fontsize=9,
#                 verticalalignment='top', fontfamily='monospace',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
#         # Plot 11: Resource usage summary
#         ax9 = fig.add_subplot(gs[3, 2])
#         ax9.axis('off')
        
#         resource_text = "RESOURCE USAGE:\n\n"
#         total_hours = (time.time() - training_system.training_start_time) / 3600
#         resource_text += f"Total time: {total_hours:.1f} hours\n"
#         resource_text += f"Models trained: {len(ppo_results) + len(ddpg_results)}\n"
#         resource_text += f"Avg time per model: {total_hours/(len(ppo_results) + len(ddpg_results)):.1f}h\n"
#         resource_text += f"Device used: {'GPU' if torch.cuda.is_available() else 'CPU'}\n"
#         resource_text += f"Total models saved: {len([r for r in ppo_results + ddpg_results if r[0]])}\n"
        
#         ax9.text(0.05, 0.95, resource_text, transform=ax9.transAxes, fontsize=9,
#                 verticalalignment='top', fontfamily='monospace',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
#         # Use constrained layout instead of tight_layout for better spacing
#         plt.subplots_adjust(hspace=0.5, wspace=0.4)
#         plot_path = f"{training_system.output_dir}/plots/comprehensive_comparison.png"
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
#         plt.show()
        
#         training_system.logger.info(f"Comprehensive comparison plots saved to {plot_path}")

#     except Exception as e:
#         training_system.logger.error(f"Error creating plots: {e}")


def create_academic_training_curves(training_system, algorithm_name, model_path, evaluation_data=None):
    """
    Create publication-quality training curve plots for academic reports.

    This function generates comprehensive training analysis plots following
    academic standards for reinforcement learning research.

    References:
    - "Deep Reinforcement Learning: An Overview" (Li, 2017)
    - "Empirical Methodology for RL" (Henderson et al., 2018)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import pandas as pd

    # Set academic plotting style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12

    try:
        # Use provided evaluation data (preferred) or load from evaluations.npz
        if evaluation_data is not None:
            training_data = evaluation_data
        else:
            # Fallback: try to load from log directory
            training_data = load_training_data(model_path)

        if not training_data:
            training_system.logger.warning(f"No training data found for {algorithm_name}")
            return None

        # Create figure with academic layout - improved spacing
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{algorithm_name} Training Analysis - Soccer RL Environment',
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust subplot spacing with more space for suptitle
        plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.88, bottom=0.08)

        # Plot 1: Reward Learning Curve with Confidence Intervals
        timesteps = training_data.get('timesteps', [])
        rewards = training_data.get('rewards', [])

        if len(timesteps) > 0 and len(rewards) > 0:
            # Calculate moving average and confidence intervals
            window_size = len(rewards) // 50  # Smooth over 2% of data
            if window_size < 10:
                window_size = 10

            # Calculate the moving average and standard deviation of the rewards
            # to create a smoothed learning curve with confidence intervals
            rewards_smooth = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            rewards_std = pd.Series(rewards).rolling(window=window_size, center=True).std()

            # Convert timesteps to millions for readability
            timesteps_m = [t / 1e6 for t in timesteps]

            # Plot smoothed learning curve with confidence intervals
            ax1.plot(timesteps_m, rewards_smooth, linewidth=2.5, alpha=0.9,
                    label=f'{algorithm_name} Mean Reward', color='navy')
            ax1.fill_between(timesteps_m,
                           rewards_smooth - rewards_std,
                           rewards_smooth + rewards_std,
                           alpha=0.3, color='navy', label='±1 Standard Deviation')

            # Add raw data as scatter for transparency
            sample_indices = np.linspace(0, len(rewards)-1, min(1000, len(rewards)), dtype=int)
            ax1.scatter([timesteps_m[i] for i in sample_indices],
                       [rewards[i] for i in sample_indices],
                       alpha=0.1, s=1, color='darkblue')

            ax1.set_xlabel('Training Steps (Millions)', fontsize=14)
            ax1.set_ylabel('Episode Reward', fontsize=14)
            ax1.set_title('Learning Curve with Confidence Intervals', fontsize=16, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)

            # Add convergence analysis
            final_performance = np.mean(rewards[-min(100, len(rewards)//10):])
            ax1.axhline(y=final_performance, color='red', linestyle='--', alpha=0.7,
                       label=f'Final Performance: {final_performance:.2f}')

        # Plot 2: Training Stability Analysis
        if 'episode_lengths' in training_data:
            ep_lengths = training_data['episode_lengths']
            ax2.hist(ep_lengths, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
            ax2.set_xlabel('Episode Length (Steps)', fontsize=14)
            ax2.set_ylabel('Frequency', fontsize=14)
            ax2.set_title('Episode Length Distribution', fontsize=16, fontweight='bold')
            ax2.axvline(np.mean(ep_lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(ep_lengths):.1f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Reward Components Timeline (if available)
        if 'reward_components' in training_data:
            components = training_data['reward_components']
            component_names = list(components.keys())
            
            # Create a timeline plot showing how reward components evolve
            colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))
            
            for i, (comp_name, comp_values) in enumerate(components.items()):
                # Smooth the component values over time
                if len(comp_values) > 20:
                    window = max(10, len(comp_values) // 50)
                    smoothed = pd.Series(comp_values).rolling(window=window, center=True).mean()
                    episodes = range(len(comp_values))
                    
                    ax3.plot(episodes, smoothed, linewidth=2.5, alpha=0.8,
                           color=colors[i], label=f'{comp_name.replace("_", " ").title()}')
            
            ax3.set_xlabel('Episode Number', fontsize=14)
            ax3.set_ylabel('Reward Component Value', fontsize=14)
            ax3.set_title('Reward Components Evolution', fontsize=16, fontweight='bold')
            ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
            
        else:
            # If no reward components, show episode length over time
            if 'episode_lengths' in training_data:
                ep_lengths = training_data['episode_lengths']
                episodes = range(len(ep_lengths))
                
                # Smooth episode lengths
                window = max(10, len(ep_lengths) // 50)
                ep_smooth = pd.Series(ep_lengths).rolling(window=window, center=True).mean()
                
                ax3.plot(episodes, ep_smooth, linewidth=2.5, color='darkgreen', alpha=0.8)
                ax3.axhline(np.mean(ep_lengths), color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {np.mean(ep_lengths):.1f}')
                
                ax3.set_xlabel('Episode Number', fontsize=14)
                ax3.set_ylabel('Episode Length (Steps)', fontsize=14)
                ax3.set_title('Episode Length Progression', fontsize=16, fontweight='bold')
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3)

        # Plot 4: Performance Metrics Summary
        ax4.axis('off')

        # Calculate key statistics
        if len(rewards) > 0:
            stats_text = "PERFORMANCE STATISTICS\n" + "="*30 + "\n\n"
            stats_text += f"Algorithm: {algorithm_name}\n"
            stats_text += f"Total Training Steps: {max(timesteps):,}\n"
            stats_text += f"Total Episodes: {len(rewards):,}\n\n"

            # Performance metrics
            stats_text += "REWARD ANALYSIS:\n"
            stats_text += f"Final Mean Reward: {final_performance:.3f}\n"
            stats_text += f"Best Episode Reward: {max(rewards):.3f}\n"
            stats_text += f"Worst Episode Reward: {min(rewards):.3f}\n"
            stats_text += f"Reward Standard Deviation: {np.std(rewards):.3f}\n\n"

            # Convergence analysis
            early_mean = np.mean(rewards[:len(rewards)//4])
            late_mean = np.mean(rewards[-len(rewards)//4:])
            improvement = ((late_mean - early_mean) / abs(early_mean)) * 100 if early_mean != 0 else 0

            stats_text += "LEARNING PROGRESS:\n"
            stats_text += f"Early Performance: {early_mean:.3f}\n"
            stats_text += f"Late Performance: {late_mean:.3f}\n"
            stats_text += f"Improvement: {improvement:+.1f}%\n\n"

            # Statistical significance test
            early_rewards = rewards[:len(rewards)//4]
            late_rewards = rewards[-len(rewards)//4:]
            t_stat, p_value = stats.ttest_ind(late_rewards, early_rewards)

            stats_text += "STATISTICAL ANALYSIS:\n"
            stats_text += f"t-statistic: {t_stat:.3f}\n"
            stats_text += f"p-value: {p_value:.6f}\n"
            if p_value < 0.05:
                stats_text += "Significant Learning (p < 0.05)\n"
            else:
                stats_text += "No Significant Learning (p >= 0.05)\n"

        ax4.text(0.15, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8))

        # Save with academic naming convention
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = f"{training_system.output_dir}/plots/{algorithm_name}_training_analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        training_system.logger.info(f"Academic training curves saved to {plot_path}")
        return plot_path

    except Exception as e:
        training_system.logger.error(f"Error creating academic training curves: {e}")
        return None

def create_algorithm_comparison_plot(training_system, ppo_data, ddpg_data, title="Algorithm Comparison"):
    """
    Create publication-quality comparison plot between PPO and DDPG.

    This generates a comprehensive academic-style comparison with:
    - Reward progression scatter plots
    - Statistical confidence intervals
    - Convergence analysis
    - Performance benchmarking
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats

    # Academic styling
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    try:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 14))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Improve subplot spacing for 2x3 layout with more space for suptitle
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88, bottom=0.08)

        # Plot 1: Training Curves Comparison
        ppo_rewards = []
        ddpg_rewards = []
        ppo_timesteps = []
        ddpg_timesteps = []

        if ppo_data and 'timesteps' in ppo_data and 'rewards' in ppo_data:
            if len(ppo_data['timesteps']) > 0 and len(ppo_data['rewards']) > 0:
                ppo_timesteps = [t/1e6 for t in ppo_data['timesteps']]  # Convert to millions
                ppo_rewards = ppo_data['rewards']

                # Smooth PPO curve
                window = len(ppo_rewards) // 50
                if window < 10:
                    window = 10
                ppo_smooth = pd.Series(ppo_rewards).rolling(window=window, center=True).mean()
                ppo_std = pd.Series(ppo_rewards).rolling(window=window, center=True).std()

                ax1.plot(ppo_timesteps, ppo_smooth, linewidth=3, color='#2E86C1',
                        label='PPO', alpha=0.9)
                ax1.fill_between(ppo_timesteps, ppo_smooth - ppo_std, ppo_smooth + ppo_std,
                               alpha=0.3, color='#2E86C1')

        if ddpg_data and 'timesteps' in ddpg_data and 'rewards' in ddpg_data:
            if len(ddpg_data['timesteps']) > 0 and len(ddpg_data['rewards']) > 0:
                ddpg_timesteps = [t/1e6 for t in ddpg_data['timesteps']]  # Convert to millions
                ddpg_rewards = ddpg_data['rewards']

                # Smooth DDPG curve
                window = len(ddpg_rewards) // 50
                if window < 10:
                    window = 10
                ddpg_smooth = pd.Series(ddpg_rewards).rolling(window=window, center=True).mean()
                ddpg_std = pd.Series(ddpg_rewards).rolling(window=window, center=True).std()

                ax1.plot(ddpg_timesteps, ddpg_smooth, linewidth=3, color='#E74C3C',
                        label='DDPG', alpha=0.9)
                ax1.fill_between(ddpg_timesteps, ddpg_smooth - ddpg_std, ddpg_smooth + ddpg_std,
                               alpha=0.3, color='#E74C3C')

        ax1.set_xlabel('Training Steps (Millions)', fontsize=14)
        ax1.set_ylabel('Episode Reward', fontsize=14)
        ax1.set_title('Learning Curves Comparison', fontsize=16, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Episode Length Comparison
        if ppo_data and ddpg_data and 'episode_lengths' in ppo_data and 'episode_lengths' in ddpg_data:
            ppo_episode_lengths = ppo_data['episode_lengths']
            ddpg_episode_lengths = ddpg_data['episode_lengths']
            
            # Calculate rolling averages for episode lengths
            window = len(ppo_episode_lengths) // 50
            if window < 10:
                window = 10
            
            ppo_ep_smooth = pd.Series(ppo_episode_lengths).rolling(window=window, center=True).mean()
            ddpg_ep_smooth = pd.Series(ddpg_episode_lengths).rolling(window=window, center=True).mean()
            
            ppo_timesteps_ep = [t/1e6 for t in ppo_data['timesteps'][:len(ppo_episode_lengths)]]
            ddpg_timesteps_ep = [t/1e6 for t in ddpg_data['timesteps'][:len(ddpg_episode_lengths)]]
            
            ax2.plot(ppo_timesteps_ep, ppo_ep_smooth, linewidth=3, color='#2E86C1',
                    label='PPO', alpha=0.9)
            ax2.plot(ddpg_timesteps_ep, ddpg_ep_smooth, linewidth=3, color='#E74C3C',
                    label='DDPG', alpha=0.9)
            
            ax2.set_xlabel('Training Steps (Millions)', fontsize=14)
            ax2.set_ylabel('Episode Length (Steps)', fontsize=14)
            ax2.set_title('Episode Length Comparison', fontsize=16, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=12)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Reward Distribution Comparison
        if ppo_data and ddpg_data:
            # Final 25% of training for comparison
            ppo_final = ppo_rewards[-len(ppo_rewards)//4:] if len(ppo_rewards) > 0 else []
            ddpg_final = ddpg_rewards[-len(ddpg_rewards)//4:] if len(ddpg_rewards) > 0 else []

            if len(ppo_final) > 0 and len(ddpg_final) > 0:
                ax3.hist(ppo_final, bins=30, alpha=0.7, color='#2E86C1',
                        label=f'PPO (μ={np.mean(ppo_final):.2f})', density=True)
                ax3.hist(ddpg_final, bins=30, alpha=0.7, color='#E74C3C',
                        label=f'DDPG (μ={np.mean(ddpg_final):.2f})', density=True)

                ax3.axvline(np.mean(ppo_final), color='#1B4F72', linestyle='--', linewidth=2)
                ax3.axvline(np.mean(ddpg_final), color='#922B21', linestyle='--', linewidth=2)

                ax3.set_xlabel('Episode Reward', fontsize=14)
                ax3.set_ylabel('Density', fontsize=14)
                ax3.set_title('Final Performance Distribution', fontsize=16, fontweight='bold')
                ax3.legend(fontsize=12)
                ax3.grid(True, alpha=0.3)

        # Plot 4: Convergence Analysis
        if ppo_data and ddpg_data and len(ppo_rewards) > 0 and len(ddpg_rewards) > 0:
            # Sample efficiency comparison
            reward_thresholds = np.linspace(min(min(ppo_rewards), min(ddpg_rewards)),
                                          max(max(ppo_rewards), max(ddpg_rewards)), 20)

            ppo_convergence = []
            ddpg_convergence = []

            for threshold in reward_thresholds:
                # Find first time each algorithm reaches threshold
                ppo_idx = next((i for i, r in enumerate(ppo_rewards) if r >= threshold), len(ppo_rewards))
                ddpg_idx = next((i for i, r in enumerate(ddpg_rewards) if r >= threshold), len(ddpg_rewards))

                ppo_convergence.append(ppo_timesteps[min(ppo_idx, len(ppo_timesteps)-1)] if ppo_idx < len(ppo_timesteps) else float('inf'))
                ddpg_convergence.append(ddpg_timesteps[min(ddpg_idx, len(ddpg_timesteps)-1)] if ddpg_idx < len(ddpg_timesteps) else float('inf'))

            ax4.plot(reward_thresholds, ppo_convergence, 'o-', color='#2E86C1',
                    label='PPO', linewidth=2, markersize=6)
            ax4.plot(reward_thresholds, ddpg_convergence, 's-', color='#E74C3C',
                    label='DDPG', linewidth=2, markersize=6)

            ax4.set_xlabel('Reward Threshold', fontsize=14)
            ax4.set_ylabel('Steps to Convergence (Millions)', fontsize=14)
            ax4.set_title('Sample Efficiency Comparison', fontsize=16, fontweight='bold')
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3)

        # Plot 5: Statistical Summary
        ax5.axis('off')

        if ppo_data and ddpg_data and len(ppo_rewards) > 0 and len(ddpg_rewards) > 0:
            ppo_final = ppo_rewards[-len(ppo_rewards)//4:]
            ddpg_final = ddpg_rewards[-len(ddpg_rewards)//4:]

            # Perform statistical tests
            t_stat, p_value = stats.ttest_ind(ppo_final, ddpg_final)
            effect_size = (np.mean(ppo_final) - np.mean(ddpg_final)) / np.sqrt((np.var(ppo_final) + np.var(ddpg_final))/2)

            summary_text = "STATISTICAL COMPARISON\n" + "="*30 + "\n\n"
            summary_text += f"PPO Performance:\n"
            summary_text += f"  Final Mean: {np.mean(ppo_final):.3f} ± {np.std(ppo_final):.3f}\n"
            summary_text += f"  Best Episode: {max(ppo_rewards):.3f}\n"
            summary_text += f"  Episodes: {len(ppo_rewards):,}\n\n"

            summary_text += f"DDPG Performance:\n"
            summary_text += f"  Final Mean: {np.mean(ddpg_final):.3f} ± {np.std(ddpg_final):.3f}\n"
            summary_text += f"  Best Episode: {max(ddpg_rewards):.3f}\n"
            summary_text += f"  Episodes: {len(ddpg_rewards):,}\n\n"

            summary_text += f"STATISTICAL TESTS:\n"
            summary_text += f"  t-statistic: {t_stat:.3f}\n"
            summary_text += f"  p-value: {p_value:.6f}\n"
            summary_text += f"  Effect size (Cohen's d): {effect_size:.3f}\n\n"

            # Determine winner
            if p_value < 0.05:
                if np.mean(ppo_final) > np.mean(ddpg_final):
                    summary_text += "🏆 PPO significantly outperforms DDPG\n"
                else:
                    summary_text += "🏆 DDPG significantly outperforms PPO\n"
            else:
                summary_text += "⚖️ No significant difference between algorithms\n"

            # Effect size interpretation
            if abs(effect_size) < 0.2:
                summary_text += "Effect size: Small\n"
            elif abs(effect_size) < 0.8:
                summary_text += "Effect size: Medium\n"
            else:
                summary_text += "Effect size: Large\n"

        ax5.text(0.02, 0.98, summary_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))

        # Plot 6: Enhanced Metrics Visualization
        if ppo_data and ddpg_data and 'enhanced_metrics' in ppo_data and 'enhanced_metrics' in ddpg_data:
            ppo_metrics = ppo_data['enhanced_metrics']
            ddpg_metrics = ddpg_data['enhanced_metrics']
            
            # Create radar chart for various metrics
            metrics_names = ['Goals\nScored', 'Ball\nPossession', 'Collision\nAvoidance', 
                           'Boundary\nKeeping', 'Action\nEfficiency']
            
            # Normalize metrics to 0-1 scale for radar chart
            ppo_values = [
                np.mean(ppo_metrics['goals_scored']) / 10,  # Goals per 10 episodes
                np.mean(ppo_metrics['possession_time']) / 100,  # Possession as fraction
                1 - np.mean(ppo_metrics['collision_frequency']) / 50,  # Invert collisions
                1 - np.mean(ppo_metrics['out_of_bounds']) / 20,  # Invert out of bounds
                np.mean(ppo_metrics['action_efficiency'])  # Already 0-1
            ]
            
            ddpg_values = [
                np.mean(ddpg_metrics['goals_scored']) / 10,
                np.mean(ddpg_metrics['possession_time']) / 100,
                1 - np.mean(ddpg_metrics['collision_frequency']) / 50,
                1 - np.mean(ddpg_metrics['out_of_bounds']) / 20,
                np.mean(ddpg_metrics['action_efficiency'])
            ]
            
            # Create polar plot
            angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ppo_values += ppo_values[:1]  # Complete the circle
            ddpg_values += ddpg_values[:1]  # Complete the circle
            
            # Convert ax6 to polar
            ax6.remove()
            ax6 = fig.add_subplot(2, 3, 6, projection='polar')
            
            ax6.plot(angles, ppo_values, 'o-', linewidth=2, label='PPO', color='#2E86C1')
            ax6.fill(angles, ppo_values, alpha=0.25, color='#2E86C1')
            ax6.plot(angles, ddpg_values, 's-', linewidth=2, label='DDPG', color='#E74C3C')
            ax6.fill(angles, ddpg_values, alpha=0.25, color='#E74C3C')
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(metrics_names, fontsize=10)
            ax6.set_ylim(0, 1)
            ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax6.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
            ax6.set_title('Performance Metrics\nComparison', fontsize=12, fontweight='bold', pad=20)
            ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
            ax6.grid(True, alpha=0.3)
        else:
            # If no enhanced metrics, show episode length distribution
            ax6.axis('off')
            info_text = "EPISODE LENGTH ANALYSIS\n" + "="*25 + "\n\n"
            
            if ppo_data and ddpg_data and 'episode_lengths' in ppo_data and 'episode_lengths' in ddpg_data:
                ppo_ep_lens = ppo_data['episode_lengths']
                ddpg_ep_lens = ddpg_data['episode_lengths']
                
                info_text += f"PPO Episode Lengths:\n"
                info_text += f"  Mean: {np.mean(ppo_ep_lens):.1f} steps\n"
                info_text += f"  Std: {np.std(ppo_ep_lens):.1f} steps\n"
                info_text += f"  Range: {np.min(ppo_ep_lens):.0f}-{np.max(ppo_ep_lens):.0f}\n\n"
                
                info_text += f"DDPG Episode Lengths:\n"
                info_text += f"  Mean: {np.mean(ddpg_ep_lens):.1f} steps\n"
                info_text += f"  Std: {np.std(ddpg_ep_lens):.1f} steps\n"
                info_text += f"  Range: {np.min(ddpg_ep_lens):.0f}-{np.max(ddpg_ep_lens):.0f}\n\n"
                
                # Statistical comparison
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(ppo_ep_lens, ddpg_ep_lens)
                info_text += f"Statistical Test:\n"
                info_text += f"  t-statistic: {t_stat:.3f}\n"
                info_text += f"  p-value: {p_val:.4f}\n"
                if p_val < 0.05:
                    winner = "PPO" if np.mean(ppo_ep_lens) > np.mean(ddpg_ep_lens) else "DDPG"
                    info_text += f"  Winner: {winner}\n"
            
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.8))

        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = f"{training_system.output_dir}/plots/algorithm_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        training_system.logger.info(f"Algorithm comparison plot saved to {plot_path}")
        return plot_path

    except Exception as e:
        training_system.logger.error(f"Error creating comparison plot: {e}")
        return None


def create_enhanced_metrics_comparison(training_system, ppo_data, ddpg_data, title="Enhanced Metrics Comparison"):
    """
    Create advanced metrics comparison plot with soccer-specific metrics.
    
    This function visualizes key performance indicators beyond just rewards:
    - Goals scored progression
    - Ball possession time trends
    - Collision frequency analysis
    - Out-of-bounds incidents
    - Ball contact efficiency
    - Final goal distance trends
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats

    # Academic styling
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 11

    try:
        # Create a comprehensive figure with 6 subplots (2x3)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Improve subplot spacing with more space for suptitle
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.88, bottom=0.08)

        # Check if enhanced metrics are available
        ppo_metrics = ppo_data.get('enhanced_metrics', {})
        ddpg_metrics = ddpg_data.get('enhanced_metrics', {})
        
        if not ppo_metrics or not ddpg_metrics:
            training_system.logger.warning("Enhanced metrics not available, using basic metrics")
            return None

        # Plot 1: Goals Scored Over Time (Cumulative Success Rate)
        if 'goals_scored' in ppo_metrics and 'goals_scored' in ddpg_metrics:
            ppo_goals_cum = np.cumsum(ppo_metrics['goals_scored'])
            ddpg_goals_cum = np.cumsum(ddpg_metrics['goals_scored'])
            
            ppo_episodes = range(len(ppo_goals_cum))
            ddpg_episodes = range(len(ddpg_goals_cum))
            
            ax1.plot(ppo_episodes, ppo_goals_cum, linewidth=2.5, color='#2E86C1', 
                    label='PPO', alpha=0.9)
            ax1.plot(ddpg_episodes, ddpg_goals_cum, linewidth=2.5, color='#E74C3C', 
                    label='DDPG', alpha=0.9)
            
            ax1.set_xlabel('Episode Number', fontsize=12)
            ax1.set_ylabel('Cumulative Goals Scored', fontsize=12)
            ax1.set_title('Goal Scoring Progress', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

        # Plot 2: Ball Possession Time Trends
        if 'possession_time_pct' in ppo_metrics and 'possession_time_pct' in ddpg_metrics:
            # Smooth possession data
            window_ppo = max(10, len(ppo_metrics['possession_time_pct']) // 50)
            window_ddpg = max(10, len(ddpg_metrics['possession_time_pct']) // 50)
            
            ppo_poss_smooth = pd.Series(ppo_metrics['possession_time_pct']).rolling(window=window_ppo, center=True).mean()
            ddpg_poss_smooth = pd.Series(ddpg_metrics['possession_time_pct']).rolling(window=window_ddpg, center=True).mean()
            
            ax2.plot(range(len(ppo_poss_smooth)), ppo_poss_smooth, linewidth=2.5, 
                    color='#2E86C1', label='PPO', alpha=0.9)
            ax2.plot(range(len(ddpg_poss_smooth)), ddpg_poss_smooth, linewidth=2.5, 
                    color='#E74C3C', label='DDPG', alpha=0.9)
            
            # Add target line at 60% (good possession)
            ax2.axhline(60, color='green', linestyle='--', alpha=0.7, label='Target: 60%')
            
            ax2.set_xlabel('Episode Number', fontsize=12)
            ax2.set_ylabel('Ball Possession (%)', fontsize=12)
            ax2.set_title('Ball Possession Efficiency', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Collision Frequency Trends (Learning Efficiency)
        if 'collision_count' in ppo_metrics and 'collision_count' in ddpg_metrics:
            # Smooth collision data
            window_ppo = max(10, len(ppo_metrics['collision_count']) // 50)
            window_ddpg = max(10, len(ddpg_metrics['collision_count']) // 50)
            
            ppo_coll_smooth = pd.Series(ppo_metrics['collision_count']).rolling(window=window_ppo, center=True).mean()
            ddpg_coll_smooth = pd.Series(ddpg_metrics['collision_count']).rolling(window=window_ddpg, center=True).mean()
            
            ax3.plot(range(len(ppo_coll_smooth)), ppo_coll_smooth, linewidth=2.5, 
                    color='#2E86C1', label='PPO', alpha=0.9)
            ax3.plot(range(len(ddpg_coll_smooth)), ddpg_coll_smooth, linewidth=2.5, 
                    color='#E74C3C', label='DDPG', alpha=0.9)
            
            ax3.set_xlabel('Episode Number', fontsize=12)
            ax3.set_ylabel('Collisions per Episode', fontsize=12)
            ax3.set_title('Collision Frequency (Lower is Better)', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)

        # Plot 4: Out of Bounds Analysis - Line Chart Over Time
        if 'out_of_bounds_count' in ppo_metrics and 'out_of_bounds_count' in ddpg_metrics:
            ppo_oob = ppo_metrics['out_of_bounds_count']
            ddpg_oob = ddpg_metrics['out_of_bounds_count']

            # Smooth the data for clearer trends
            window_ppo = max(10, len(ppo_oob) // 50)
            window_ddpg = max(10, len(ddpg_oob) // 50)

            ppo_oob_smooth = pd.Series(ppo_oob).rolling(window=window_ppo, center=True).mean()
            ddpg_oob_smooth = pd.Series(ddpg_oob).rolling(window=window_ddpg, center=True).mean()

            # Plot line charts showing improvement over time
            ax4.plot(range(len(ppo_oob_smooth)), ppo_oob_smooth, linewidth=2.5,
                    color='#2E86C1', label='PPO', alpha=0.9)
            ax4.plot(range(len(ddpg_oob_smooth)), ddpg_oob_smooth, linewidth=2.5,
                    color='#E74C3C', label='DDPG', alpha=0.9)

            ax4.set_xlabel('Episode Number', fontsize=12)
            ax4.set_ylabel('Out of Bounds Count per Episode', fontsize=12)
            ax4.set_title('Ball Control Quality Over Time (Lower is Better)', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3)

        # Plot 5: Final Ball Distance Over Time (Goal Approach Quality)
        if 'final_ball_distance' in ppo_metrics and 'final_ball_distance' in ddpg_metrics:
            ppo_dist = ppo_metrics['final_ball_distance']
            ddpg_dist = ddpg_metrics['final_ball_distance']

            # Smooth the data for clearer trends
            window_ppo = max(10, len(ppo_dist) // 50)
            window_ddpg = max(10, len(ddpg_dist) // 50)

            ppo_dist_smooth = pd.Series(ppo_dist).rolling(window=window_ppo, center=True).mean()
            ddpg_dist_smooth = pd.Series(ddpg_dist).rolling(window=window_ddpg, center=True).mean()

            # Plot line charts showing distance to goal improvement
            ax5.plot(range(len(ppo_dist_smooth)), ppo_dist_smooth, linewidth=2.5,
                    color='#2E86C1', label='PPO', alpha=0.9)
            ax5.plot(range(len(ddpg_dist_smooth)), ddpg_dist_smooth, linewidth=2.5,
                    color='#E74C3C', label='DDPG', alpha=0.9)

            # Add reference line for "good" distance (e.g., 2 meters)
            ax5.axhline(2.0, color='green', linestyle='--', alpha=0.7, label='Target: 2m')

            ax5.set_xlabel('Episode Number', fontsize=12)
            ax5.set_ylabel('Final Ball Distance (meters)', fontsize=12)
            ax5.set_title('Goal Approach Quality Over Time (Lower is Better)', fontsize=14, fontweight='bold')
            ax5.legend(fontsize=11)
            ax5.grid(True, alpha=0.3)

        # Plot 6: Performance Summary Statistics
        ax6.axis('off')
        
        # Calculate summary statistics
        summary_text = "ENHANCED METRICS SUMMARY\n" + "="*35 + "\n\n"
        
        if ppo_metrics and ddpg_metrics:
            # Goals comparison
            ppo_total_goals = sum(ppo_metrics.get('goals_scored', [0]))
            ddpg_total_goals = sum(ddpg_metrics.get('goals_scored', [0]))
            
            summary_text += f"GOAL SCORING:\n"
            summary_text += f"  PPO Total Goals: {ppo_total_goals}\n"
            summary_text += f"  DDPG Total Goals: {ddpg_total_goals}\n"
            summary_text += f"  PPO Success Rate: {(ppo_total_goals/len(ppo_metrics.get('goals_scored', [1])))*100:.1f}%\n"
            summary_text += f"  DDPG Success Rate: {(ddpg_total_goals/len(ddpg_metrics.get('goals_scored', [1])))*100:.1f}%\n\n"
            
            # Possession comparison
            ppo_avg_poss = np.mean(ppo_metrics.get('possession_time_pct', [0]))
            ddpg_avg_poss = np.mean(ddpg_metrics.get('possession_time_pct', [0]))
            
            summary_text += f"BALL POSSESSION:\n"
            summary_text += f"  PPO Average: {ppo_avg_poss:.1f}%\n"
            summary_text += f"  DDPG Average: {ddpg_avg_poss:.1f}%\n\n"
            
            # Collision analysis
            ppo_avg_coll = np.mean(ppo_metrics.get('collision_count', [0]))
            ddpg_avg_coll = np.mean(ddpg_metrics.get('collision_count', [0]))
            
            summary_text += f"COLLISION ANALYSIS:\n"
            summary_text += f"  PPO Avg/Episode: {ppo_avg_coll:.1f}\n"
            summary_text += f"  DDPG Avg/Episode: {ddpg_avg_coll:.1f}\n\n"
            
            # Ball control
            ppo_avg_contact = np.mean(ppo_metrics.get('ball_contact_time_pct', [0]))
            ddpg_avg_contact = np.mean(ddpg_metrics.get('ball_contact_time_pct', [0]))
            
            summary_text += f"BALL CONTROL:\n"
            summary_text += f"  PPO Contact Time: {ppo_avg_contact:.1f}%\n"
            summary_text += f"  DDPG Contact Time: {ddpg_avg_contact:.1f}%\n\n"
            
            # Determine better algorithm
            ppo_score = (ppo_total_goals * 2) + (ppo_avg_poss / 10) - (ppo_avg_coll * 0.5)
            ddpg_score = (ddpg_total_goals * 2) + (ddpg_avg_poss / 10) - (ddpg_avg_coll * 0.5)

            summary_text += "OVERALL PERFORMANCE:\n"
            if ppo_score > ddpg_score:
                summary_text += "  PPO shows better performance\n"
                summary_text += f"  (Score: PPO={ppo_score:.1f}, DDPG={ddpg_score:.1f})\n"
            elif ddpg_score > ppo_score:
                summary_text += "  DDPG shows better performance\n"
                summary_text += f"  (Score: PPO={ppo_score:.1f}, DDPG={ddpg_score:.1f})\n"
            else:
                summary_text += "  Comparable performance\n"
                summary_text += f"  (Score: PPO={ppo_score:.1f}, DDPG={ddpg_score:.1f})\n"

        ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.8))

        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = f"{training_system.output_dir}/plots/enhanced_metrics_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        training_system.logger.info(f"Enhanced metrics comparison plot saved to {plot_path}")
        return plot_path

    except Exception as e:
        training_system.logger.error(f"Error creating enhanced metrics comparison: {e}")
        return None


def load_training_data(log_dir):
    """
    Load training data from evaluations.npz in the algorithm log directory.

    Args:
        log_dir: Path to algorithm log directory (e.g., ppo_logs/ or ddpg_logs/)

    Returns:
        Dictionary with 'timesteps', 'rewards', 'episode_lengths' as 1D arrays, or None
    """
    training_data = {}

    try:
        # Load from evaluations.npz in the log directory
        eval_path = os.path.join(log_dir, "evaluations.npz")
        if not os.path.exists(eval_path):
            logging.warning(f"evaluations.npz not found at {eval_path}")
            return None

        data = np.load(eval_path)

        # Get data from evaluations.npz
        timesteps = data.get('timesteps', [])
        rewards = data.get('results', [])  # Mean rewards from evaluation episodes
        ep_lengths = data.get('ep_lengths', [])  # Mean episode lengths

        # EvalCallback saves results as 2D arrays (n_evals x n_episodes)
        # We need to flatten or average them for plotting
        if isinstance(rewards, np.ndarray) and rewards.ndim == 2:
            # Already averaged by EvalCallback, just take the mean across episodes
            rewards = np.mean(rewards, axis=1)
        if isinstance(ep_lengths, np.ndarray) and ep_lengths.ndim == 2:
            ep_lengths = np.mean(ep_lengths, axis=1)

        # Ensure 1D arrays and convert to float
        training_data['timesteps'] = np.atleast_1d(np.squeeze(timesteps)).astype(float)
        training_data['rewards'] = np.atleast_1d(np.squeeze(rewards)).astype(float)
        training_data['episode_lengths'] = np.atleast_1d(np.squeeze(ep_lengths)).astype(float)

        logging.info(f"Loaded training data from {eval_path}: {len(training_data['timesteps'])} evaluation points")
        return training_data

    except Exception as e:
        logging.error(f"Error loading training data from {log_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to load hyperparameters into a Dict[Str: Str]:
def load_hyperparameters_from_config(file_path) -> dict:
    """Load hyperparameters from a JSON or YAML file into a dictionary."""
    import json
    import yaml

    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use .json or .yml/.yaml")
    except Exception as e:
        logging.error(f"Error loading hyperparameters: {e}")
        return {}

def run_academic_training_pipeline(total_timesteps=2500000, reward_type="smooth"):
    """
    Run academic-quality training pipeline for single PPO and DDPG models.

    This function trains one PPO and one DDPG model with detailed logging,
    comprehensive evaluation, and publication-quality plots suitable for
    academic reports and presentations.

    Args:
        total_timesteps: Total training steps (recommended 2.5M for final models)
        reward_type: Reward function type ("smooth", "hybrid", or "original")

    Returns:
        Dictionary containing training results and analysis
    """
    # Initialize enhanced training system
    training_system = MultiModelTrainingSystem()
    training_system.training_start_time = time.time()

    training_system.logger.info("="*80)
    training_system.logger.info("STARTING ACADEMIC TRAINING PIPELINE")
    training_system.logger.info("="*80)
    training_system.logger.info(f"Target timesteps: {total_timesteps:,}")
    training_system.logger.info(f"Reward type: {reward_type}")
    training_system.logger.info(f"Output directory: {training_system.output_dir}")
    training_system.logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    results = {
        'ppo_model_path': None,
        'ddpg_model_path': None,
        'ppo_training_data': None,
        'ddpg_training_data': None,
        'comparison_plot_path': None,
        'ppo_analysis_path': None,
        'ddpg_analysis_path': None,
        'final_evaluation': None,
        'training_summary': None
    }

    try:
        # Test environment setup
        test_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        test_env.close()
        training_system.logger.info(f"Environment validation successful with {reward_type} reward")

        # ==================== PPO TRAINING ====================
        training_system.logger.info("="*60)
        training_system.logger.info("TRAINING PPO MODEL")
        training_system.logger.info("="*60)

        # Enhanced logging setup
        ppo_log_dir = f"{training_system.output_dir}/ppo_logs"
        os.makedirs(ppo_log_dir, exist_ok=True)

        # Create SEPARATE training and evaluation environments
        ppo_train_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        ppo_train_env = Monitor(ppo_train_env, ppo_log_dir, allow_early_resets=True)

        # Separate evaluation environment with Monitor wrapper
        ppo_eval_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        ppo_eval_log_dir = f"{ppo_log_dir}/eval"
        os.makedirs(ppo_eval_log_dir, exist_ok=True)
        ppo_eval_env = Monitor(ppo_eval_env, ppo_eval_log_dir, allow_early_resets=True)
        
        ppo_config = load_hyperparameters_from_config("configs/hyperparams_config.yaml")['algorithm']['PPO']['params']

        # Save PPO hyperparameters
        ppo_hyperparams_file = f"{training_system.output_dir}/hyperparameters/ppo_hyperparameters.json"
        with open(ppo_hyperparams_file, 'w') as f:
            json.dump({
                'algorithm': 'PPO',
                'total_timesteps': total_timesteps,
                'reward_type': reward_type,
                'hyperparameters': ppo_config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        training_system.logger.info(f"PPO hyperparameters saved to {ppo_hyperparams_file}")

        # Create PPO model using training environment
        ppo_model = create_ppo_model(ppo_train_env, ppo_config, ppo_log_dir)

        training_system.logger.info("Starting PPO training...")

        # Calculate adaptive eval_freq (evaluate 10 times during training, minimum 1000)
        ppo_eval_freq = max(1000, total_timesteps // 10)
        training_system.logger.info(f"PPO eval_freq: {ppo_eval_freq} (evaluating every {ppo_eval_freq} steps)")

        # Calculate adaptive checkpoint_freq (checkpoint 5 times during training, minimum 10000)
        ppo_checkpoint_freq = max(10000, total_timesteps // 5)
        training_system.logger.info(f"PPO checkpoint_freq: {ppo_checkpoint_freq} (checkpointing every {ppo_checkpoint_freq} steps)")

        # Create callbacks using separate environments
        model_tracker, eval_callback, checkpoint_callback, enhanced_metrics_callback = create_callbacks_and_tracker(
            algorithm_name='PPO',
            training_system=training_system,
            variant_name='PPO_academic',
            eval_env=ppo_eval_env,
            train_env=ppo_train_env,
            eval_freq=ppo_eval_freq,
            n_eval_episodes=20,
            checkpoint_freq=ppo_checkpoint_freq,
            verbose=1
        )
        model_tracker.set_model(ppo_model)

        # Train PPO model with all callbacks
        ppo_model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback, checkpoint_callback, enhanced_metrics_callback],
            progress_bar=True
        )

        # Save enhanced metrics history after training
        enhanced_metrics_callback.save_metrics()
        ppo_enhanced_metrics = enhanced_metrics_callback.get_metrics_history()

        # Save final PPO model
        ppo_model_path = f"{ppo_log_dir}/final_ppo_model"
        ppo_model.save(ppo_model_path)
        results['ppo_model_path'] = ppo_model_path

        training_system.logger.info(f"PPO training completed. Model saved to {ppo_model_path}")

        # ==================== DDPG TRAINING ====================
        training_system.logger.info("="*60)
        training_system.logger.info("TRAINING DDPG MODEL")
        training_system.logger.info("="*60)

        # Enhanced logging setup
        ddpg_log_dir = f"{training_system.output_dir}/ddpg_logs"
        os.makedirs(ddpg_log_dir, exist_ok=True)

        # Create SEPARATE training and evaluation environments
        ddpg_train_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        ddpg_train_env = Monitor(ddpg_train_env, ddpg_log_dir, allow_early_resets=True)

        # Separate evaluation environment with Monitor wrapper
        ddpg_eval_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        ddpg_eval_log_dir = f"{ddpg_log_dir}/eval"
        os.makedirs(ddpg_eval_log_dir, exist_ok=True)
        ddpg_eval_env = Monitor(ddpg_eval_env, ddpg_eval_log_dir, allow_early_resets=True)
        
        ddpg_config = load_hyperparameters_from_config("configs/hyperparams_config.yaml")['algorithm']['DDPG']['params']

        # Save DDPG hyperparameters
        ddpg_hyperparams_file = f"{training_system.output_dir}/hyperparameters/ddpg_hyperparameters.json"
        with open(ddpg_hyperparams_file, 'w') as f:
            json.dump({
                'algorithm': 'DDPG',
                'total_timesteps': total_timesteps,
                'reward_type': reward_type,
                'hyperparameters': ddpg_config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        training_system.logger.info(f"DDPG hyperparameters saved to {ddpg_hyperparams_file}")

        # Create DDPG model using training environment
        ddpg_model = create_ddpg_model(ddpg_train_env, ddpg_config, ddpg_log_dir)

        training_system.logger.info("Starting DDPG training...")

        # Calculate adaptive eval_freq (evaluate 10 times during training, minimum 1000)
        ddpg_eval_freq = max(1000, total_timesteps // 10)
        training_system.logger.info(f"DDPG eval_freq: {ddpg_eval_freq} (evaluating every {ddpg_eval_freq} steps)")

        # Calculate adaptive checkpoint_freq (checkpoint 5 times during training, minimum 10000)
        ddpg_checkpoint_freq = max(10000, total_timesteps // 5)
        training_system.logger.info(f"DDPG checkpoint_freq: {ddpg_checkpoint_freq} (checkpointing every {ddpg_checkpoint_freq} steps)")

        # Create callbacks using separate environments
        model_tracker_ddpg, eval_callback_ddpg, checkpoint_callback_ddpg, enhanced_metrics_callback_ddpg = create_callbacks_and_tracker(
            algorithm_name='DDPG',
            training_system=training_system,
            variant_name='DDPG_academic',
            eval_env=ddpg_eval_env,
            train_env=ddpg_train_env,
            eval_freq=ddpg_eval_freq,
            n_eval_episodes=20,
            checkpoint_freq=ddpg_checkpoint_freq,
            verbose=1
        )
        model_tracker_ddpg.set_model(ddpg_model)

        # Train DDPG model with all callbacks
        ddpg_model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker_ddpg, eval_callback_ddpg, checkpoint_callback_ddpg, enhanced_metrics_callback_ddpg],
            progress_bar=True
        )

        # Save enhanced metrics history after training
        enhanced_metrics_callback_ddpg.save_metrics()
        ddpg_enhanced_metrics = enhanced_metrics_callback_ddpg.get_metrics_history()

        # Save final DDPG model
        ddpg_model_path = f"{ddpg_log_dir}/final_ddpg_model"
        ddpg_model.save(ddpg_model_path)
        results['ddpg_model_path'] = ddpg_model_path

        training_system.logger.info(f"DDPG training completed. Model saved to {ddpg_model_path}")

        # ==================== COPY BEST MODELS TO CENTRAL DIRECTORY ====================
        # Copy best models (from EvalCallback) to central models directory
        # These are the models with highest eval performance during training
        import shutil

        training_system.logger.info("Copying best models to central models directory...")

        # Copy PPO best model (best eval performance during training)
        ppo_best_src = f"{ppo_log_dir}/best_model/best_model.zip"
        if os.path.exists(ppo_best_src):
            ppo_best_dst = f"{training_system.output_dir}/models/ppo/ppo_best.zip"
            shutil.copy2(ppo_best_src, ppo_best_dst)
            training_system.logger.info(f"  PPO best model -> {ppo_best_dst}")
        else:
            training_system.logger.warning(f"  PPO best model not found at {ppo_best_src}")

        # Copy DDPG best model (best eval performance during training)
        ddpg_best_src = f"{ddpg_log_dir}/best_model/best_model.zip"
        if os.path.exists(ddpg_best_src):
            ddpg_best_dst = f"{training_system.output_dir}/models/ddpg/ddpg_best.zip"
            shutil.copy2(ddpg_best_src, ddpg_best_dst)
            training_system.logger.info(f"  DDPG best model -> {ddpg_best_dst}")
        else:
            training_system.logger.warning(f"  DDPG best model not found at {ddpg_best_src}")

        # ==================== ANALYSIS AND PLOTTING ====================
        training_system.logger.info("="*60)
        training_system.logger.info("GENERATING ACADEMIC ANALYSIS")
        training_system.logger.info("="*60)

        # Load training data for analysis (from algorithm log directories)
        # evaluations.npz is in ppo_logs/ and ddpg_logs/, not next to the model files
        ppo_training_data = load_training_data(ppo_log_dir)
        ddpg_training_data = load_training_data(ddpg_log_dir)

        results['ppo_training_data'] = ppo_training_data
        results['ddpg_training_data'] = ddpg_training_data

        # Create individual algorithm analysis plots
        if ppo_training_data:
            ppo_analysis_path = create_academic_training_curves(
                training_system, "PPO", ppo_model_path, ppo_training_data
            )
            results['ppo_analysis_path'] = ppo_analysis_path

        if ddpg_training_data:
            ddpg_analysis_path = create_academic_training_curves(
                training_system, "DDPG", ddpg_model_path, ddpg_training_data
            )
            results['ddpg_analysis_path'] = ddpg_analysis_path

        # Create comparative analysis plot
        if ppo_training_data and ddpg_training_data:
            comparison_plot_path = create_algorithm_comparison_plot(
                training_system, ppo_training_data, ddpg_training_data,
                f"PPO vs DDPG: Soccer RL ({reward_type.title()} Reward)"
            )
            results['comparison_plot_path'] = comparison_plot_path

        # ==================== FINAL EVALUATION (Run BEFORE enhanced metrics plotting) ====================
        training_system.logger.info("="*60)
        training_system.logger.info("COMPREHENSIVE MODEL EVALUATION")
        training_system.logger.info("="*60)

        # Load best models for evaluation
        best_ppo = PPO.load(f"{ppo_log_dir}/final_ppo_model")
        best_ddpg = DDPG.load(f"{ddpg_log_dir}/final_ddpg_model")

        # Comprehensive evaluation on multiple difficulties
        difficulties = ["easy", "medium", "hard"]
        evaluation_results = {"PPO": {}, "DDPG": {}}

        for difficulty in difficulties:
            training_system.logger.info(f"Evaluating on {difficulty} difficulty...")

            # PPO evaluation
            ppo_eval_env = SoccerEnv(
                config_path=training_system.config_path,
                difficulty=difficulty,
                reward_type=reward_type
            )
            ppo_comp_eval_log_dir = f"{ppo_log_dir}/comp_eval_{difficulty}"
            os.makedirs(ppo_comp_eval_log_dir, exist_ok=True)
            ppo_eval_env = Monitor(ppo_eval_env, ppo_comp_eval_log_dir, allow_early_resets=True)
            ppo_results = evaluate_model_comprehensive(
                best_ppo, ppo_eval_env, n_episodes=50, algorithm="PPO"
            )
            evaluation_results["PPO"][difficulty] = ppo_results
            ppo_eval_env.close()

            # DDPG evaluation
            ddpg_eval_env = SoccerEnv(
                config_path=training_system.config_path,
                difficulty=difficulty,
                reward_type=reward_type
            )
            ddpg_comp_eval_log_dir = f"{ddpg_log_dir}/comp_eval_{difficulty}"
            os.makedirs(ddpg_comp_eval_log_dir, exist_ok=True)
            ddpg_eval_env = Monitor(ddpg_eval_env, ddpg_comp_eval_log_dir, allow_early_resets=True)
            ddpg_results = evaluate_model_comprehensive(
                best_ddpg, ddpg_eval_env, n_episodes=50, algorithm="DDPG"
            )
            evaluation_results["DDPG"][difficulty] = ddpg_results
            ddpg_eval_env.close()

        results['final_evaluation'] = evaluation_results

        # ==================== ENHANCED METRICS PLOTTING ====================
        # Extract enhanced metrics from evaluation results and training history for plotting
        if ppo_training_data and ddpg_training_data:
            training_system.logger.info("Enriching training data with enhanced metrics...")

            # Option 1: Use final evaluation metrics (from 'medium' difficulty)
            if 'medium' in evaluation_results['PPO'] and 'medium' in evaluation_results['DDPG']:
                ppo_medium_results = evaluation_results['PPO']['medium']
                ddpg_medium_results = evaluation_results['DDPG']['medium']

                # Add enhanced metrics from final evaluation to training data dictionaries
                if 'enhanced_metrics' in ppo_medium_results:
                    ppo_training_data['enhanced_metrics'] = ppo_medium_results['enhanced_metrics']
                    training_system.logger.info(f"Added PPO final evaluation enhanced metrics: {list(ppo_medium_results['enhanced_metrics'].keys())}")

                if 'enhanced_metrics' in ddpg_medium_results:
                    ddpg_training_data['enhanced_metrics'] = ddpg_medium_results['enhanced_metrics']
                    training_system.logger.info(f"Added DDPG final evaluation enhanced metrics: {list(ddpg_medium_results['enhanced_metrics'].keys())}")

            # Option 2: Also add training history metrics (tracked throughout training)
            # These provide time-series data for plotting trends over training
            if ppo_enhanced_metrics:
                ppo_training_data['enhanced_metrics_history'] = ppo_enhanced_metrics
                training_system.logger.info(f"Added PPO training history with {len(ppo_enhanced_metrics.get('timesteps', []))} evaluation points")

            if ddpg_enhanced_metrics:
                ddpg_training_data['enhanced_metrics_history'] = ddpg_enhanced_metrics
                training_system.logger.info(f"Added DDPG training history with {len(ddpg_enhanced_metrics.get('timesteps', []))} evaluation points")

            # Now create enhanced metrics comparison plot with enriched data
            training_system.logger.info("Creating enhanced metrics comparison plot...")
            enhanced_comparison_path = create_enhanced_metrics_comparison(
                training_system, ppo_training_data, ddpg_training_data,
                f"Enhanced Metrics: PPO vs DDPG ({reward_type.title()} Reward)"
            )
            if enhanced_comparison_path:
                results['enhanced_comparison_path'] = enhanced_comparison_path
                training_system.logger.info(f"Enhanced comparison plot saved to {enhanced_comparison_path}")
            else:
                training_system.logger.info("Enhanced metrics not available - skipping enhanced comparison plot")

        # ==================== 3-WAY POLICY COMPARISON ====================
        # Compare DDPG vs PPO vs Hand-Coded baseline
        training_system.logger.info("="*60)
        training_system.logger.info("RUNNING 3-WAY POLICY COMPARISON")
        training_system.logger.info("="*60)

        try:
            # Use final trained models for comparison
            comparison_results = compare_three_policies(
                ppo_model=best_ppo,
                ddpg_model=best_ddpg,
                config_path=training_system.config_path,
                difficulty="medium",  # Use medium difficulty for fair comparison
                n_episodes=50,
                output_dir=training_system.output_dir,
                logger=training_system.logger
            )
            results['3way_comparison'] = comparison_results
            training_system.logger.info("2-way policy comparison completed successfully")
            training_system.logger.info(f"Results saved to: {comparison_results.get('json_path', 'N/A')}")
            training_system.logger.info(f"Summary report: {comparison_results.get('summary_report', 'N/A')}")
        except Exception as e:
            training_system.logger.error(f"Error during 2-way policy comparison: {e}")
            import traceback
            traceback.print_exc()

        # ==================== ACADEMIC SUMMARY ====================
        total_training_time = time.time() - training_system.training_start_time

        # Create comprehensive training summary
        # Deep copy configs to avoid circular references and convert to JSON-serializable format
        def clean_config(config):
            """Convert config dict to JSON-serializable format"""
            if isinstance(config, dict):
                return {k: clean_config(v) for k, v in config.items()}
            elif isinstance(config, (list, tuple)):
                return [clean_config(item) for item in config]
            elif isinstance(config, (np.int64, np.int32, np.integer)):
                return int(config)
            elif isinstance(config, (np.float64, np.float32, np.floating)):
                return float(config)
            elif isinstance(config, np.ndarray):
                return config.tolist()
            else:
                return config

        training_summary = {
            'experiment_details': {
                'total_timesteps': total_timesteps,
                'reward_function': reward_type,
                'training_duration_hours': total_training_time / 3600,
                'device_used': 'GPU' if torch.cuda.is_available() else 'CPU',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'environment': 'Soccer RL 2D Simulation'
            },
            'algorithm_configurations': {
                'PPO': clean_config(ppo_config),
                'DDPG': clean_config(ddpg_config)
            },
            'performance_summary': {},
            'statistical_analysis': {},
            'model_paths': {
                'ppo_final': ppo_model_path,
                'ppo_best': f"{ppo_log_dir}/best_model/best_model",
                'ddpg_final': ddpg_model_path,
                'ddpg_best': f"{ddpg_log_dir}/best_model/best_model"
            },
            'analysis_artifacts': {
                'ppo_training_curves': str(results.get('ppo_analysis_path', '')),
                'ddpg_training_curves': str(results.get('ddpg_analysis_path', '')),
                'algorithm_comparison': str(results.get('comparison_plot_path', ''))
            }
        }

        # Add performance statistics
        for algo in ['PPO', 'DDPG']:
            algo_summary = {}
            for difficulty in difficulties:
                if algo in evaluation_results and difficulty in evaluation_results[algo]:
                    eval_data = evaluation_results[algo][difficulty]
                    algo_summary[difficulty] = {
                        'mean_reward': eval_data['mean_reward'],
                        'std_reward': eval_data['std_reward'],
                        'success_rate': eval_data['success_rate'],
                        'mean_episode_length': eval_data['mean_episode_length']
                    }
            training_summary['performance_summary'][algo] = algo_summary

        # Statistical comparison
        if 'PPO' in evaluation_results and 'DDPG' in evaluation_results:
            from scipy import stats

            for difficulty in difficulties:
                if (difficulty in evaluation_results['PPO'] and
                    difficulty in evaluation_results['DDPG']):

                    ppo_rewards = evaluation_results['PPO'][difficulty]['episode_rewards']
                    ddpg_rewards = evaluation_results['DDPG'][difficulty]['episode_rewards']

                    t_stat, p_value = stats.ttest_ind(ppo_rewards, ddpg_rewards)

                    training_summary['statistical_analysis'][difficulty] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_difference': p_value < 0.05,
                        'better_algorithm': 'PPO' if np.mean(ppo_rewards) > np.mean(ddpg_rewards) else 'DDPG'
                    }

        results['training_summary'] = training_summary

        # Save comprehensive results
        results_file = f"{training_system.output_dir}/academic_training_results.json"
        with open(results_file, 'w') as f:
            # Recursive function to convert all data to JSON-serializable format
            def deep_clean_for_json(obj, seen=None):
                """Recursively clean data structures for JSON serialization"""
                if seen is None:
                    seen = set()

                # Handle circular references
                obj_id = id(obj)
                if obj_id in seen:
                    return "<circular reference>"

                # Only track mutable objects
                if isinstance(obj, (dict, list)):
                    seen.add(obj_id)

                try:
                    if isinstance(obj, dict):
                        result = {k: deep_clean_for_json(v, seen) for k, v in obj.items()}
                        seen.discard(obj_id)
                        return result
                    elif isinstance(obj, (list, tuple)):
                        result = [deep_clean_for_json(item, seen) for item in obj]
                        seen.discard(obj_id)
                        return result
                    elif isinstance(obj, (np.int64, np.int32, np.integer)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (str, int, float, bool)) or obj is None:
                        return obj
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    else:
                        return str(obj)
                finally:
                    # Clean up seen set
                    seen.discard(obj_id)

            # Deep clean all results
            clean_results = deep_clean_for_json(results)

            json.dump(clean_results, f, indent=2)

        # ==================== FINAL REPORT ====================
        training_system.logger.info("="*80)
        training_system.logger.info("ACADEMIC TRAINING PIPELINE COMPLETED")
        training_system.logger.info("="*80)
        training_system.logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
        training_system.logger.info(f"Results saved to: {results_file}")

        # Performance summary
        for algo in ['PPO', 'DDPG']:
            if algo in evaluation_results:
                training_system.logger.info(f"\n{algo} FINAL PERFORMANCE:")
                for difficulty in difficulties:
                    if difficulty in evaluation_results[algo]:
                        perf = evaluation_results[algo][difficulty]
                        training_system.logger.info(
                            f"  {difficulty.capitalize()}: "
                            f"{perf['mean_reward']:.2f}±{perf['std_reward']:.2f} reward, "
                            f"{perf['success_rate']:.1f}% success rate"
                        )

        # Artifact summary
        training_system.logger.info(f"\nGENERATED ARTIFACTS:")
        training_system.logger.info(f"  PPO model: {ppo_model_path}")
        training_system.logger.info(f"  DDPG model: {ddpg_model_path}")
        if results.get('comparison_plot_path'):
            training_system.logger.info(f"  Comparison plot: {results['comparison_plot_path']}")
        if results.get('ppo_analysis_path'):
            training_system.logger.info(f"  PPO analysis: {results['ppo_analysis_path']}")
        if results.get('ddpg_analysis_path'):
            training_system.logger.info(f"  DDPG analysis: {results['ddpg_analysis_path']}")

        training_system.logger.info("\nAcademic training pipeline completed successfully!")
        training_system.logger.info("All artifacts are ready for your academic report and presentation.")

        return results

    except Exception as e:
        training_system.logger.error(f"Academic training pipeline failed: {e}")
        import traceback
        training_system.logger.error(f"Traceback: {traceback.format_exc()}")
        return results

    finally:
        # Cleanup all environments
        if 'ppo_train_env' in locals():
            ppo_train_env.close()
        if 'ppo_eval_env' in locals():
            ppo_eval_env.close()
        if 'ddpg_train_env' in locals():
            ddpg_train_env.close()
        if 'ddpg_eval_env' in locals():
            ddpg_eval_env.close()


def main():
    """Main function with user options"""
    import argparse

    parser = argparse.ArgumentParser(description='Soccer RL Training Pipeline')
    parser.add_argument('--mode', choices=['multi', 'academic'], default='academic',
                      help='Training mode: multi for multiple variants, academic for single models with analysis')
    parser.add_argument('--timesteps', type=int, default=2500000,
                      help='Total training timesteps (default: 2.5M)')
    parser.add_argument('--reward', choices=['original', 'smooth', 'hybrid'], default='smooth',
                      help='Reward function type (default: smooth)')
    # parser.add_argument('--ppo-variants', type=int, default=3,
    #                   help='Number of PPO variants for multi mode')
    # parser.add_argument('--ddpg-variants', type=int, default=3,
    #                   help='Number of DDPG variants for multi mode')

    args = parser.parse_args()

    if args.mode == 'academic':
        # Create a temporary training system for logging before actual training starts
        temp_system = MultiModelTrainingSystem()
        logger = temp_system.logger

        logger.info("="*80)
        logger.info("STARTING ACADEMIC TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Training timesteps: {args.timesteps:,}")
        logger.info(f"Reward function: {args.reward}")
        logger.info(f"Expected duration: ~{args.timesteps/1000000 * 2:.1f} hours")
        logger.info("="*80)

        results = run_academic_training_pipeline(
            total_timesteps=args.timesteps,
            reward_type=args.reward
        )

        if results and results.get('training_summary'):
            logger.info("\nACADEMIC TRAINING COMPLETED!")
            logger.info("\nGenerated artifacts for your report:")
            for key, path in results.items():
                if path and isinstance(path, str) and ('plot' in key or 'analysis' in key):
                    logger.info(f"  {key}: {path}")

    # elif args.mode == 'multi':
    #     print("="*80)
    #     print("STARTING MULTI-MODEL TRAINING PIPELINE")
    #     print("="*80)
    #     print(f"PPO variants: {args.ppo_variants}")
    #     print(f"DDPG variants: {args.ddpg_variants}")
    #     print(f"Timesteps per model: {args.timesteps:,}")
    #     print("="*80)

    #     run_multi_model_training_pipeline(
    #         num_ppo_variants=args.ppo_variants,
    #         num_ddpg_variants=args.ddpg_variants,
    #         timesteps_per_model=args.timesteps
    #     )

if __name__ == "__main__":
    main()