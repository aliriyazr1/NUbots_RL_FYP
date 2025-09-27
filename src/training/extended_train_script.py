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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.environments.soccerenv import SoccerEnv
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
        directories = [
            experiment_dir,
            os.path.join(experiment_dir, "models", "ppo"),
            os.path.join(experiment_dir, "models", "ddpg"),
            os.path.join(experiment_dir, "plots"),
            os.path.join(experiment_dir, "logs"),
            os.path.join(experiment_dir, "tensorboard_logs"),  # Added tensorboard directory
            os.path.join(experiment_dir, "evaluations"),
            os.path.join(experiment_dir, "hyperparameters"),
            os.path.join(experiment_dir, "checkpoints")  # Added checkpoints directory
        ]

        # Create all directories
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)

        print(f"Created experiment directory: {experiment_dir}")
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
            self._evaluate_and_save_model()
        
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
    verbose: int = 1
) -> Tuple[ModelTracker, EvalCallback]:
    """
    Create ModelTracker and EvalCallback for training.
    
    Args:
        algorithm_name: Name of algorithm ("PPO", "DDPG", etc.)
        training_system: Training system instance with output_dir, logger, etc.
        variant_name: Name of the variant being trained
        eval_env: Environment for evaluation
        train_env: Training environment (optional, for timestep tracking)
        eval_freq: Frequency of evaluation (steps)
        n_eval_episodes: Number of episodes per evaluation
        verbose: Verbosity level
    
    Returns:
        Tuple of (ModelTracker, EvalCallback)
    """
    # Create ModelTracker
    model_tracker = ModelTracker(algorithm_name, training_system, variant_name, verbose=verbose)
    
    # Set train_env if provided for timestep tracking
    if train_env is not None:
        model_tracker.train_env = train_env
    
    # Create EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{training_system.output_dir}/models/{algorithm_name.lower()}/",
        log_path=f"{training_system.output_dir}/logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose
    )
    
    return model_tracker, eval_callback

def evaluate_model_comprehensive(model: Union[PPO, DDPG], env: Monitor,
                                n_episodes: int = 50, algorithm: str = "") -> Dict[str, Any]:
    """
    Comprehensive model evaluation with detailed metrics.

    Args:
        model: Trained model to evaluate
        env: Environment for evaluation
        n_episodes: Number of evaluation episodes
        algorithm: Algorithm name for logging

    Returns:
        Dictionary with evaluation results
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            # Check for goal (success)
            if terminated and hasattr(env.unwrapped, '_check_goal'):
                if env.unwrapped._check_goal():
                    success_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

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
        'total_episodes': n_episodes
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

def create_comprehensive_comparison_plots(training_system, ppo_results, ddpg_results, final_evaluation):
    """Create comprehensive plots comparing all model variants"""
    training_system.logger.info("Creating comprehensive comparison plots...")
    
    try:
        # Create a large figure with multiple subplots - increased spacing for clarity
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.4, top=0.88, bottom=0.08, left=0.08, right=0.95)
        
        fig.suptitle(f'Multi-Model Training Results - {training_system.timestamp}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Plot 1: PPO Variants Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ppo_names = [result[2] for result in ppo_results]
        ppo_scores = [result[1] for result in ppo_results]
        
        bars1 = ax1.bar(ppo_names, ppo_scores, color='skyblue', alpha=0.8)
        ax1.set_title('PPO Variants Performance', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Evaluation Score', fontsize=10)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Highlight best PPO
        if ppo_scores:
            best_idx = ppo_scores.index(max(ppo_scores))
            bars1[best_idx].set_color('gold')
            bars1[best_idx].set_edgecolor('orange')
            bars1[best_idx].set_linewidth(2)
        
        # Plot 2: DDPG Variants Performance Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ddpg_names = [result[2] for result in ddpg_results]
        ddpg_scores = [result[1] for result in ddpg_results]
        
        bars2 = ax2.bar(ddpg_names, ddpg_scores, color='lightcoral', alpha=0.8)
        ax2.set_title('DDPG Variants Performance', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Evaluation Score', fontsize=10)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Highlight best DDPG
        if ddpg_scores:
            best_idx = ddpg_scores.index(max(ddpg_scores))
            bars2[best_idx].set_color('gold')
            bars2[best_idx].set_edgecolor('orange')
            bars2[best_idx].set_linewidth(2)
        
        # Plot 3: Best Models Difficulty Comparison
        if final_evaluation["PPO"] and final_evaluation["DDPG"]:
            ax3 = fig.add_subplot(gs[0, 2])
            difficulties = ['easy', 'medium', 'hard']
            x_pos = np.arange(len(difficulties))
            width = 0.35
            
            ppo_rewards = [final_evaluation["PPO"][diff]['mean_reward'] for diff in difficulties]
            ddpg_rewards = [final_evaluation["DDPG"][diff]['mean_reward'] for diff in difficulties]
            
            ax3.bar(x_pos - width/2, ppo_rewards, width, label='Best PPO', color='skyblue', alpha=0.8)
            ax3.bar(x_pos + width/2, ddpg_rewards, width, label='Best DDPG', color='lightcoral', alpha=0.8)
            ax3.set_xlabel('Difficulty Level', fontsize=10)
            ax3.set_ylabel('Average Reward', fontsize=10)
            ax3.set_title('Best Models: Difficulty Performance', fontweight='bold', fontsize=12)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(difficulties, fontsize=8)
            ax3.tick_params(axis='y', labelsize=8)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Success Rates by Difficulty
        if final_evaluation["PPO"] and final_evaluation["DDPG"]:
            ax4 = fig.add_subplot(gs[1, 0])
            ppo_success = [final_evaluation["PPO"][diff]['success_rate'] for diff in difficulties]
            ddpg_success = [final_evaluation["DDPG"][diff]['success_rate'] for diff in difficulties]
            
            ax4.bar(x_pos - width/2, ppo_success, width, label='Best PPO', color='skyblue', alpha=0.8)
            ax4.bar(x_pos + width/2, ddpg_success, width, label='Best DDPG', color='lightcoral', alpha=0.8)
            ax4.set_xlabel('Difficulty Level', fontsize=10)
            ax4.set_ylabel('Success Rate (%)', fontsize=10)
            ax4.set_title('Goal Scoring Success Rate', fontweight='bold', fontsize=12)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(difficulties, fontsize=8)
            ax4.tick_params(axis='y', labelsize=8)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Training Summary Text
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        summary_text = "TRAINING SUMMARY:\n\n"
        summary_text += f"PPO variants trained: {len(ppo_results)}\n"
        summary_text += f"DDPG variants trained: {len(ddpg_results)}\n"
        summary_text += f"Total training time: {(time.time() - training_system.training_start_time)/3600:.1f} hours\n"
        summary_text += f"Best PPO score: {training_system.best_ppo_score:.2f}\n"
        summary_text += f"Best DDPG score: {training_system.best_ddpg_score:.2f}\n"
        
        if training_system.best_ppo_score > training_system.best_ddpg_score:
            summary_text += f"\n OVERALL WINNER: PPO\n"
        elif training_system.best_ddpg_score > training_system.best_ppo_score:
            summary_text += f"\n OVERALL WINNER: DDPG\n"
        else:
            summary_text += f"\n TIE!\n"
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Plot 6: Best Models Details
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        models_text = "BEST MODELS FOUND:\n\n"
        
        if ppo_results:
            best_ppo = max(ppo_results, key=lambda x: x[1])
            models_text += f"PPO Winner:\n"
            models_text += f"  Variant: {best_ppo[2]}\n"
            models_text += f"  Score: {best_ppo[1]:.2f}\n"
            models_text += f"  Path: {os.path.basename(best_ppo[0]) if best_ppo[0] else 'None'}\n\n"
        
        if ddpg_results:
            best_ddpg = max(ddpg_results, key=lambda x: x[1])
            models_text += f"DDPG Winner:\n"
            models_text += f"  Variant: {best_ddpg[2]}\n"
            models_text += f"  Score: {best_ddpg[1]:.2f}\n"
            models_text += f"  Path: {os.path.basename(best_ddpg[0]) if best_ddpg[0] else 'None'}\n"
        
        ax6.text(0.05, 0.95, models_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Plot 7-9: Individual variant performance distributions
        if len(ppo_results) > 1:
            ax7 = fig.add_subplot(gs[2, :])
            ppo_variant_names = [r[2] for r in ppo_results]
            ppo_variant_scores = [r[1] for r in ppo_results]
            ddpg_variant_names = [r[2] for r in ddpg_results] 
            ddpg_variant_scores = [r[1] for r in ddpg_results]
            
            # Combined comparison of all variants
            all_variants = [(f"PPO-{name}", score) for name, score in zip(ppo_variant_names, ppo_variant_scores)]
            all_variants += [(f"DDPG-{name}", score) for name, score in zip(ddpg_variant_names, ddpg_variant_scores)]
            
            # Sort by performance
            all_variants.sort(key=lambda x: x[1], reverse=True)
            
            names = [v[0] for v in all_variants]
            scores = [v[1] for v in all_variants]
            colors = ['skyblue' if 'PPO' in name else 'lightcoral' for name in names]
            
            bars = ax7.barh(names, scores, color=colors, alpha=0.8)
            ax7.set_xlabel('Evaluation Score', fontsize=10)
            ax7.set_title('All Model Variants Performance Ranking', fontweight='bold', fontsize=12)
            ax7.tick_params(axis='both', labelsize=8)
            ax7.grid(True, alpha=0.3)
            
            # Highlight top performer
            if bars:
                bars[0].set_color('gold')
                bars[0].set_edgecolor('orange')
                bars[0].set_linewidth(2)
        
        # Plot 10: Hyperparameter insights
        ax8 = fig.add_subplot(gs[3, :2])
        ax8.axis('off')
        
        insights_text = "HYPERPARAMETER INSIGHTS:\n\n"
        
        if ppo_results:
            best_ppo = max(ppo_results, key=lambda x: x[1])
            worst_ppo = min(ppo_results, key=lambda x: x[1])
            
            insights_text += f"PPO Analysis:\n"
            insights_text += f"  Best variant: {best_ppo[2]} (Score: {best_ppo[1]:.2f})\n"
            insights_text += f"  Worst variant: {worst_ppo[2]} (Score: {worst_ppo[1]:.2f})\n"
            insights_text += f"  Performance range: {best_ppo[1] - worst_ppo[1]:.2f}\n\n"
            
        if ddpg_results:
            best_ddpg = max(ddpg_results, key=lambda x: x[1])
            worst_ddpg = min(ddpg_results, key=lambda x: x[1])
            
            insights_text += f"DDPG Analysis:\n"
            insights_text += f"  Best variant: {best_ddpg[2]} (Score: {best_ddpg[1]:.2f})\n"
            insights_text += f"  Worst variant: {worst_ddpg[2]} (Score: {worst_ddpg[1]:.2f})\n"
            insights_text += f"  Performance range: {best_ddpg[1] - worst_ddpg[1]:.2f}\n"
        
        ax8.text(0.05, 0.95, insights_text, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # Plot 11: Resource usage summary
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        resource_text = "RESOURCE USAGE:\n\n"
        total_hours = (time.time() - training_system.training_start_time) / 3600
        resource_text += f"Total time: {total_hours:.1f} hours\n"
        resource_text += f"Models trained: {len(ppo_results) + len(ddpg_results)}\n"
        resource_text += f"Avg time per model: {total_hours/(len(ppo_results) + len(ddpg_results)):.1f}h\n"
        resource_text += f"Device used: {'GPU' if torch.cuda.is_available() else 'CPU'}\n"
        resource_text += f"Total models saved: {len([r for r in ppo_results + ddpg_results if r[0]])}\n"
        
        ax9.text(0.05, 0.95, resource_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        # Use constrained layout instead of tight_layout for better spacing
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plot_path = f"{training_system.output_dir}/plots/comprehensive_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        training_system.logger.info(f"Comprehensive comparison plots saved to {plot_path}")

    except Exception as e:
        training_system.logger.error(f"Error creating plots: {e}")


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
        # Load training data from TensorBoard logs or evaluations.npz
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

        if timesteps and rewards:
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
        if rewards:
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
                stats_text += "✓ Significant Learning (p < 0.05)\n"
            else:
                stats_text += "⚠ No Significant Learning\n"

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
        if ppo_data and 'timesteps' in ppo_data and 'rewards' in ppo_data:
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
            ppo_final = ppo_rewards[-len(ppo_rewards)//4:] if ppo_rewards else []
            ddpg_final = ddpg_rewards[-len(ddpg_rewards)//4:] if ddpg_rewards else []

            if ppo_final and ddpg_final:
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
        if ppo_data and ddpg_data and ppo_rewards and ddpg_rewards:
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

        if ppo_data and ddpg_data and ppo_rewards and ddpg_rewards:
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

        # Plot 4: Out of Bounds Analysis
        if 'out_of_bounds_count' in ppo_metrics and 'out_of_bounds_count' in ddpg_metrics:
            # Create violin plots for out of bounds distribution
            oob_data = [ppo_metrics['out_of_bounds_count'], ddpg_metrics['out_of_bounds_count']]
            oob_labels = ['PPO', 'DDPG']
            
            parts = ax4.violinplot(oob_data, positions=[1, 2], showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor('#2E86C1')
            parts['bodies'][1].set_facecolor('#E74C3C')
            
            ax4.set_xticks([1, 2])
            ax4.set_xticklabels(oob_labels, fontsize=12)
            ax4.set_ylabel('Out of Bounds Count per Episode', fontsize=12)
            ax4.set_title('Ball Control Quality Distribution', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')

        # Plot 5: Ball Contact Efficiency Heatmap
        if 'ball_contact_time_pct' in ppo_metrics and 'ball_contact_time_pct' in ddpg_metrics:
            # Create a comparison heatmap showing efficiency over time
            # Bin episodes into segments for heatmap
            n_segments = 20
            ppo_segments = np.array_split(ppo_metrics['ball_contact_time_pct'], n_segments)
            ddpg_segments = np.array_split(ddpg_metrics['ball_contact_time_pct'], n_segments)
            
            # Calculate statistics for each segment
            ppo_means = [np.mean(seg) for seg in ppo_segments]
            ddpg_means = [np.mean(seg) for seg in ddpg_segments]
            
            # Create heatmap data
            heatmap_data = np.array([ppo_means, ddpg_means])
            
            im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax5.set_yticks([0, 1])
            ax5.set_yticklabels(['PPO', 'DDPG'], fontsize=12)
            ax5.set_xlabel('Training Progress (Segments)', fontsize=12)
            ax5.set_title('Ball Contact Efficiency Heatmap (%)', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
            cbar.set_label('Contact Time %', fontsize=10)

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
            
            if ppo_score > ddpg_score:
                summary_text += "🏆 PPO shows better overall performance\n"
            elif ddpg_score > ppo_score:
                summary_text += "🏆 DDPG shows better overall performance\n"
            else:
                summary_text += "⚖️ Comparable performance between algorithms\n"

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


def load_training_data(model_path):
    """Load training data from various sources (TensorBoard, evaluations.npz, etc.)"""
    training_data = {}

    try:
        # Try to load from evaluations.npz first
        eval_path = os.path.join(os.path.dirname(model_path), "evaluations.npz")
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            training_data['timesteps'] = data.get('timesteps', [])
            training_data['rewards'] = data.get('results', [])
            training_data['episode_lengths'] = data.get('ep_lengths', [])

        # Try to load TensorBoard data if available
        tb_log_dir = os.path.join(os.path.dirname(model_path), "tb_logs")
        if os.path.exists(tb_log_dir):
            # This would require tensorboard parsing - simplified for now
            pass

        return training_data if training_data else None

    except Exception as e:
        print(f"Error loading training data: {e}")
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
        print(f"Error loading hyperparameters: {e}")
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
        
        # Separate evaluation environment (no monitoring to avoid interference)
        ppo_eval_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        
        ppo_config = load_hyperparameters_from_config("configs/hyperparams_config.yaml")['algorithm']['PPO']['params']

        # Create PPO model using training environment
        ppo_model = create_ppo_model(ppo_train_env, ppo_config, ppo_log_dir)

        training_system.logger.info("Starting PPO training...")

        # Create callbacks using separate environments
        model_tracker, eval_callback = create_callbacks_and_tracker(
            algorithm_name='PPO',
            training_system=training_system,
            variant_name='PPO_academic',
            eval_env=ppo_eval_env,
            train_env=ppo_train_env,
            eval_freq=10000,
            n_eval_episodes=20,
            verbose=1
        )
        model_tracker.set_model(ppo_model)

        # Train PPO model
        ppo_model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            progress_bar=True
        )

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
        
        # Separate evaluation environment
        ddpg_eval_env = SoccerEnv(config_path=training_system.config_path, reward_type=reward_type)
        
        ddpg_config = load_hyperparameters_from_config("configs/hyperparams_config.yaml")['algorithm']['DDPG']['params']

        # Create DDPG model using training environment
        ddpg_model = create_ddpg_model(ddpg_train_env, ddpg_config, ddpg_log_dir)

        training_system.logger.info("Starting DDPG training...")

        # Create callbacks using separate environments
        model_tracker_ddpg, eval_callback_ddpg = create_callbacks_and_tracker(
            algorithm_name='DDPG',
            training_system=training_system,
            variant_name='DDPG_academic',
            eval_env=ddpg_eval_env,
            train_env=ddpg_train_env,
            eval_freq=10000,
            n_eval_episodes=20,
            verbose=1
        )
        model_tracker_ddpg.set_model(ddpg_model)

        # Train DDPG model
        ddpg_model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker_ddpg, eval_callback_ddpg],
            progress_bar=True
        )

        # Save final DDPG model
        ddpg_model_path = f"{ddpg_log_dir}/final_ddpg_model"
        ddpg_model.save(ddpg_model_path)
        results['ddpg_model_path'] = ddpg_model_path

        training_system.logger.info(f"DDPG training completed. Model saved to {ddpg_model_path}")

        # ==================== ANALYSIS AND PLOTTING ====================
        training_system.logger.info("="*60)
        training_system.logger.info("GENERATING ACADEMIC ANALYSIS")
        training_system.logger.info("="*60)

        # Load training data for analysis
        ppo_training_data = load_training_data(ppo_model_path)
        ddpg_training_data = load_training_data(ddpg_model_path)

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

        # ==================== FINAL EVALUATION ====================
        training_system.logger.info("="*60)
        training_system.logger.info("COMPREHENSIVE MODEL EVALUATION")
        training_system.logger.info("="*60)

        # Load best models for evaluation
        best_ppo = PPO.load(f"{ppo_log_dir}/best_model/best_model")
        best_ddpg = DDPG.load(f"{ddpg_log_dir}/best_model/best_model")

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
            ddpg_results = evaluate_model_comprehensive(
                best_ddpg, ddpg_eval_env, n_episodes=50, algorithm="DDPG"
            )
            evaluation_results["DDPG"][difficulty] = ddpg_results
            ddpg_eval_env.close()

        results['final_evaluation'] = evaluation_results

        # ==================== ACADEMIC SUMMARY ====================
        total_training_time = time.time() - training_system.training_start_time

        # Create comprehensive training summary
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
                'PPO': ppo_config,
                'DDPG': ddpg_config
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
                'ppo_training_curves': results.get('ppo_analysis_path'),
                'ddpg_training_curves': results.get('ddpg_analysis_path'),
                'algorithm_comparison': results.get('comparison_plot_path')
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
            # Convert numpy types for JSON serialization
            def json_serializable(obj):
                if isinstance(obj, (np.int64, np.int32, np.integer)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                return obj

            # Clean results for JSON
            clean_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    clean_results[key] = {k: json_serializable(v) for k, v in value.items() if v is not None}
                elif isinstance(value, (list, tuple)):
                    clean_results[key] = [json_serializable(v) for v in value]
                else:
                    clean_results[key] = json_serializable(value)

            json.dump(clean_results, f, indent=2, default=json_serializable)

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

        training_system.logger.info("\n🎉 Academic training pipeline completed successfully!")
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
        print("="*80)
        print("STARTING ACADEMIC TRAINING PIPELINE")
        print("="*80)
        print(f"Training timesteps: {args.timesteps:,}")
        print(f"Reward function: {args.reward}")
        print(f"Expected duration: ~{args.timesteps/1000000 * 2:.1f} hours")
        print("="*80)

        results = run_academic_training_pipeline(
            total_timesteps=args.timesteps,
            reward_type=args.reward
        )

        if results and results.get('training_summary'):
            print("\n🎉 ACADEMIC TRAINING COMPLETED!")
            print("\nGenerated artifacts for your report:")
            for key, path in results.items():
                if path and isinstance(path, str) and ('plot' in key or 'analysis' in key):
                    print(f"  {key}: {path}")

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