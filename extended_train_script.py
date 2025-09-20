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
from SoccerEnv.soccerenv import SoccerEnv
from train_GUI import TrainGUI
import os, time, yaml, torch, datetime, json, copy
from collections import deque
import logging


class MultiModelTrainingSystem:
    """System for training multiple variants of each algorithm and selecting the best performers"""
    
    def __init__(self, config_path="SoccerEnv/field_config.yaml"):
        self.config_path = config_path
        self.training_start_time = None
        
        # Track best models for each algorithm type
        self.best_ppo_models = []  # List of (model_path, score, config) tuples
        self.best_ddpg_models = []  # List of (model_path, score, config) tuples
        self.best_ppo_score = -np.inf
        self.best_ddpg_score = -np.inf
        self.best_ppo_path = None
        self.best_ddpg_path = None
        
        # Create output directory structure
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"multi_model_training_{self.timestamp}"
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
        
    def _create_directories(self):
        """Create organized directory structure"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/models/ppo",
            f"{self.output_dir}/models/ddpg", 
            f"{self.output_dir}/plots",
            f"{self.output_dir}/logs",
            f"{self.output_dir}/evaluations",
            f"{self.output_dir}/hyperparameters"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
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


def get_ppo_hyperparameter_variants():
    """Generate different PPO hyperparameter configurations to try"""
    variants = {
        "conservative": {
            "learning_rate": 2e-4,
            "n_steps": 2048,
            "batch_size": 128,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "net_arch": [128, 128],
        },
        "aggressive": {
            "learning_rate": 5e-4,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 15,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "clip_range": 0.3,
            "ent_coef": 0.15,
            "vf_coef": 0.8,
            "net_arch": [256, 256, 128],
        },
        "balanced": {
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.1,
            "vf_coef": 0.5,
            "net_arch": [128, 128, 64],
        },
        "exploration": {
            "learning_rate": 3e-4,
            "n_steps": 8192,
            "batch_size": 512,
            "n_epochs": 8,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "clip_range": 0.25,
            "ent_coef": 0.2,  # Higher exploration
            "vf_coef": 0.3,
            "net_arch": [256, 128, 64],
        },
        "large_network": {
            "learning_rate": 1e-4,
            "n_steps": 2048,
            "batch_size": 128,
            "n_epochs": 12,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.08,
            "vf_coef": 0.5,
            "net_arch": [512, 512, 256, 128],
        }
    }
    return variants


def get_ddpg_hyperparameter_variants():
    """Generate different DDPG hyperparameter configurations to try"""
    variants = {
        "conservative": {
            "learning_rate": 5e-5,
            "buffer_size": 500000,
            "learning_starts": 10000,
            "batch_size": 128,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "net_arch": [128, 128],
            "noise_sigma": 0.05,
        },
        "aggressive": {
            "learning_rate": 2e-4,
            "buffer_size": 1000000,
            "learning_starts": 25000,
            "batch_size": 512,
            "tau": 0.02,
            "gamma": 0.995,
            "train_freq": 4,
            "gradient_steps": 4,
            "net_arch": [512, 512, 256],
            "noise_sigma": 0.15,
        },
        "balanced": {
            "learning_rate": 1e-4,
            "buffer_size": 1000000,
            "learning_starts": 20000,
            "batch_size": 256,
            "tau": 0.01,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 2,
            "net_arch": [256, 256, 128],
            "noise_sigma": 0.1,
        },
        "exploration": {
            "learning_rate": 1e-4,
            "buffer_size": 800000,
            "learning_starts": 30000,
            "batch_size": 256,
            "tau": 0.008,
            "gamma": 0.98,
            "train_freq": 2,
            "gradient_steps": 3,
            "net_arch": [256, 128, 64],
            "noise_sigma": 0.2,  # More exploration noise
        },
        "fast_learning": {
            "learning_rate": 3e-4,
            "buffer_size": 500000,
            "learning_starts": 5000,
            "batch_size": 128,
            "tau": 0.05,
            "gamma": 0.99,
            "train_freq": 8,
            "gradient_steps": 8,
            "net_arch": [128, 64],
            "noise_sigma": 0.1,
        }
    }
    return variants

def train_ppo_variant(training_system, variant_name, hyperparams, total_timesteps=1500000):
    """Train a single PPO variant with specific hyperparameters"""
    training_system.logger.info(f"Starting PPO variant '{variant_name}' training...")
    
    try:
        # Create environments
        train_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        eval_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        
        # Create PPO model with variant-specific hyperparameters
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
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
            tensorboard_log=f"{training_system.output_dir}/logs"
        )
        
        # Setup callbacks
        model_tracker = ModelTracker("PPO", training_system, variant_name, verbose=1)
        model_tracker.set_model(model)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ppo/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=50000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1,
        )
        
        # Train model
        training_system.logger.info(f"Training PPO-{variant_name} for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"PPO-{variant_name} training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ppo/final_ppo_{variant_name}_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Save hyperparameters for this variant
        hyperparam_file = f"{training_system.output_dir}/hyperparameters/ppo_{variant_name}_hyperparams.json"
        with open(hyperparam_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        return final_model_path, final_score, variant_name
        
    except Exception as e:
        training_system.logger.error(f"PPO-{variant_name} training failed: {e}")
        return None, -np.inf, variant_name
        
    finally:
        train_env.close()
        eval_env.close()


def train_ddpg_variant(training_system, variant_name, hyperparams, total_timesteps=1500000):
    """Train a single DDPG variant with specific hyperparameters"""
    training_system.logger.info(f"Starting DDPG variant '{variant_name}' training...")
    
    try:
        # Create environments
        train_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        eval_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        
        # Action noise for exploration
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=hyperparams["noise_sigma"] * np.ones(n_actions)
        )
        
        # Create DDPG model with variant-specific hyperparameters
        model = DDPG(
            policy="MlpPolicy",
            env=train_env,
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
            tensorboard_log=f"{training_system.output_dir}/logs"
        )
        
        # Setup callbacks
        model_tracker = ModelTracker("DDPG", training_system, variant_name, verbose=1)
        model_tracker.set_model(model)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ddpg/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=50000,  # Same as PPO for fair comparison
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        # Train model
        training_system.logger.info(f"Training DDPG-{variant_name} for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"DDPG-{variant_name} training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ddpg/final_ddpg_{variant_name}_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Save hyperparameters for this variant
        hyperparam_file = f"{training_system.output_dir}/hyperparameters/ddpg_{variant_name}_hyperparams.json"
        with open(hyperparam_file, 'w') as f:
            json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                      for k, v in hyperparams.items()}, f, indent=2)
        
        return final_model_path, final_score, variant_name
        
    except Exception as e:
        training_system.logger.error(f"DDPG-{variant_name} training failed: {e}")
        return None, -np.inf, variant_name
        
    finally:
        train_env.close()
        eval_env.close()

def continue_ddpg_training(training_system, model_path, run_name, total_timesteps=800000):
    """Continue training an existing DDPG model with comprehensive logging"""
    
    training_system.logger.info(f"Loading existing DDPG model: {model_path}")
    training_system.logger.info(f"Continue training run: '{run_name}'")
    
    try:
        # Load the existing model
        model = DDPG.load(model_path)
        training_system.logger.info("Model loaded successfully!")

        initial_timesteps = model.num_timesteps  # Remember where we started
        print(f"🔍 Starting retraining from timestep: {initial_timesteps}")
        
        # Create environments
        train_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        eval_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        
        # Update model's environment and tensorboard logging
        model.set_env(train_env)
        # model.tensorboard_log = f"{training_system.output_dir}/logs"
        model.tensorboard_log = None

        # Setup callbacks
        # model_tracker = ModelTracker("DDPG_Continue", training_system, run_name, verbose=1)
        model_tracker = ModelTracker("DDPG", training_system, run_name, verbose=1)
        model_tracker.set_model(model)
        
        train_env.total_timesteps_trained = model.num_timesteps  # Pass current timesteps to env
        train_env.initial_timesteps = initial_timesteps  # Store the starting point
        train_env.relative_timesteps = 0  # Reset relative timesteps
        model_tracker.train_env = train_env  # Pass env to tracker for timestep tracking

        # ADD THIS DEBUG:
        print(f"🔍 train_env type: {type(train_env)}")
        print(f"🔍 train_env has total_timesteps_trained: {hasattr(train_env, 'total_timesteps_trained')}")
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ddpg/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=15000,  # Evaluate every 15k steps
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=1
        )
        
        # Train model (continue from existing state)
        training_system.logger.info(f"Continue training {run_name} for {total_timesteps:,} additional timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            tb_log_name=None,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from existing timestep count
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"{run_name} continued training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ddpg/{run_name}_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Save training information
        continue_info = {
            'original_model_path': model_path,
            'run_name': run_name,
            'additional_timesteps': total_timesteps,
            'training_time_hours': training_time/3600,
            'final_score': final_score,
            'continued_model_path': final_model_path,
            'timestamp': training_system.timestamp
        }
        
        info_file = f"{training_system.output_dir}/summaries/continued_{run_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump(continue_info, f, indent=2)
        
        training_system.logger.info(f"Continued model saved: {final_model_path}")
        training_system.logger.info(f"Final score: {final_score:.2f}")
        
        return final_model_path, final_score, run_name
        
    except Exception as e:
        training_system.logger.error(f"Continue training {run_name} failed: {e}")
        return None, -np.inf, run_name
        
    finally:
        train_env.close()
        eval_env.close()


def continue_multiple_ddpg_models(model_info_list, timesteps_per_model=800000, config_path="SoccerEnv/field_config.yaml"):
    """Continue training multiple existing DDPG models
    
    Args:
        model_info_list: List of tuples (model_path, run_name)
                        e.g., [("models/my_model.zip", "aggressive_v1"), 
                               ("models/other_model.zip", "aggressive_v2")]
        timesteps_per_model: Additional timesteps to train each model
        config_path: Path to field configuration
    """
    
    # Initialize training system
    training_system = MultiModelTrainingSystem(config_path)
    training_system.training_start_time = time.time()
    training_system.logger.info(f"Continuing training for {len(model_info_list)} DDPG models")
    training_system.logger.info(f"{timesteps_per_model:,} additional timesteps per model")
    
    results = []
    
    for i, (model_path, run_name) in enumerate(model_info_list):
        training_system.logger.info("="*60)
        training_system.logger.info(f"CONTINUING MODEL {i+1}/{len(model_info_list)}: {run_name}")
        training_system.logger.info(f"Original model: {model_path}")
        training_system.logger.info("="*60)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            training_system.logger.error(f"Model file not found: {model_path}")
            results.append((None, -np.inf, run_name))
            continue
        
        continued_model_path, score, name = continue_ddpg_training(
            training_system, 
            model_path,
            run_name, 
            timesteps_per_model
        )
        
        results.append((continued_model_path, score, name))
        
        training_system.logger.info(f"Completed continuing {run_name}: Score = {score:.2f}")
    
    # Create final summary
    total_time = (time.time() - training_system.training_start_time) / 3600
    
    training_system.logger.info("="*80)
    training_system.logger.info("CONTINUED TRAINING COMPLETE!")
    training_system.logger.info("="*80)
    training_system.logger.info(f"Total training time: {total_time:.2f} hours")
    training_system.logger.info(f"Models continued: {len(results)}")
    
    # Find best model
    successful_models = [(path, score, name) for path, score, name in results if path is not None]
    
    if successful_models:
        best_model = max(successful_models, key=lambda x: x[1])
        training_system.logger.info(f"Best continued model: {best_model[2]} (Score: {best_model[1]:.2f})")
        training_system.logger.info(f"Best model path: {best_model[0]}")
        
        # Save overall summary
        overall_summary = {
            'training_session': training_system.timestamp,
            'training_type': 'continue_existing_models',
            'total_time_hours': total_time,
            'models_continued': len(results),
            'successful_continuations': len(successful_models),
            'additional_timesteps_per_model': timesteps_per_model,
            'best_continued_model': {
                'name': best_model[2],
                'score': best_model[1],
                'path': best_model[0]
            },
            'all_results': [{'name': name, 'score': score, 'path': path} for path, score, name in results],
            'original_models': [{'original_path': model_path, 'run_name': run_name} 
                              for model_path, run_name in model_info_list]
        }
        
        final_summary_file = f"{training_system.output_dir}/final_continued_training_summary.json"
        with open(final_summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
            
        training_system.logger.info(f"Final summary saved: {final_summary_file}")
    else:
        training_system.logger.error("No models continued successfully!")
    
    training_system.logger.info(f"All outputs saved to: {training_system.output_dir}")
    
    return results, training_system.output_dir

def train_all_ppo_variants(training_system, total_timesteps=1500000):
    """Train all PPO variants and track the best performer"""
    training_system.logger.info("="*60)
    training_system.logger.info("STARTING MULTI-PPO TRAINING")
    training_system.logger.info("="*60)
    
    ppo_variants = get_ppo_hyperparameter_variants()
    ppo_results = []
    
    for i, (variant_name, hyperparams) in enumerate(ppo_variants.items(), 1):
        training_system.logger.info(f"\n[{i}/{len(ppo_variants)}] Training PPO variant: {variant_name}")
        training_system.logger.info(f"Hyperparameters: {hyperparams}")
        
        model_path, score, variant = train_ppo_variant(
            training_system, variant_name, hyperparams, total_timesteps
        )
        
        ppo_results.append((model_path, score, variant, hyperparams))
        
        training_system.logger.info(f"PPO-{variant_name} final score: {score:.2f}")
        
        if score > training_system.best_ppo_score:
            training_system.logger.info(f"NEW BEST PPO VARIANT: {variant_name} (Score: {score:.2f})")
    
    # Log PPO summary
    training_system.logger.info("\n" + "="*60)
    training_system.logger.info("PPO VARIANTS SUMMARY")
    training_system.logger.info("="*60)
    
    # Sort by score
    ppo_results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_path, score, variant, hyperparams) in enumerate(ppo_results):
        status = " BEST" if i == 0 else f"#{i+1}"
        training_system.logger.info(f"{status} PPO-{variant}: {score:.2f}")
    
    return ppo_results


def train_all_ddpg_variants(training_system, total_timesteps=1500000):
    """Train all DDPG variants and track the best performer"""
    training_system.logger.info("="*60)
    training_system.logger.info("STARTING MULTI-DDPG TRAINING")
    training_system.logger.info("="*60)
    
    ddpg_variants = get_ddpg_hyperparameter_variants()
    ddpg_results = []
    
    for i, (variant_name, hyperparams) in enumerate(ddpg_variants.items(), 1):
        training_system.logger.info(f"\n[{i}/{len(ddpg_variants)}] Training DDPG variant: {variant_name}")
        training_system.logger.info(f"Hyperparameters: {hyperparams}")
        
        model_path, score, variant = train_ddpg_variant(
            training_system, variant_name, hyperparams, total_timesteps
        )
        
        ddpg_results.append((model_path, score, variant, hyperparams))
        
        training_system.logger.info(f"DDPG-{variant_name} final score: {score:.2f}")
        
        if score > training_system.best_ddpg_score:
            training_system.logger.info(f"NEW BEST DDPG VARIANT: {variant_name} (Score: {score:.2f})")
    
    # Log DDPG summary
    training_system.logger.info("\n" + "="*60)
    training_system.logger.info("DDPG VARIANTS SUMMARY")
    training_system.logger.info("="*60)
    
    # Sort by score
    ddpg_results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_path, score, variant, hyperparams) in enumerate(ddpg_results):
        status = " BEST" if i == 0 else f"#{i+1}"
        training_system.logger.info(f"{status} DDPG-{variant}: {score:.2f}")
    
    return ddpg_results

def train_new_ddpg_model(training_system, run_name, hyperparams=None, total_timesteps=1500000, n_envs=4):
    """Train a brand new DDPG model from scratch with comprehensive logging"""
    
    training_system.logger.info(f"Creating new DDPG model from scratch")
    training_system.logger.info(f"New training run: '{run_name}'")
    
    # Use default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = {
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "learning_starts": 25000,
            "batch_size": 256,
            "tau": 0.01,
            "gamma": 0.99,
            "noise_sigma": 0.1,
            "train_freq": (2, "step"),
            "gradient_steps": 1,
            "net_arch": [256, 256, 128]
        }
    
    training_system.logger.info(f"Hyperparameters: {hyperparams}")
    
    def make_env(rank):
        """Create a single environment - this runs in separate process"""
        def _init():
            env = SoccerEnv(
                difficulty="easy", 
                config_path=training_system.config_path,
                render_mode=None  # IMPORTANT: No rendering in parallel envs
            )
            env = Monitor(env)
            return env
        return _init
    
    try:
        # Create environments
        # Create parallel training environments
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        
        # Single evaluation environment (non-parallel for consistency)
        eval_env = Monitor(SoccerEnv(
            difficulty="easy", 
            config_path=training_system.config_path,
            render_mode=None
        ))  

        # Action noise for exploration
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=hyperparams["noise_sigma"] * np.ones(n_actions)
        )
        
        # Create brand new DDPG model
        model = DDPG(
            policy="MlpPolicy",
            env=train_env,
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
            device="cpu",
            tensorboard_log=f"{training_system.output_dir}/logs"
        )
        
        training_system.logger.info("New DDPG model created successfully!")
        
        # Setup callbacks
        model_tracker = ModelTracker("DDPG_Parallel", training_system, run_name, verbose=1)
        model_tracker.set_model(model)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ddpg/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=15000,  # Evaluate every 15k steps
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=1
        )
        
        # Train model from scratch
        training_system.logger.info(f"Training new {run_name} for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, eval_callback],
            tb_log_name=run_name,  # Custom tensorboard log name
            progress_bar=True,
            reset_num_timesteps=True  # Start from timestep 0
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"{run_name} training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ddpg/new_{run_name}_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Save training information
        new_model_info = {
            'run_name': run_name,
            'hyperparameters': hyperparams,
            'total_timesteps': total_timesteps,
            'training_time_hours': training_time/3600,
            'final_score': final_score,
            'model_path': final_model_path,
            'timestamp': training_system.timestamp,
            'training_type': 'parallel',
            'device': "cpu"
        }
        
        info_file = f"{training_system.output_dir}/summaries/new_{run_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump(new_model_info, f, indent=2)
        
        # Save hyperparameters separately for easy access
        hyperparam_file = f"{training_system.output_dir}/hyperparameters/new_{run_name}_hyperparams.json"
        with open(hyperparam_file, 'w') as f:
            json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                      for k, v in hyperparams.items()}, f, indent=2)
        
        training_system.logger.info(f"New model saved: {final_model_path}")
        training_system.logger.info(f"Final score: {final_score:.2f}")
        
        return final_model_path, final_score, run_name
        
    except Exception as e:
        training_system.logger.error(f"New model training {run_name} failed: {e}")
        return None, -np.inf, run_name
        
    finally:
        train_env.close()
        eval_env.close()

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
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Multi-Model Training Results - {training_system.timestamp}', fontsize=16, fontweight='bold')
        
        # Plot 1: PPO Variants Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ppo_names = [result[2] for result in ppo_results]
        ppo_scores = [result[1] for result in ppo_results]
        
        bars1 = ax1.bar(ppo_names, ppo_scores, color='skyblue', alpha=0.8)
        ax1.set_title('PPO Variants Performance', fontweight='bold')
        ax1.set_ylabel('Evaluation Score')
        ax1.tick_params(axis='x', rotation=45)
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
        ax2.set_title('DDPG Variants Performance', fontweight='bold')
        ax2.set_ylabel('Evaluation Score')
        ax2.tick_params(axis='x', rotation=45)
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
            ax3.set_xlabel('Difficulty Level')
            ax3.set_ylabel('Average Reward')
            ax3.set_title('Best Models: Difficulty Performance')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(difficulties)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Success Rates by Difficulty
        if final_evaluation["PPO"] and final_evaluation["DDPG"]:
            ax4 = fig.add_subplot(gs[1, 0])
            ppo_success = [final_evaluation["PPO"][diff]['success_rate'] for diff in difficulties]
            ddpg_success = [final_evaluation["DDPG"][diff]['success_rate'] for diff in difficulties]
            
            ax4.bar(x_pos - width/2, ppo_success, width, label='Best PPO', color='skyblue', alpha=0.8)
            ax4.bar(x_pos + width/2, ddpg_success, width, label='Best DDPG', color='lightcoral', alpha=0.8)
            ax4.set_xlabel('Difficulty Level')
            ax4.set_ylabel('Success Rate (%)')
            ax4.set_title('Goal Scoring Success Rate')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(difficulties)
            ax4.legend()
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
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
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
        
        ax6.text(0.05, 0.95, models_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
            ax7.set_xlabel('Evaluation Score')
            ax7.set_title('All Model Variants Performance Ranking', fontweight='bold')
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
        
        ax8.text(0.05, 0.95, insights_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
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
        
        ax9.text(0.05, 0.95, resource_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        plot_path = f"{training_system.output_dir}/plots/comprehensive_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        training_system.logger.info(f"Comprehensive comparison plots saved to {plot_path}")
        
    except Exception as e:
        training_system.logger.error(f"Error creating plots: {e}")


def run_multi_model_training_pipeline(num_ppo_variants=None, num_ddpg_variants=None, timesteps_per_model=1500000):
    """Run the complete multi-model training pipeline"""
    # Initialize training system
    training_system = MultiModelTrainingSystem()
    training_system.training_start_time = time.time()
    
    training_system.logger.info("="*80)
    training_system.logger.info("STARTING MULTI-MODEL TRAINING PIPELINE")
    training_system.logger.info("="*80)
    training_system.logger.info(f"Output directory: {training_system.output_dir}")
    training_system.logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Validate environment setup
    try:
        test_env = SoccerEnv(config_path=training_system.config_path)
        test_env.close()
        training_system.logger.info("Environment validation successful")
    except Exception as e:
        training_system.logger.error(f"Environment validation failed: {e}")
        return
    
    # Get hyperparameter variants
    ppo_variants = get_ppo_hyperparameter_variants()
    ddpg_variants = get_ddpg_hyperparameter_variants()
    
    # Limit variants if specified
    if num_ppo_variants:
        ppo_variants = dict(list(ppo_variants.items())[:num_ppo_variants])
    if num_ddpg_variants:
        ddpg_variants = dict(list(ddpg_variants.items())[:num_ddpg_variants])
    
    training_system.logger.info(f"Training configuration:")
    training_system.logger.info(f"  PPO variants to train: {len(ppo_variants)}")
    training_system.logger.info(f"  DDPG variants to train: {len(ddpg_variants)}")
    training_system.logger.info(f"  Timesteps per model: {timesteps_per_model:,}")
    training_system.logger.info(f"  Estimated total time: {(len(ppo_variants) + len(ddpg_variants)) * (timesteps_per_model/1000000) * 3:.1f}-{(len(ppo_variants) + len(ddpg_variants)) * (timesteps_per_model/1000000) * 5:.1f} hours")
    
    try:
        # Train all PPO variants
        training_system.logger.info(f"\n PHASE 1: Training {len(ppo_variants)} PPO variants...")
        ppo_results = train_all_ppo_variants(training_system, timesteps_per_model)
        
        # Train all DDPG variants
        training_system.logger.info(f"\n PHASE 2: Training {len(ddpg_variants)} DDPG variants...")
        ddpg_results = train_all_ddpg_variants(training_system, timesteps_per_model)
        
        # Comprehensive final evaluation of best models
        training_system.logger.info(f"\n PHASE 3: Final evaluation of best models...")
        final_results = comprehensive_final_evaluation(training_system, ppo_results, ddpg_results)
        
        # Create comprehensive plots
        create_comprehensive_comparison_plots(training_system, ppo_results, ddpg_results, final_results)
        
        # Print final summary
        total_time = (time.time() - training_system.training_start_time) / 3600
        training_system.logger.info("\n" + "="*80)
        training_system.logger.info("🎉 MULTI-MODEL TRAINING COMPLETE!")
        training_system.logger.info("="*80)
        training_system.logger.info(f"Total training time: {total_time:.2f} hours")
        training_system.logger.info(f"Models successfully trained: {len([r for r in ppo_results + ddpg_results if r[0]])}")
        training_system.logger.info(f"Output saved to: {training_system.output_dir}")
        
        training_system.logger.info("\n🏆 CHAMPIONS:")
        if training_system.best_ppo_path:
            best_ppo_variant = next(r[2] for r in ppo_results if r[0] == training_system.best_ppo_path)
            training_system.logger.info(f"  PPO Champion: {best_ppo_variant} (Score: {training_system.best_ppo_score:.2f})")
            training_system.logger.info(f"    Path: {training_system.best_ppo_path}")
        
        if training_system.best_ddpg_path:
            best_ddpg_variant = next(r[2] for r in ddpg_results if r[0] == training_system.best_ddpg_path)
            training_system.logger.info(f"  DDPG Champion: {best_ddpg_variant} (Score: {training_system.best_ddpg_score:.2f})")
            training_system.logger.info(f"    Path: {training_system.best_ddpg_path}")
        
        # Save final summary
        summary = {
            'training_duration_hours': total_time,
            'ppo_variants_trained': len(ppo_results),
            'ddpg_variants_trained': len(ddpg_results),
            'best_ppo_model': training_system.best_ppo_path,
            'best_ppo_score': training_system.best_ppo_score,
            'best_ddpg_model': training_system.best_ddpg_path,
            'best_ddpg_score': training_system.best_ddpg_score,
            'ppo_results': [(path, float(score), variant) for path, score, variant, _ in ppo_results],
            'ddpg_results': [(path, float(score), variant) for path, score, variant, _ in ddpg_results],
            'final_evaluation': final_results,
            'timestamp': training_system.timestamp
        }
        
        summary_file = f"{training_system.output_dir}/multi_model_training_summary.json"
        with open(summary_file, 'w') as f:
            # Handle numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            # Clean summary for JSON
            clean_summary = {}
            for key, value in summary.items():
                if isinstance(value, dict):
                    clean_summary[key] = {k: convert_numpy(v) for k, v in value.items()}
                elif isinstance(value, list):
                    clean_summary[key] = [convert_numpy(v) for v in value]
                else:
                    clean_summary[key] = convert_numpy(value)
            
            json.dump(clean_summary, f, indent=2)
        
        training_system.logger.info(f"Training summary saved to {summary_file}")
        
    except Exception as e:
        training_system.logger.error(f"Multi-model training pipeline failed: {e}")
        
    finally:
        training_system.logger.info("Multi-model training pipeline finished")


def main():
    """Main function with user options"""
    print("Multi-Model Soccer RL Training System")
    print("=" * 60)
    print("\nThis system will:")
    print("1. Train MULTIPLE PPO variants with different hyperparameters")
    print("2. Train MULTIPLE DDPG variants with different hyperparameters") 
    print("3. Each variant trains for LONG periods (1.5M timesteps default)")
    print("4. Automatically track and save the BEST model from each algorithm")
    print("5. Provide comprehensive comparison of ALL variants")
    print("6. Save all results with timestamps for organization")
    
    print(f"\nBy default, this trains:")
    print(f"  - {len(get_ppo_hyperparameter_variants())} PPO variants")
    print(f"  - {len(get_ddpg_hyperparameter_variants())} DDPG variants") 
    print(f"  - {len(get_ppo_hyperparameter_variants()) + len(get_ddpg_hyperparameter_variants())} models total")
    
    print(f"\nEstimated time: {(len(get_ppo_hyperparameter_variants()) + len(get_ddpg_hyperparameter_variants())) * 4:.0f}-{(len(get_ppo_hyperparameter_variants()) + len(get_ddpg_hyperparameter_variants())) * 6:.0f} hours")
    print("WARNING: This is a VERY long-running process!")
    
    print("\nTraining Options:")
    print("1. Full multi-model training (all variants)")
    print("2. Limited training (specify number of variants)")
    print("3. Quick test (100k timesteps per variant)")
    print("4. Custom timesteps")
    print("5. Fine-tune an existing model")
    print("6. Train a brand new DDPG model from scratch")
    print("0. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print(f"\n Starting FULL multi-model training...")
        print(f"This will train {len(get_ppo_hyperparameter_variants()) + len(get_ddpg_hyperparameter_variants())} models!")
        print("You can monitor progress in the terminal and log files.")
        confirm = input("Type 'YES' to confirm and start: ").strip().upper()
        if confirm == "YES":
            run_multi_model_training_pipeline()
        else:
            print("Training cancelled.")
            
    elif choice == "2":
        print("\n Limited training mode:")
        try:
            num_ppo = int(input(f"Number of PPO variants to train (max {len(get_ppo_hyperparameter_variants())}): "))
            num_ddpg = int(input(f"Number of DDPG variants to train (max {len(get_ddpg_hyperparameter_variants())}): "))
            
            print(f"\nThis will train {num_ppo + num_ddpg} models total")
            print(f"Estimated time: {(num_ppo + num_ddpg) * 4:.0f}-{(num_ppo + num_ddpg) * 6:.0f} hours")
            
            confirm = input("Type 'YES' to confirm: ").strip().upper()
            if confirm == "YES":
                run_multi_model_training_pipeline(num_ppo, num_ddpg)
            else:
                print("Training cancelled.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
        
    elif choice == "3":
        print("\n⚡ Quick test mode (100k timesteps per variant)...")
        confirm = input("Type 'YES' to confirm: ").strip().upper()
        if confirm == "YES":
            run_multi_model_training_pipeline(timesteps_per_model=100000)
        else:
            print("Training cancelled.")
            
    elif choice == "4":
        print("\n  Custom timesteps mode:")
        try:
            timesteps = int(input("Timesteps per model (e.g., 500000): "))
            num_ppo = int(input(f"Number of PPO variants (max {len(get_ppo_hyperparameter_variants())}): "))
            num_ddpg = int(input(f"Number of DDPG variants (max {len(get_ddpg_hyperparameter_variants())}): "))
            
            estimated_hours = (num_ppo + num_ddpg) * (timesteps/1000000) * 3.5
            print(f"\nThis will train {num_ppo + num_ddpg} models for {timesteps:,} timesteps each")
            print(f"Estimated time: {estimated_hours:.1f} hours")
            
            confirm = input("Type 'YES' to confirm: ").strip().upper()
            if confirm == "YES":
                run_multi_model_training_pipeline(num_ppo, num_ddpg, timesteps)
            else:
                print("Training cancelled.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
    elif choice == "5":
        print(f"\n This will continue training an existing model...")
        print("You can monitor progress in the terminal and log files.")
        
        # model_path = input("Enter the path to the existing model (.zip file): ").strip()
        #TODO: Fix the GUI  pls
        app = TrainGUI()
    
        model_path = app.open_file_dialog()
        if model_path:
            app.close_window()

        #TODO: MAybe change this to have multiple models to continue training?
        run_name = f"retrained_ddpg_model"
        timesteps = int(input("Enter the number of additional timesteps to train (e.g., 500000): ").strip())
        
        if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return
            
        model_info_list = [(model_path, run_name)]
        # Run continued training
        results, output_dir = continue_multiple_ddpg_models(model_info_list, timesteps)
        print(f"\n🎉 Continued training completed!")
        print(f"📁 Results saved to: {output_dir}")
    elif choice == "6":
        print(f"\n🆕 This will train a brand new DDPG model from scratch...")
        print("You can monitor progress in the terminal and log files.")
        
        run_name = f"new_ddpg_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        timesteps = input("Enter timesteps to train (default 1500000): ").strip()
        timesteps = int(timesteps) if timesteps else 1500000
        
        # Initialize training system
        training_system = MultiModelTrainingSystem()
        training_system.training_start_time = time.time()
        
        print(f"\n🚀 Starting training of new model: {run_name}")
        print(f"📊 Timesteps: {timesteps:,}")
        print(f"📁 Output directory: {training_system.output_dir}")
        
        # Train the new model
        final_model_path, final_score, name = train_new_ddpg_model(
            training_system, 
            run_name, 
            hyperparams=None,  # Use defaults from your function
            total_timesteps=timesteps
        )
        
        if final_model_path:
            print(f"\n🎉 New model training completed!")
            print(f"📈 Final score: {final_score:.2f}")
            print(f"💾 Model saved: {final_model_path}")
            print(f"📁 All files saved to: {training_system.output_dir}")
        else:
            print(f"\n❌ Training failed for {name}")

    else:
        print("Exiting...")


if __name__ == "__main__":
    main()