
# THis is from calculate_reward_function
 # Function to calculate rewards as an episode progresses during training   
    def _calculate_reward(self) -> float:
        reward = 0.0

        # Load parameters from config
        reward_params = self.field_config.config.get('reward_parameters', {})
        robot_params = self.field_config.config.get('robot_parameters', {})
        strategic_zones = self.field_config.config.get('strategic_zones', {})
        
        # Time penalty
        reward += reward_params['time_penalty']

        # Calculate key distances  using configurable field dimensions
        robot_x, robot_y = self.robot_pos
        ball_x, ball_y = self.ball_pos
        robot_ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        opponent_ball_distance = np.linalg.norm(self.opponent_pos - self.ball_pos)
        goal_center = self._goal_center # Goal distance that considers BOTH X and Y coordinates (adapt to field size)
        ball_to_goal_distance = np.linalg.norm(self.ball_pos - goal_center)
        robot_to_goal_distance = np.linalg.norm(self.robot_pos - goal_center)
        robot_opponent_distance = np.linalg.norm(self.robot_pos - self.opponent_pos)

        # Convert thresholds to pixels
        contact_threshold = self.field_config.meters_to_pixels(robot_params.get('contact_threshold', 0.25))
        close_threshold = self.field_config.meters_to_pixels(robot_params.get('close_threshold', 0.4))
        medium_threshold = self.field_config.meters_to_pixels(robot_params.get('medium_threshold', 0.8))
        far_threshold = self.field_config.meters_to_pixels(robot_params.get('far_threshold', 1.5))
        
        #TODO: THIS IS NOT USED Check if robot is actually in front of goal (scaled to field size)
        # goal_area_x_min = self.field_width * 0.75  # 75% of field width
        # goal_area_y_min = self.field_height * 0.4   # 40% of field height
        # goal_area_y_max = self.field_height * 0.6   # 60% of field height
        # goal_aligned = (robot_x > goal_area_x_min and goal_area_y_min < robot_y < goal_area_y_max)
        
        # PRIORITY 1: Major outcomes (also since the episode should end with either of these 3 outcomes)
        # HUGE reward for scoring
        if self._check_goal():
            # reward += reward_params['goal_scored']
            print("🎉 GOAL SCORED!")
            return reward_params.get('goal_scored', 100.0)

        # HUGE penalty for opponent scoring
        if self._check_opponent_goal():
            # reward += -reward_params['goal_scored']  # Negative of goal reward
            print("😭 OPPONENT SCORED!")
            return -reward_params.get('goal_scored', 100.0) # Negative of goal reward
            
        if self._check_ball_out_of_play():
            return reward_params.get('out_of_bounds_penalty', -10.0)

        if robot_opponent_distance < self.collision_distance:
            return reward_params.get('collision_penalty')

        # PRIORITY 2: Ball acquisition
        if robot_ball_distance < contact_threshold:
            reward += reward_params.get('contact_reward', 15.0)
        elif robot_ball_distance < close_threshold:
            reward += reward_params.get('close_reward', 8.0)
        elif robot_ball_distance < medium_threshold:
            reward += reward_params.get('medium_reward', 4.0)
        elif robot_ball_distance < far_threshold:
            reward += reward_params.get('far_reward', 1.0)
        else:
            reward += reward_params.get('too_far_penalty', -2.0)

        max_distance = self._max_distance  # Maximum distance in the field
        ball_proximity_reward = (max_distance - robot_ball_distance) / max_distance # Idea to maximise this reward as the robot is close to the ball
        reward += ball_proximity_reward * reward_params.get('proximity_multiplier', 10.0)  # Strong ball proximity reward

        # PHASE 2: Reward for moving towards ball (velocity-based)
        robot_speed = np.linalg.norm(self.robot_vel)
        if robot_ball_distance > close_threshold and robot_speed > 0.1:  # Only if moving fast enough
            # Calculate if robot is moving towards ball
            ball_direction = (self.ball_pos - self.robot_pos)
            if np.linalg.norm(ball_direction) > 0:
                ball_direction_norm = ball_direction / np.linalg.norm(ball_direction)
                robot_velocity_norm = self.robot_vel / robot_speed
                
                # Reward for moving towards ball
                velocity_alignment = np.dot(robot_velocity_norm, ball_direction_norm)
                if velocity_alignment > 0:
                    reward += velocity_alignment * robot_speed * 0.9  # Reward moving towards ball

        # PHASE 3: Reward for pushing ball toward goal
        
        
        # Only reward ball progress when robot is close to ball
        if robot_ball_distance < medium_threshold:
            max_goal_distance = np.linalg.norm([0, self.field_height//2] - goal_center)
            goal_progress = (max_goal_distance - ball_to_goal_distance) / max_goal_distance
            reward += goal_progress * 4.0
            
            # Extra reward if robot is "shepherding" ball toward goal
            shepherding_threshold = self.field_config.meters_to_pixels(0.35)  # 35cm
            if robot_ball_distance < shepherding_threshold:
                robot_ball_goal_angle = self._calculate_shepherding_angle()
                if robot_ball_goal_angle < 30:  # Very good angle
                    reward += 6.0
                elif robot_ball_goal_angle < 60:  # Good angle
                    reward += 4.0
                elif robot_ball_goal_angle < 90:  # Okay angle
                    reward += 2.0

        # PHASE 4: Reward for ball speed toward goal (good pushing)
        ball_speed = np.linalg.norm(self.ball_vel)
        if ball_speed > 0.1 and robot_ball_distance < medium_threshold:  # Only when close to ball
            ball_to_goal = goal_center - self.ball_pos
            if np.linalg.norm(ball_to_goal) > 1e-6:
                ball_to_goal_normalized = ball_to_goal / np.linalg.norm(ball_to_goal)
                ball_vel_normalized = self.ball_vel / ball_speed
                
                # Dot product: 1 = moving directly toward goal, -1 = away from goal
                goal_direction_alignment = np.dot(ball_vel_normalized, ball_to_goal_normalized)
                if goal_direction_alignment > 0:
                    reward += goal_direction_alignment * min(ball_speed, 2.0) * 1.5
        
        

        # === ANTI-EXPLOIT PENALTIES ===
        # 4A: STRONG penalties for edge/corner behavior (scaled to field size)
        edge_threshold = min(self.field_width, self.field_height) * 0.05  # Reduced from 0.1 to 0.05
        corner_threshold = min(self.field_width, self.field_height) * 0.08  # Reduced from 0.15 to 0.08

        # Robot near edges
        if (robot_x < edge_threshold or robot_x > (self.field_width - edge_threshold) or 
            robot_y < edge_threshold or robot_y > (self.field_height - edge_threshold)):
            reward += reward_params['edge_penalty']  # Strong penalty for robot camping edges
        
        # Ball near edges (prevents pushing ball to edges)
        if (ball_x < edge_threshold or ball_x > (self.field_width - edge_threshold) or 
            ball_y < edge_threshold or ball_y > (self.field_height - edge_threshold)):

            if self._check_ball_out_of_play():
                reward -= 2.0  # Major penalty for losing ball
        
        # 4C: Reward for ball being in center field (encourage proper play)
        field_center = np.array([self.field_width // 2, self.field_height // 2])
        ball_to_center_distance = np.linalg.norm(self.ball_pos - field_center)
        center_zone_radius = min(self.field_width, self.field_height) * 0.25  # 25% of field size
        if ball_to_center_distance < center_zone_radius:  # Ball near center
            reward += reward_params['center_field_bonus']

        # Collision (moderate penalty)
        robot_opponent_distance = np.linalg.norm(self.robot_pos - self.opponent_pos)
        if robot_opponent_distance < self.collision_distance:
            reward += reward_params['collision_penalty']
            
        # Corner camping prevention (scaled to field)
        # corner_threshold = min(self.field_width, self.field_height) * 0.15  # 15% of field size
        if (robot_x < corner_threshold or robot_x > (self.field_width - corner_threshold) or 
            robot_y < corner_threshold or robot_y > (self.field_height - corner_threshold)):
            reward += reward_params['corner_penalty']

        # Extra reward for getting ball into dangerous areas (near goal) - scaled to field
        dangerous_area_x = self.field_width * 0.75  # 75% of field width
        dangerous_area_y_min = self.field_height * 0.375  # 37.5% of field height
        dangerous_area_y_max = self.field_height * 0.625  # 62.5% of field height
        
        if ball_x > dangerous_area_x and dangerous_area_y_min < ball_y < dangerous_area_y_max:
            reward += 3.0  # Ball in scoring area
            
            # Even more if robot is still close (maintaining control)
            if robot_ball_distance < self.field_config.meters_to_pixels(0.4):  # Within 40cm
                reward += 5.0 # (USED TO BE 3.0)

        # Penalty if opponent is closer to ball (competitive element)
        if opponent_ball_distance < robot_ball_distance:
            reward -= 0.05

        # Clip rewards to prevent explosions
        reward = np.clip(reward, -2.0, 30.0)
        
        # Safety check
        if not np.isfinite(reward):
            reward = 0.0
            
        return float(reward)




# his was from train_ppo_agent curriculum learning through sequential envrionment changes in train_soccer_rl.py
easy_reward, _ = evaluate_policy(model, train_env, n_eval_episodes=5)
print(f"✅ Easy Phase Complete! Average reward: {easy_reward:.2f}")
train_env.close()

medium_env = Monitor(SoccerEnv(difficulty="medium", render_mode=None))
model.set_env(medium_env) # Switch model to new environment

# Continue training on medium difficulty
print("Training for 100,000 steps on medium difficulty...")
model.learn(total_timesteps=250000, progress_bar=True)

# Evaluate medium performance
medium_reward, _ = evaluate_policy(model, medium_env, n_eval_episodes=5)
print(f"✅ Medium Phase Complete! Average reward: {medium_reward:.2f}")

medium_env.close()

# PHASE 3: Hard (Expert Skills)
print("\n🔥 PHASE 3: HARD DIFFICULTY")
print("Goal: Master challenging opponent and tight game rules")

hard_env = Monitor(SoccerEnv(difficulty="hard", render_mode=None))

# Switch model to hardest environment
model.set_env(hard_env)

# Final training on hard difficulty
print("Training for 150,000 steps on hard difficulty...")
model.learn(total_timesteps=200000, progress_bar=True)

# Final evaluation
hard_reward, _ = evaluate_policy(model, hard_env, n_eval_episodes=5)
print(f"✅ Hard Phase Complete! Average reward: {hard_reward:.2f}")

hard_env.close()


"""
Extended training system for long-duration training with automatic best model selection
Idiot responsible for this crap: Ali Riyaz
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from SoccerEnv.soccerenv import SoccerEnv
import os, time, yaml, torch, datetime, json
from collections import deque
import logging


class ExtendedTrainingSystem:
    """System for training multiple models and selecting the best performers"""
    
    def __init__(self, config_path="SoccerEnv/field_config.yaml"):
        self.config_path = config_path
        self.training_start_time = None
        self.results_log = []
        self.best_models = {"PPO": None, "DDPG": None}
        self.best_scores = {"PPO": -np.inf, "DDPG": -np.inf}
        
        # Create output directory structure
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"extended_training_{self.timestamp}"
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
            f"{self.output_dir}/evaluations"
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
    
    def __init__(self, algorithm_name, training_system, verbose=0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.training_system = training_system
        self.episode_rewards = deque(maxlen=100)
        self.best_avg_reward = -np.inf
        self.evaluation_counter = 0
        self.models_saved = 0
        self.model = None
    
    def set_model(self, model):
        # Method to set model reference
        self.model = model
        
    def _on_step(self) -> bool:
        # Track episode completion
        infos = self.locals.get('infos', [])
        if isinstance(infos, dict):
            infos = [infos]
            
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
        # Evaluate and save model every 50k steps
        if self.num_timesteps % 100000 == 0 and len(self.episode_rewards) >= 20:
            self._evaluate_and_save_model()
            
        return True
    
    def _evaluate_and_save_model(self):
        """Evaluate current model and save if it's the best so far"""
        if self.model is None:  # Safety check
            self.training_system.logger.error("Model reference not set!")
            return
        
        try:
            current_avg_reward = np.mean(list(self.episode_rewards))
            
            # Test on all difficulties for comprehensive evaluation
            eval_score = self._comprehensive_evaluation()
            
            self.training_system.logger.info(
                f"{self.algorithm_name} Step {self.num_timesteps}: "
                f"Recent Avg reward = {current_avg_reward:.2f}, Eval score = {eval_score:.2f}"
            )
            
            # Save model if it's the best so far
            if eval_score > self.best_avg_reward:
                self.best_avg_reward = eval_score
                save_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"{self.training_system.output_dir}/models/{self.algorithm_name.lower()}/best_{save_timestamp}_step{self.num_timesteps}"
                self.model.save(model_path)
                
                # Update training system's best model tracking
                self.training_system.best_models[self.algorithm_name] = model_path
                self.training_system.best_scores[self.algorithm_name] = eval_score
                
                self.training_system.logger.info(
                    f"NEW BEST {self.algorithm_name} MODEL! Score: {eval_score:.2f} (saved to {model_path})"
                )
                
                # Save detailed evaluation results
                self._save_evaluation_details(eval_score)
                
            self.models_saved += 1
            
        except Exception as e:
            self.training_system.logger.error(f"Error in model evaluation: {e}")
    
    def _comprehensive_evaluation(self):
        """Evaluate model across all difficulties"""
        total_score = 0

        # These weights create a weighted average across difficulties
        # easy=0.2 means easy performance contributes 20% to final score
        # medium=0.3 means medium performance contributes 30% to final score  
        # hard=0.5 means hard performance contributes 50% to final score
        # This prioritises models that perform well on harder difficulties
        weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}  # Weight harder difficulties more
        
        for difficulty, weight in weights.items():
            try:
                eval_env = Monitor(SoccerEnv(difficulty=difficulty, config_path=self.training_system.config_path))
                mean_reward, _ = evaluate_policy(self.model, eval_env, n_eval_episodes=5, deterministic=True)
                eval_env.close()
                
                weighted_score = mean_reward * weight
                total_score += weighted_score
                
            except Exception as e:
                self.training_system.logger.error(f"Error evaluating {difficulty}: {e}")
                # Use penalty for failed evaluation to avoid skewing results
                total_score += -50.0 * weight
                
        return total_score
    
    def _save_evaluation_details(self, eval_score):
        """Save detailed evaluation results"""
        eval_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'algorithm': self.algorithm_name,
            'timesteps': self.num_timesteps,
            'eval_score': eval_score,
            'recent_rewards': list(self.episode_rewards)[-20:],  # Last 20 episodes
            'models_saved_so_far': self.models_saved
        }
        
        eval_file = f"{self.training_system.output_dir}/evaluations/{self.algorithm_name.lower()}_evaluations.json"
        
        # Load existing data or create new
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        existing_data.append(eval_data)
        
        with open(eval_file, 'w') as f:
            json.dump(existing_data, f, indent=2)

class ExtendedProgressTracker(BaseCallback):
    """Enhanced progress tracker with curriculum learning"""
    
    def __init__(self, train_env_wrapper, eval_env_wrapper, training_system, verbose=0):
        super().__init__(verbose)
        self.train_env_wrapper = train_env_wrapper
        self.eval_env_wrapper = eval_env_wrapper 
        self.training_system = training_system
        self.current_difficulty = "easy"
        self.last_difficulty_change = 0
        self.episode_rewards = deque(maxlen=100)
        
        # Load config for thresholds
        try:
            with open(training_system.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except:
            self.config = None
            training_system.logger.warning("Could not load config for curriculum progression")
        
        # Progressive difficulty thresholds
        self.difficulty_thresholds = {
            "medium": 200000,   # Switch after more training
            "hard": 800000      # Switch much later
        }
        
        # Performance thresholds (more conservative)
        self.performance_thresholds = {
            "medium": 10.0,     # Higher threshold
            "hard": 20.0        # Much higher threshold
        }

    def _on_step(self) -> bool:
        # Track episode rewards
        infos = self.locals.get('infos', [])
        if isinstance(infos, dict):
            infos = [infos]
            
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        
        # Check for curriculum progression
        if len(self.episode_rewards) >= 50:  # Need sufficient data
            self._check_difficulty_progression()
            
        return True
    
    def _check_difficulty_progression(self):
        """Check if we should progress to next difficulty"""
        current_timesteps = self.num_timesteps
        avg_reward = np.mean(list(self.episode_rewards))
        
        # Conservative progression to medium
        if (self.current_difficulty == "easy" and 
            current_timesteps >= self.difficulty_thresholds["medium"] and
            current_timesteps - self.last_difficulty_change > 100000 and
            avg_reward > self.performance_thresholds["medium"]):
            
            self._upgrade_difficulty("medium")
            
        # Conservative progression to hard  
        elif (self.current_difficulty == "medium" and 
              current_timesteps >= self.difficulty_thresholds["hard"] and
              current_timesteps - self.last_difficulty_change > 200000 and
              avg_reward > self.performance_thresholds["hard"]):
            
            self._upgrade_difficulty("hard")
    
    def _upgrade_difficulty(self, new_difficulty):
        """Upgrade to new difficulty level"""
        self.training_system.logger.info(
            f"UPGRADING DIFFICULTY: {self.current_difficulty} → {new_difficulty} "
            f"(Step {self.num_timesteps})"
        )
        
        self.current_difficulty = new_difficulty
        self.last_difficulty_change = self.num_timesteps
        
        # Update environments if config available
        if self.config:
            try:
                diff_settings = self.config['difficulty_settings'][new_difficulty]
                self._update_env_difficulty(new_difficulty, diff_settings)
            except Exception as e:
                self.training_system.logger.error(f"Error updating difficulty: {e}")
    
    def _update_env_difficulty(self, difficulty, diff_settings):
        """Update environment difficulty parameters"""
        try:
            # Update training environment
            if hasattr(self.train_env_wrapper, 'env'):
                env = self.train_env_wrapper.env
                env.difficulty = difficulty
                env.max_steps = diff_settings['max_steps']
                env.possession_distance = env.field_config.meters_to_pixels(diff_settings['possession_distance'])
                env.collision_distance = env.field_config.meters_to_pixels(diff_settings['collision_distance'])
                env.opponent_speed = diff_settings['opponent_speed']
            
            # Update evaluation environment  
            if hasattr(self.eval_env_wrapper, 'env'):
                env = self.eval_env_wrapper.env
                env.difficulty = difficulty
                env.max_steps = diff_settings['max_steps']
                env.possession_distance = env.field_config.meters_to_pixels(diff_settings['possession_distance'])
                env.collision_distance = env.field_config.meters_to_pixels(diff_settings['collision_distance'])
                env.opponent_speed = diff_settings['opponent_speed']
                
            self.training_system.logger.info(f"Environments updated to {difficulty} difficulty")
            
        except Exception as e:
            self.training_system.logger.error(f"Error updating environment: {e}")

def train_extended_ppo(training_system, total_timesteps=2000000):
    """Train PPO for extended duration with automatic best model selection"""
    training_system.logger.info("Starting extended PPO training...")
    
    try:
        # Create environments
        train_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        eval_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        
        # Create PPO model with robust hyperparameters
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            policy_kwargs=dict(
                net_arch=[128, 128, 64],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            device="cpu",
            tensorboard_log=f"{training_system.output_dir}/logs"
        )
        
        # Setup callbacks
        model_tracker = ModelTracker("PPO", training_system, verbose=1)
        model_tracker.set_model(model)

        progress_tracker = ExtendedProgressTracker(train_env, eval_env, training_system, verbose=1)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ppo/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=50000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        # Train model
        training_system.logger.info(f"Training PPO for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, progress_tracker, eval_callback],
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"PPO training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ppo/final_model_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Update best model if this is better
        if final_score > training_system.best_scores["PPO"]:
            training_system.best_models["PPO"] = final_model_path
            training_system.best_scores["PPO"] = final_score
            
        return model, final_score
        
    except Exception as e:
        training_system.logger.error(f"PPO training failed: {e}")
        return None, -np.inf
        
    finally:
        train_env.close()
        eval_env.close()

def train_extended_ddpg(training_system, total_timesteps=2000000):
    """Train DDPG for extended duration with automatic best model selection"""
    training_system.logger.info("Starting extended DDPG training...")
    
    try:
        # Create environments
        train_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        eval_env = Monitor(SoccerEnv(difficulty="easy", config_path=training_system.config_path))
        
        # Action noise for exploration
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        # Create DDPG model
        model = DDPG(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-4,
            buffer_size=1000000,  # Large buffer for long training
            learning_starts=20000,  # More initial exploration
            batch_size=256,
            tau=0.01,
            gamma=0.99,
            action_noise=action_noise,
            train_freq=4,
            gradient_steps=2,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            device="cpu",
            tensorboard_log=f"{training_system.output_dir}/logs"
        )
        
        # Setup callbacks
        model_tracker = ModelTracker("DDPG", training_system, verbose=1)
        progress_tracker = ExtendedProgressTracker(train_env, eval_env, training_system, verbose=1)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{training_system.output_dir}/models/ddpg/",
            log_path=f"{training_system.output_dir}/logs/",
            eval_freq=25000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        # Train model
        training_system.logger.info(f"Training DDPG for {total_timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[model_tracker, progress_tracker, eval_callback],
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        training_system.logger.info(f"DDPG training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation and save
        final_score = model_tracker._comprehensive_evaluation()
        final_model_path = f"{training_system.output_dir}/models/ddpg/final_model_{training_system.timestamp}"
        model.save(final_model_path)
        
        # Update best model if this is better
        if final_score > training_system.best_scores["DDPG"]:
            training_system.best_models["DDPG"] = final_model_path
            training_system.best_scores["DDPG"] = final_score
            
        return model, final_score
        
    except Exception as e:
        training_system.logger.error(f"DDPG training failed: {e}")
        return None, -np.inf
        
    finally:
        train_env.close()
        eval_env.close()

def comprehensive_final_evaluation(training_system):
    """Comprehensive evaluation of best models"""
    training_system.logger.info("Starting comprehensive final evaluation...")
    
    results = {}
    
    for algorithm in ["PPO", "DDPG"]:
        if training_system.best_models[algorithm]:
            training_system.logger.info(f"Evaluating best {algorithm} model...")
            
            try:
                # Load best model
                if algorithm == "PPO":
                    model = PPO.load(training_system.best_models[algorithm])
                else:
                    model = DDPG.load(training_system.best_models[algorithm])
                
                # Test on all difficulties with more episodes
                difficulty_results = {}
                
                for difficulty in ["easy", "medium", "hard"]:
                    training_system.logger.info(f"  Testing {algorithm} on {difficulty}...")
                    
                    test_env = Monitor(SoccerEnv(difficulty=difficulty, config_path=training_system.config_path))
                    
                    # Extended evaluation
                    episode_rewards = []
                    episode_lengths = []
                    goals_scored = 0
                    
                    for episode in range(20):  # More episodes for robust evaluation
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
                    difficulty_results[difficulty] = {
                        'mean_reward': np.mean(episode_rewards),
                        'std_reward': np.std(episode_rewards),
                        'mean_length': np.mean(episode_lengths),
                        'goals_scored': goals_scored,
                        'success_rate': (goals_scored / 20) * 100,
                        'all_rewards': episode_rewards
                    }
                
                results[algorithm] = difficulty_results
                
            except Exception as e:
                training_system.logger.error(f"Error evaluating {algorithm}: {e}")
                results[algorithm] = None
    
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

def create_final_comparison_plots(training_system, results):
    """Create comprehensive final comparison plots"""
    training_system.logger.info("Creating final comparison plots...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Extended Training Results - {training_system.timestamp}', fontsize=16, fontweight='bold')
        
        difficulties = ['easy', 'medium', 'hard']
        x_pos = np.arange(len(difficulties))
        width = 0.35
        
        # Extract data safely
        ppo_rewards = []
        ddpg_rewards = []
        ppo_success = []
        ddpg_success = []
        
        for diff in difficulties:
            # PPO data
            if results.get("PPO") and diff in results["PPO"]:
                ppo_rewards.append(results["PPO"][diff]['mean_reward'])
                ppo_success.append(results["PPO"][diff]['success_rate'])
            else:
                ppo_rewards.append(0)
                ppo_success.append(0)
                
            # DDPG data
            if results.get("DDPG") and diff in results["DDPG"]:
                ddpg_rewards.append(results["DDPG"][diff]['mean_reward'])
                ddpg_success.append(results["DDPG"][diff]['success_rate'])
            else:
                ddpg_rewards.append(0)
                ddpg_success.append(0)
        
        # Plot 1: Average Rewards
        ax1.bar(x_pos - width/2, ppo_rewards, width, label='PPO', color='skyblue', alpha=0.8)
        ax1.bar(x_pos + width/2, ddpg_rewards, width, label='DDPG', color='lightcoral', alpha=0.8)
        ax1.set_xlabel('Difficulty Level')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Average Episode Reward by Difficulty')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(difficulties)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success Rates
        ax2.bar(x_pos - width/2, ppo_success, width, label='PPO', color='skyblue', alpha=0.8)
        ax2.bar(x_pos + width/2, ddpg_success, width, label='DDPG', color='lightcoral', alpha=0.8)
        ax2.set_xlabel('Difficulty Level')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Goal Scoring Success Rate')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(difficulties)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Best model summary
        ax3.axis('off')
        summary_text = "BEST MODELS FOUND:\n\n"
        
        for algorithm in ["PPO", "DDPG"]:
            if training_system.best_models[algorithm]:
                summary_text += f"{algorithm}:\n"
                summary_text += f"  Score: {training_system.best_scores[algorithm]:.2f}\n"
                summary_text += f"  Path: {os.path.basename(training_system.best_models[algorithm])}\n\n"
            else:
                summary_text += f"{algorithm}: No valid model found\n\n"
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 4: Training summary
        ax4.axis('off')
        training_text = "TRAINING SUMMARY:\n\n"
        training_text += f"Total duration: {(time.time() - training_system.training_start_time)/3600:.1f} hours\n"
        training_text += f"Models saved: {len([m for m in training_system.best_models.values() if m])}\n"
        training_text += f"Output directory: {training_system.output_dir}\n"
        
        ax4.text(0.05, 0.95, training_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plot_path = f"{training_system.output_dir}/plots/final_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        training_system.logger.info(f"Final comparison plots saved to {plot_path}")
        
    except Exception as e:
        training_system.logger.error(f"Error creating plots: {e}")

def run_extended_training_pipeline():
    """Run the complete extended training pipeline"""
    # Initialize training system
    training_system = ExtendedTrainingSystem()
    training_system.training_start_time = time.time()
    
    training_system.logger.info("="*60)
    training_system.logger.info("STARTING EXTENDED TRAINING PIPELINE")
    training_system.logger.info("="*60)
    training_system.logger.info(f"Output directory: {training_system.output_dir}")
    
    # Validate environment setup
    try:
        test_env = SoccerEnv(config_path=training_system.config_path)
        test_env.close()
        training_system.logger.info("Environment validation successful")
    except Exception as e:
        training_system.logger.error(f"Environment validation failed: {e}")
        return
    
    # Extended training configuration
    extended_timesteps = 2000000  # 2M timesteps (~6-8 hours depending on hardware)
    
    training_system.logger.info(f"Training configuration:")
    training_system.logger.info(f"  Total timesteps per algorithm: {extended_timesteps:,}")
    training_system.logger.info(f"  Estimated total time: 12-16 hours")
    training_system.logger.info(f"  Model evaluations every 50k steps")
    training_system.logger.info(f"  Automatic curriculum progression enabled")
    
    # Train both algorithms
    try:
        # Train PPO
        training_system.logger.info("\nSTARTING PPO TRAINING PHASE")
        ppo_model, ppo_score = train_extended_ppo(training_system, extended_timesteps)
        
        # Train DDPG  
        training_system.logger.info("\nSTARTING DDPG TRAINING PHASE")
        ddpg_model, ddpg_score = train_extended_ddpg(training_system, extended_timesteps)
        
        # Comprehensive final evaluation
        training_system.logger.info("\nSTARTING FINAL EVALUATION")
        final_results = comprehensive_final_evaluation(training_system)
        
        # Create final plots
        create_final_comparison_plots(training_system, final_results)
        
        # Print final summary
        total_time = (time.time() - training_system.training_start_time) / 3600
        training_system.logger.info("\n" + "="*60)
        training_system.logger.info("EXTENDED TRAINING COMPLETE!")
        training_system.logger.info("="*60)
        training_system.logger.info(f"Total training time: {total_time:.2f} hours")
        training_system.logger.info(f"Output saved to: {training_system.output_dir}")
        
        training_system.logger.info("\nBEST MODELS:")
        for algorithm in ["PPO", "DDPG"]:
            if training_system.best_models[algorithm]:
                training_system.logger.info(f"  {algorithm}: {training_system.best_models[algorithm]}")
                training_system.logger.info(f"    Score: {training_system.best_scores[algorithm]:.2f}")
            else:
                training_system.logger.info(f"  {algorithm}: No successful model")
        
        # Save final summary
        summary = {
            'training_duration_hours': total_time,
            'best_models': training_system.best_models,
            'best_scores': training_system.best_scores,
            'final_results': final_results,
            'timestamp': training_system.timestamp
        }
        
        summary_file = f"{training_system.output_dir}/training_summary.json"
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
                else:
                    clean_summary[key] = convert_numpy(value)
            
            json.dump(clean_summary, f, indent=2)
        
        training_system.logger.info(f"Training summary saved to {summary_file}")
        
    except Exception as e:
        training_system.logger.error(f"Extended training pipeline failed: {e}")
        
    finally:
        training_system.logger.info("Extended training pipeline finished")

def main():
    """Main function with user options"""
    print("Soccer RL Extended Training System")
    print("=" * 50)
    print("\nThis system will:")
    print("1. Train PPO and DDPG for extended periods (2M timesteps each)")
    print("2. Automatically save best performing models during training")
    print("3. Use curriculum learning (easy -> medium -> hard)")
    print("4. Provide comprehensive evaluation and comparison")
    print("5. Save all results with timestamps for organization")
    print("\nEstimated total time: 12-16 hours")
    print("Warning: This is a long-running process. Ensure your system won't sleep/hibernate.")
    
    print("\nOptions:")
    print("1. Run full extended training (recommended)")
    print("2. Quick test run (100k timesteps per algorithm)")
    print("3. Resume from existing models (if available)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\nStarting full extended training...")
        print("You can monitor progress in the terminal and log files.")
        print("Training will continue automatically until completion.")
        input("Press Enter to confirm and start training...")
        run_extended_training_pipeline()
        
    elif choice == "2":
        print("\nStarting quick test run...")
        # Initialize training system
        training_system = ExtendedTrainingSystem()
        training_system.training_start_time = time.time()
        
        # Quick training with 100k timesteps
        ppo_model, ppo_score = train_extended_ppo(training_system, total_timesteps=100000)
        ddpg_model, ddpg_score = train_extended_ddpg(training_system, total_timesteps=100000)
        
        # Quick evaluation
        if ppo_model or ddpg_model:
            results = comprehensive_final_evaluation(training_system)
            create_final_comparison_plots(training_system, results)
            
        print("Quick test completed!")
        
    elif choice == "3":
        print("Resume functionality not implemented yet.")
        print("Please use option 1 or 2.")
        
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
"""
THIS IS From test_trained_model.py"""

def compare_models(config_path="SoccerEnv/field_config.yaml"):
    """Compare PPO vs DDPG visually"""
    print("🥊 Comparing PPO vs DDPG models")
    print("Testing each model for 3 episodes...\n")
    
    models_to_test = ["soccer_rl_ppo_final", "soccer_rl_ddpg_final"]
    difficulties = ["easy", "medium", "hard"]
    results = {}

    for model_name in models_to_test:
        print(f"{'='*50}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*50}")
        model_results = {}

        for difficulty in difficulties:
            print(f"\n🎯 Difficulty: {difficulty.title()}")
            result = watch_trained_robot(model_name, episodes=3, difficulty=difficulty, config_path=config_path)
            
            if result:
                model_results[difficulty] = result
                print(f"   Summary: {result['success_rate']:.1f}% goals, {result['avg_reward']:.1f} avg reward")
            
            if len(difficulties) > 1:  # Only pause if testing multiple difficulties
                input("   Press Enter to continue...")
        results[model_name] = model_results
        
        if len(models_to_test) > 1:  # Only pause if testing multiple models
            input(f"\n✅ {model_name} testing complete. Press Enter for next model...")

    # Generate comparison summary
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for difficulty in difficulties:
        print(f"\n🎯 {difficulty.upper()} Difficulty:")
        print(f"{'Model':<15} {'Success Rate':<15} {'Avg Reward':<15} {'Avg Length':<12}")
        print("-" * 65)
        
        for model_name in models_to_test:
            if model_name in results and difficulty in results[model_name]:
                data = results[model_name][difficulty]
                model_short = model_name.replace("soccer_rl_", "").replace("_final", "").upper()
                print(f"{model_short:<15} {data['success_rate']:<15.1f}% {data['avg_reward']:<15.2f} {data['avg_length']:<12.1f}")
            else:
                model_short = model_name.replace("soccer_rl_", "").replace("_final", "").upper()
                print(f"{model_short:<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
    
    return results

def test_random_baseline(config_path="SoccerEnv/field_config.yaml"):
    """Test random policy for comparison"""
    print("🎲 Testing random policy (baseline)...")
    
    env = SoccerEnv(render_mode="human", config_path=config_path)
    
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(env.max_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        time.sleep(0.02)  # Slow it down so we can watch
        
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"🎲 Random policy result:")
    print(f"   Reward: {total_reward:.2f}")
    print(f"   Steps: {steps}")
    print("This is what your trained model should beat!")

def detailed_model_comparison(episodes=5, difficulty="medium", config_path="SoccerEnv/field_config.yaml", testing_mode=True):
    """
    Test both PPO and DDPG models for detailed comparison with plots
    """
    print(f"🔬 DETAILED MODEL COMPARISON")
    print(f"Testing both models for {episodes} episodes each on {difficulty} difficulty")
    print("="*60)
    
    # Store results for plotting
    results = {
        'PPO': {'rewards': [], 'episode_lengths': [], 'timesteps': []},
        'DDPG': {'rewards': [], 'episode_lengths': [], 'timesteps': []}
    }
    
    models_to_test = [
        ('PPO', 'soccer_rl_ppo_final'),
        ('DDPG', 'soccer_rl_ddpg_final')
    ]
    
    for model_type, model_name in models_to_test:
        print(f"\n🤖 Testing {model_type} Model ({model_name})")
        print("-" * 40)
        
        try:
            # Load model
            if model_type == "PPO":
                model = PPO.load(model_name)
            else:
                model = DDPG.load(model_name)
            print(f"✅ {model_type} model loaded successfully!")
            
            # Test episodes
            episode_rewards = []
            episode_lengths = []
            cumulative_timesteps = []
            total_timesteps = 0
            goals_scored = 0
            
            env = SoccerEnv(render_mode="human", difficulty=difficulty, config_path=config_path, testing_mode=testing_mode)  # With rendering to see what's happening
            
            for episode in range(episodes):
                print(f"  Episode {episode + 1}/{episodes}...", end=" ")
                
                obs, _ = env.reset()
                episode_reward = 0
                steps = 0
                
                for step in range(env.max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    steps += 1
                    total_timesteps += 1
                    
                    if terminated:
                        if env._check_goal():
                            goals_scored += 1
                        break
                    elif truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                cumulative_timesteps.append(total_timesteps)
                
                print(f"Reward: {episode_reward:.2f}, Steps: {steps}")
            
            env.close()
            
            # Store results
            results[model_type]['rewards'] = episode_rewards
            results[model_type]['episode_lengths'] = episode_lengths
            results[model_type]['timesteps'] = cumulative_timesteps
            results[model_type]['goals_scored'] = goals_scored
            results[model_type]['success_rate'] = (goals_scored / episodes) * 100
            
            # Print summary
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)
            
            print(f"  📊 {model_type} Summary:")
            print(f"    Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"    Average Episode Length: {mean_length:.1f} ± {std_length:.1f} steps")
            print(f"    Goals Scored: {goals_scored}/{episodes} ({(goals_scored/episodes)*100:.1f}%)")
            
        except FileNotFoundError:
            print(f"❌ {model_type} model file '{model_name}' not found!")
            results[model_type] = None
    
    # Create plots - THIS IS THE KEY PART THAT WAS MISSING!
    print(f"\n📈 Creating comprehensive comparison plots...")
    create_comparison_plots(results, episodes, difficulty)
    
    return results

def create_comparison_plots(results, episodes, difficulty):
    """
    Create research-style comparison plots
    """
    print(f"\n📈 Creating comparison plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for consistency with research paper style
    colors = {'PPO': '#1f77b4', 'DDPG': '#2ca02c'}  # Blue, Green
    
    # Plot 1: Episode Rewards vs Episodes (not cumulative timesteps)
    ax1.set_title(f'Episode Rewards Comparison\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            episodes_range = range(1, episodes + 1)
            rewards = results[model_type]['rewards']
            
            # Plot individual episode rewards as line with markers
            ax1.plot(episodes_range, rewards, 
                    color=colors[model_type], 
                    marker='o', 
                    markersize=6,
                    linewidth=2.5,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add trend line
            z = np.polyfit(episodes_range, rewards, 1)
            p = np.poly1d(z)
            ax1.plot(episodes_range, p(episodes_range), 
                    color=colors[model_type], 
                    linestyle='--', 
                    alpha=0.5,
                    linewidth=1)
    
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(range(1, episodes + 1))
    
    # Plot 2: Episode Length vs Episodes
    ax2.set_title(f'Episode Length Comparison\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Length (steps)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            episodes_range = range(1, episodes + 1)
            lengths = results[model_type]['episode_lengths']
            
            # Plot episode lengths as line with markers
            ax2.plot(episodes_range, lengths, 
                    color=colors[model_type], 
                    marker='s', 
                    markersize=6,
                    linewidth=2.5,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add trend line
            z = np.polyfit(episodes_range, lengths, 1)
            p = np.poly1d(z)
            ax2.plot(episodes_range, p(episodes_range), 
                    color=colors[model_type], 
                    linestyle='--', 
                    alpha=0.5,
                    linewidth=1)
    
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xticks(range(1, episodes + 1))
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{difficulty}_{episodes}episodes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional cumulative performance plot (similar to research paper)
    create_cumulative_performance_plot(results, episodes, difficulty)

def create_cumulative_performance_plot(results, episodes, difficulty):
    """
    Create cumulative performance plots similar to the research paper
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'PPO': '#1f77b4', 'DDPG': '#2ca02c'}
    
    # Plot 1: Cumulative Average Reward
    ax1.set_title(f'Cumulative Average Reward\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Cumulative Average Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            rewards = results[model_type]['rewards']
            cumulative_avg_rewards = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
            episodes_range = range(1, episodes + 1)
            
            ax1.plot(episodes_range, cumulative_avg_rewards, 
                    color=colors[model_type], 
                    linewidth=3,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add confidence band (using standard error)
            cumulative_std = [np.std(rewards[:i+1]) / np.sqrt(i+1) for i in range(len(rewards))]
            ax1.fill_between(episodes_range, 
                           np.array(cumulative_avg_rewards) - np.array(cumulative_std),
                           np.array(cumulative_avg_rewards) + np.array(cumulative_std),
                           color=colors[model_type], alpha=0.2)
    
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(range(1, episodes + 1))
    
    # Plot 2: Cumulative Average Episode Length
    ax2.set_title(f'Cumulative Average Episode Length\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Cumulative Average Episode Length (steps)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            lengths = results[model_type]['episode_lengths']
            cumulative_avg_lengths = [np.mean(lengths[:i+1]) for i in range(len(lengths))]
            episodes_range = range(1, episodes + 1)
            
            ax2.plot(episodes_range, cumulative_avg_lengths, 
                    color=colors[model_type], 
                    linewidth=3,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add confidence band
            cumulative_std = [np.std(lengths[:i+1]) / np.sqrt(i+1) for i in range(len(lengths))]
            ax2.fill_between(episodes_range, 
                           np.array(cumulative_avg_lengths) - np.array(cumulative_std),
                           np.array(cumulative_avg_lengths) + np.array(cumulative_std),
                           color=colors[model_type], alpha=0.2)
    
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xticks(range(1, episodes + 1))
    
    plt.tight_layout()
    plt.savefig(f'cumulative_performance_{difficulty}_{episodes}episodes.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_statistical_analysis(results, episodes=5):
    """
    Provide detailed statistical analysis of the comparison
    """
    print(f"\n📊 DETAILED STATISTICAL ANALYSIS")
    print("="*50)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            rewards = results[model_type]['rewards']
            lengths = results[model_type]['episode_lengths']
            
            print(f"\n🤖 {model_type} Statistics:")
            print(f"  Rewards:")
            print(f"    Mean: {np.mean(rewards):.3f}")
            print(f"    Std:  {np.std(rewards):.3f}")
            print(f"    Min:  {np.min(rewards):.3f}")
            print(f"    Max:  {np.max(rewards):.3f}")
            print(f"    Median: {np.median(rewards):.3f}")
            
            print(f"  Episode Lengths:")
            print(f"    Mean: {np.mean(lengths):.1f}")
            print(f"    Std:  {np.std(lengths):.1f}")
            print(f"    Min:  {np.min(lengths)}")
            print(f"    Max:  {np.max(lengths)}")
            print(f"    Median: {np.median(lengths):.1f}")
    
    # Compare models if both exist
    if results['PPO'] is not None and results['DDPG'] is not None:
        ppo_rewards = results['PPO']['rewards']
        ddpg_rewards = results['DDPG']['rewards']
        
        print(f"\n🔄 Model Comparison:")
        print(f"  Reward Difference (PPO - DDPG):")
        print(f"    Mean: {np.mean(ppo_rewards) - np.mean(ddpg_rewards):.3f}")
        print(f"    PPO better episodes: {sum(1 for p, d in zip(ppo_rewards, ddpg_rewards) if p > d)}/{episodes}")
        
        # Simple t-test equivalent
        try:
            t_stat, p_value = stats.ttest_ind(ppo_rewards, ddpg_rewards)
            print(f"    Statistical significance (t-test): p={p_value:.4f}")
            if p_value < 0.05:
                print(f"    ✅ Significant difference detected (p < 0.05)")
            else:
                print(f"    ⚠️  No significant difference (p >= 0.05)")
        except:
            print(f"    ⚠️  Could not perform t-test (scipy not available)")

        # Create comparison plots
        create_comparison_plots(results, episodes, "analysis")
        
        # Create visualization
        try:
            plt.figure(figsize=(15, 5))
            
            # Reward comparison
            plt.subplot(1, 3, 1)
            plt.boxplot([ppo_rewards, ddpg_rewards], labels=['PPO', 'DDPG'])
            plt.title('Reward Distribution')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.3)
            
            # Success rate comparison
            plt.subplot(1, 3, 2)
            models = ['PPO', 'DDPG']
            success_rates = [results['PPO']['success_rate'], results['DDPG']['success_rate']]
            colors = ['skyblue', 'lightcoral']
            plt.bar(models, success_rates, color=colors)
            plt.title('Success Rate Comparison')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            
            # Episode length comparison
            ppo_lengths = results['PPO']['episode_lengths']
            ddpg_lengths = results['DDPG']['episode_lengths']
            plt.subplot(1, 3, 3)
            plt.boxplot([ppo_lengths, ddpg_lengths], labels=['PPO', 'DDPG'])
            plt.title('Episode Length Distribution')
            plt.ylabel('Steps per Episode')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('detailed_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📈 Detailed analysis visualization saved as 'detailed_model_comparison.png'")
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")
