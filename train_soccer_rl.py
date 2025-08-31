"""
Author: Ali Riyaz
Student Number: C3412624
Last Updated: 10/08/2025
"""

# train_soccer_rl.py - For RL training
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from SoccerEnv.soccerenv import SoccerEnv
import os, time, yaml, torch, datetime
from collections import deque



#TODO: FIX THE DATA COLLECTION AND COMPARISON and plotting when Rewards VS TIMESTEPS ASAP
#TODO:
#TODO: 

def get_timestamp():
    """Generate timestamp for file naming. Format : %Y%m%d_%H%M%S"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_output_directory():
    """Create organised output directory with timestamp"""
    timestamp = get_timestamp()
    output_dir = f"training_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    return output_dir

class RealtimePlottingCallback(BaseCallback):
    """Callback for real-time plotting of training progress"""
    
    def __init__(self, plot_freq=1000, verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.episode_rewards = deque(maxlen=1000)  # Keep last 1000 episodes
        self.episode_lengths = deque(maxlen=1000)
        self.timesteps = []
        self.mean_rewards = []
        self.mean_lengths = []
        
        # Set up real-time plotting
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Soccer RL Training Progress', fontsize=16, fontweight='bold')
        
        # Initialise empty plots
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Mean Episode Reward')
        self.ax1.set_xlabel('Timesteps')
        self.ax1.set_ylabel('Mean Reward (last 100 episodes)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2, label='Mean Episode Length')
        self.ax2.set_xlabel('Timesteps')
        self.ax2.set_ylabel('Mean Episode Length (last 100 episodes)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _on_step(self) -> bool:
        # Episode detection to handle both single environment and vectorised environments
        infos = self.locals.get('infos', [])
        if isinstance(infos, dict):
            infos = [infos] # Convert single info dict to list for consistent processing
        
        # Check each info dict for completed episodes
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if self.verbose > 0:
                    print(f"🏆 Episode completed: Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Update plot every plot_freq steps
        if self.num_timesteps % self.plot_freq == 0 and len(self.episode_rewards) >= 10:
            self.timesteps.append(self.num_timesteps)
            
            # Calculate mean of last 100 episodes
            recent_rewards = list(self.episode_rewards)[-100:]
            recent_lengths = list(self.episode_lengths)[-100:]
            
            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)
            
            self.mean_rewards.append(mean_reward)
            self.mean_lengths.append(mean_length)
            
            # Update plots
            self.line1.set_data(self.timesteps, self.mean_rewards)
            self.line2.set_data(self.timesteps, self.mean_lengths)
            
            # Rescale axes
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # Draw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # Print progress
            if self.verbose > 0:
                print(f"📈 Step {self.num_timesteps}: Mean reward = {mean_reward:.2f}, Mean length = {mean_length:.1f}")
        
        return True
    
    def save_plots(self, filename="training_progress"):
        """Save the final plots"""
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.timesteps, self.mean_rewards, 'b-', linewidth=2)
        plt.title('Training Progress: Mean Episode Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward (last 100 episodes)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.timesteps, self.mean_lengths, 'g-', linewidth=2)
        plt.title('Training Progress: Mean Episode Length')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Length (last 100 episodes)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # filename = f"{self.algorithm_name.lower()}_{filename_prefix}_{timestamp}.png"
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved as '{filename}'")

class ProgressTracker(BaseCallback):
    """Track learning progress and switch difficulty for both training and evaluation"""
    
    def __init__(self, train_env_wrapper, eval_env_wrapper, verbose=0, config_path="SoccerEnv/field_config.yaml"):
        super().__init__(verbose)
        self.train_env_wrapper = train_env_wrapper
        self.eval_env_wrapper = eval_env_wrapper
        self.current_difficulty = "easy"
        self.last_difficulty_change = 0
        self.episode_rewards = deque(maxlen=50)  # Track recent performance
        
        # Load config for difficulty thresholds
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except:
            print("Warning: Could not load config file, using default difficulty progression")
            self.config = None
        
        # Difficulty progression thresholds
        self.difficulty_thresholds = {
            "medium": 100000,  # Switch to medium after 100k steps
            "hard": 250000     # Switch to hard after 250k steps
        }

    def _update_env_difficulty(self, difficulty, diff_settings):
        """Update environment difficulty parameters"""
        try:
            # Update training environment
            if hasattr(self.train_env_wrapper, 'env'):
                env = self.train_env_wrapper.env
                env.difficulty = difficulty
                env.max_steps = diff_settings['max_steps']
                # Convert meters to pixels for possession and collision distances
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
                
            print(f"✅ Environments updated to {difficulty} difficulty")
            
        except Exception as e:
            print(f"❌ Error updating environment difficulty: {e}")

    def _on_step(self) -> bool:
        # Track episode rewards for performance-based progression (Handles both single environment and vectorised environments)
        infos = self.locals.get('infos', [])
        
        # Convert single info dict to list for consistent processing
        if isinstance(infos, dict):
            infos = [infos]
        
        # Track episode rewards when episodes complete
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                if self.verbose > 0:
                    print(f"📊 Episode reward for curriculum: {episode_reward:.2f}")
        
        current_timesteps = self.num_timesteps
        
        # Check for difficulty progression based on performance AND timesteps
        avg_reward = np.mean(self.episode_rewards) if len(self.episode_rewards) >= 10 else -999
        
        # Switch to medium difficulty
        if (self.current_difficulty == "easy" and 
            current_timesteps >= self.difficulty_thresholds["medium"] and
            current_timesteps - self.last_difficulty_change > 50000 and
            avg_reward > 5.0):  # Performance threshold
            
            print(f"\n{'='*50}")
            print(f"🎯 DIFFICULTY UPGRADE: Easy → Medium (Step {current_timesteps})")
            print(f"   Recent performance: {avg_reward:.2f} average reward")
            print(f"{'='*50}")
            
            self.current_difficulty = "medium"
            self.last_difficulty_change = current_timesteps
            
            # Update environments with new difficulty parameters
            if self.config:
                diff_settings = self.config['difficulty_settings']['medium']
                self._update_env_difficulty("medium", diff_settings)
            
        # Switch to hard difficulty
        elif (self.current_difficulty == "medium" and 
              current_timesteps >= self.difficulty_thresholds["hard"] and
              current_timesteps - self.last_difficulty_change > 50000 and
              avg_reward > 8.0):  # Higher performance threshold
            
            print(f"\n{'='*50}")
            print(f"🎯 DIFFICULTY UPGRADE: Medium → Hard (Step {current_timesteps})")
            print(f"   Recent performance: {avg_reward:.2f} average reward")
            print(f"{'='*50}")
            
            self.current_difficulty = "hard"
            self.last_difficulty_change = current_timesteps
            
            # Update environments with new difficulty parameters
            if self.config:
                diff_settings = self.config['difficulty_settings']['hard']
                self._update_env_difficulty("hard", diff_settings)
        
        return True
   
    
def create_monitored_env(difficulty, config_path="SoccerEnv/field_config.yaml"):
    """Create environment with monitoring wrapper"""
    env = SoccerEnv(difficulty=difficulty, config_path=config_path)
    env = Monitor(env)  # This wrapper logs episode rewards and lengths
    return env

def train_ppo_agent(config_path="SoccerEnv/field_config.yaml"):
    """Train PPO agent with proper hyperparameters"""
    print("🚀 Training PPO Agent...")

    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        field_type = config['field_type']
        print(f"📏 Using {field_type} field configuration")
    except:
        print("⚠️  Using default field configuration")
        config_path = None
    
    # Create training and evaluation environments starting with "easy" difficulty
    train_env = create_monitored_env("easy", config_path)
    eval_env = create_monitored_env("easy", config_path)
    
    # Create model with hyperparameters for continuous control
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,      # Learning rate for PPO (First I put 3e-4 but too slow to improve then 1e-3 but still too long)

        # Training stability
        n_steps=4096,            # Number of steps to run for each environment per update (USed to be 2048) (more steps for slower environment)
        batch_size=256,           # Minibatch size (Used to be 64) (128 is more stable for larger envs)
        n_epochs=11,              # Number of epochs when optimising the surrogate loss (used to be 10 then 8) This is for better learning
        
        # Exploration VS Exploitation
        gamma=0.99,              # Discount factor (Used to be 0.97) for longer-term thinking
        gae_lambda=0.95,         # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,          # Clipping parameter (Used to be 0.2)
        ent_coef=0.2,           # Entropy coefficient for exploration (used to be 0.05)
        
        # Stability settings
        vf_coef=0.5,             # Value function coefficient
        max_grad_norm=0.5,       # Gradient clipping
        normalize_advantage= True,  # Normalize advantages for better stability

        # Network architecture for stability
        policy_kwargs=dict(
            net_arch=[128, 128, 64],      # Larger network (was [64, 64] which was a smaller and more stable network)
            activation_fn=torch.nn.ReLU, # More stable than Tanh but now changing to ReLU for better gradient flow
        ),

        verbose=1,               # Print training progress
        device="cpu",             # Use GPU (CPU more stable for small envs but trying out GPU if possible)
        tensorboard_log="./logs"
    )
    
    # Progress tracking
    progress_cb = ProgressTracker(train_env, eval_env, verbose=1, config_path=config_path)

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=15000,          # Evaluate every 15000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    plotting_callback = RealtimePlottingCallback(plot_freq=2000, verbose=1)
    
    print("Starting PPO training... This may take 5-15 minutes depending on your computer.")
    print("Watch for increasing episode rewards - that means learning is happening!")
    
    #TODO: Once the model is proving effective on small batches, increase the number of timesteps for both algos PPO and DDPG
    # Train the model - (50000 for quick testing) but 100000 timesteps for better learning but to check if things are going in the right direction
    total_timesteps = 400000 # Used to be 250000
    try:
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_cb, eval_callback, plotting_callback],
            progress_bar=True
        )

        # Test on all difficulties
        # Evaluation on randomised scenarios
        print("\n📊 Testing on all difficulty levels...")
        final_results = {}

        for diff in ["easy", "medium", "hard"]:
            test_env = create_monitored_env(diff)
            mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
            final_results[diff] = (mean_reward, std_reward)
            print(f"  {diff.title()}: {mean_reward:.2f} ± {std_reward:.2f}")
            test_env.close()
        
        # Check if curriculum actually worked
        easy_reward = final_results["easy"][0]
        medium_reward = final_results["medium"][0]
        hard_reward = final_results["hard"][0]
                
        # Print curriculum results
        print("\n" + "=" * 50)
        print("🎓 CURRICULUM LEARNING COMPLETE!")
        print("=" * 50)
        print(f"Easy difficulty performance:   {easy_reward:.2f}")
        print(f"Medium difficulty performance: {medium_reward:.2f}")
        print(f"Hard difficulty performance:   {hard_reward:.2f}")
        
        # Check if curriculum worked
        if easy_reward > medium_reward > hard_reward:
            print("✅ Curriculum working correctly - performance decreases with difficulty")
        elif hard_reward > easy_reward:
            print("🚀 Excellent! Model performs well even on hardest difficulty")
        else:
            print("⚠️ Mixed results - curriculum may need tuning")
        
        training_time = time.time() - start_time

        # Save the final model
        timestamp = get_timestamp()
        model_name = "soccer_rl_ppo_" + timestamp
        model.save(model_name)
        print(f"✅ PPO training completed! Model saved as {model_name}")
        print(f"⏱️  Training completed in {training_time:.2f} seconds")

        # Save training plots
        plotting_callback.save_plots("ppo_training_progress_" + timestamp)       
        
        return model
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        timestamp = get_timestamp()
        model_name = "soccer_rl_ppo_partial_" + timestamp

        # Save partial progress
        model.save(model_name)
        plotting_callback.save_plots("ppo_partial_training_progress_" + timestamp)
        print("💾 Partial model saved")
        return None, None
    
    finally:
        train_env.close()
        eval_env.close() 

#TODO: Check whether we need to add some kind of noise for DDPG (Example given below)
# OPTIONAL: Add noise for better DDPG exploration
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# def train_ddpg_with_noise():
#     """DDPG with exploration noise for better learning"""
    
#     # ... setup code ...
    
#     # Add exploration noise
#     n_actions = train_env.action_space.shape[-1]
#     action_noise = OrnsteinUhlenbeckActionNoise(
#         mean=np.zeros(n_actions), 
#         sigma=0.1 * np.ones(n_actions)  # Small noise for controlled exploration
#     )
    
#     model = DDPG(
#         # ... other parameters ...
#         action_noise=action_noise,  # Add the noise
#         # ... rest unchanged ...
#     )
    
#     # ... rest unchanged ...

def train_ddpg_agent(config_path="SoccerEnv/field_config.yaml"):
    """Train DDPG agent with proper hyperparameters"""
    print("🚀 Training DDPG Agent...")

    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        field_type = config['field_type']
        print(f"📏 Using {field_type} field configuration")
    except:
        print("⚠️  Using default field configuration")
        config_path = None
    
    # Create environments
    train_env = create_monitored_env("easy", config_path)
    eval_env = create_monitored_env("easy", config_path)

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)  # 10% noise
    )

    # Create DDPG model
    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,      # Learning rate for DDPG (Used to be 1e-3)
        buffer_size=500000,      # Size of the replay buffer (Used to be 100000 then 200000 but larger buffer for better sampling)
        learning_starts=10000,    # How many steps to collect before training, to encourage more initial exploration (Used to be 2000 then 5000)
        batch_size=256,          # Batch size for training (used to be 128 but larger batch for better stability)
        tau=0.01,               # Soft update coefficient for target networks (Used to be 0.005 but larger tau for faster target updates)
        gamma=0.99,              # Discount factor (Used to be 0.97)
        action_noise=action_noise,       # (Used to allow DDPG handle exploration by specfiying "None")
        train_freq=2,            # How often to update the model
        gradient_steps=1,        # How many gradient steps to take per update

        # Stability settings
        policy_kwargs=dict(
            net_arch=[256, 256, 128], # Larger network (was [64, 64])
            activation_fn=torch.nn.ReLU, # Just trying ReLU
        ),

        verbose=1,
        device="cpu"
    )
    
    # Progress tracking
    progress_cb = ProgressTracker(train_env, eval_env, verbose=1, config_path=config_path)

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=15000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    plotting_callback = RealtimePlottingCallback(plot_freq=2000, verbose=1)
    
    print("Starting DDPG training...")
    
    # Train the model
    try:
        total_timesteps = 250000 # Used to be 250000
        # Train the model
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, progress_cb, plotting_callback],
            progress_bar=True
        )
        training_time = time.time() - start_time

        # Save the final model
        timestamp = get_timestamp()
        model_name = "soccer_rl_ddpg_" + timestamp
        model.save(model_name)
        print(f"✅ DDPG training completed! Model saved as '{model_name}'")
        print(f"⏱️  Training completed in {training_time:.2f} seconds")

        # Save training plots
        plotting_callback.save_plots("ddpg_training_progress_" + timestamp)

        final_results = {}
        for diff in ["easy", "medium", "hard"]:
            test_env = create_monitored_env(diff, config_path)
            mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
            final_results[diff] = (mean_reward, std_reward)
            print(f"  {diff.title()}: {mean_reward:.2f} ± {std_reward:.2f}")
            test_env.close()
        
        return model, final_results
    
    except Exception as e:
        print(f"❌ DDPG training error: {e}")
        timestamp = get_timestamp()
        model_name = "soccer_rl_ddpg_partial_" + timestamp
        model.save(model_name)
        plotting_callback.save_plots("ddpg_partial_training_progress_" + timestamp)
        return None, None

    finally:
        train_env.close()
        eval_env.close()

def evaluate_and_compare(config_path="SoccerEnv/field_config.yaml"):
    """Evaluate and compare both models"""
    print("\n🔍 EVALUATING TRAINED MODELS...")
    
    results = {}
    models = [
        ("PPO", "soccer_rl_ppo_final"),
        ("DDPG", "soccer_rl_ddpg_final")
    ]
    
    for model_name, model_file in models:
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            # Load model
            if model_name == "PPO":
                model = PPO.load(model_file)
            else:
                model = DDPG.load(model_file)
            
            # Test on different difficulties
            difficulty_results = {}
            
            for difficulty in ["easy", "medium", "hard"]:
                print(f"  Testing on {difficulty} difficulty...")
                
                # Create test environment
                test_env = SoccerEnv(difficulty=difficulty, config_path=config_path)
                
                episode_rewards = []
                episode_lengths = []
                goals_scored = 0
                
                # Run test episodes
                for episode in range(10):
                    obs, _ = test_env.reset()
                    episode_reward = 0
                    steps = 0
                    
                    for step in range(test_env.max_steps):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = test_env.step(action)
                        episode_reward += reward
                        steps += 1
                        
                        if terminated:  # Goal scored
                            goals_scored += 1
                            break
                        elif truncated:
                            break
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(steps)
                
                test_env.close()
                
                # Store results
                difficulty_results[difficulty] = {
                    'avg_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'avg_length': np.mean(episode_lengths),
                    'goals_scored': goals_scored,
                    'success_rate': (goals_scored / 10) * 100
                }
                
                print(f"    Avg Reward: {difficulty_results[difficulty]['avg_reward']:.2f}")
                print(f"    Success Rate: {difficulty_results[difficulty]['success_rate']:.1f}%")
            
            results[model_name] = difficulty_results
            
        except FileNotFoundError:
            print(f"❌ {model_name} model file not found: {model_file}")
            results[model_name] = None
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("📊 FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n🎯 {difficulty.upper()} Difficulty:")
        print(f"{'Model':<8} {'Avg Reward':<12} {'Success Rate':<15} {'Avg Length':<12}")
        print("-" * 50)
        
        for model_name in ["PPO", "DDPG"]:
            if results[model_name] and difficulty in results[model_name]:
                data = results[model_name][difficulty]
                print(f"{model_name:<8} {data['avg_reward']:<12.2f} {data['success_rate']:<15.1f}% {data['avg_length']:<12.1f}")
            else:
                print(f"{model_name:<8} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    return results

# def plot_comparison(results):
#     """Create comparison plot"""
#     # Filter out None results
#     valid_results = {k: v for k, v in results.items() if v is not None}
    
#     if len(valid_results) == 0:
#         print("❌ No valid results to plot!")
#         return
    
#     algorithms = list(valid_results.keys())
#     means = [valid_results[algo][0] for algo in algorithms]
#     stds = [valid_results[algo][1] for algo in algorithms]
    
#     # Create plot
#     plt.figure(figsize=(10, 6))
#     colors = ['red', 'blue', 'green']
#     bars = plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7, 
#                    color=colors[:len(algorithms)])
    
#     plt.title("RL Algorithm Performance Comparison\nSoccer Environment", 
#               fontsize=14, fontweight='bold')
#     plt.ylabel("Average Episode Reward", fontsize=12)
#     plt.xlabel("Algorithm", fontsize=12)
#     plt.grid(axis='y', alpha=0.3)
    
#     # Add value labels on bars
#     for i, (mean, std) in enumerate(zip(means, stds)):
#         plt.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}', 
#                 ha='center', va='bottom', fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig("soccer_rl_comparison.png", dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("📈 Comparison plot saved as 'soccer_rl_comparison.png'")

def create_comparison_plots(ppo_results, ddpg_results):
    """Create comprehensive comparison plots"""
    print("📈 Creating final comparison plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PPO vs DDPG: Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
    
    difficulties = ['easy', 'medium', 'hard']
    x_pos = np.arange(len(difficulties))
    width = 0.35
    
    # Extract data
    ppo_rewards = [ppo_results[d]['avg_reward'] if ppo_results else 0 for d in difficulties]
    ddpg_rewards = [ddpg_results[d]['avg_reward'] if ddpg_results else 0 for d in difficulties]
    ppo_success = [ppo_results[d]['success_rate'] if ppo_results else 0 for d in difficulties]
    ddpg_success = [ddpg_results[d]['success_rate'] if ddpg_results else 0 for d in difficulties]
    
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
    ax2.set_title('Goal Scoring Success Rate by Difficulty')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(difficulties)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance Radar Chart (if both models available)
    if ppo_results and ddpg_results:
        # Simple performance comparison
        metrics = ['Easy Reward', 'Medium Reward', 'Hard Reward', 'Avg Success']
        ppo_metrics = ppo_rewards + [np.mean(ppo_success)]
        ddpg_metrics = ddpg_rewards + [np.mean(ddpg_success)]
        
        x_metrics = np.arange(len(metrics))
        ax3.plot(x_metrics, ppo_metrics, 'o-', label='PPO', linewidth=2, markersize=8)
        ax3.plot(x_metrics, ddpg_metrics, 's-', label='DDPG', linewidth=2, markersize=8)
        ax3.set_xticks(x_metrics)
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.set_title('Overall Performance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax4.axis('off')
    summary_text = "Training Summary:\n\n"
    summary_text += f"✓ Total training time: ~{400000/1000:.0f}k timesteps\n"
    summary_text += f"✓ Progressive difficulty: Easy → Medium → Hard\n"
    summary_text += f"✓ Real-time plotting enabled\n"
    summary_text += f"✓ RoboCup field dimensions used\n\n"
    
    if ppo_results and ddpg_results:
        ppo_avg = np.mean([ppo_results[d]['avg_reward'] for d in difficulties])
        ddpg_avg = np.mean([ddpg_results[d]['avg_reward'] for d in difficulties])
        better_model = "PPO" if ppo_avg > ddpg_avg else "DDPG"
        summary_text += f"🏆 Better overall performer: {better_model}\n"
        summary_text += f"   PPO average: {ppo_avg:.2f}\n"
        summary_text += f"   DDPG average: {ddpg_avg:.2f}"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    filename = "final_model_comparison_" + get_timestamp()
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Final comparison plots saved as {filename}.png'")



def main():
    """Main training function"""
    print("🤖 Soccer RL Training System")
    print("=" * 50)
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check if field configuration exists
    config_path = "SoccerEnv/field_config.yaml"
    if not os.path.exists(config_path):
        print(f"⚠️  Warning: {config_path} not found!")
        print("Please create the field configuration file first.")
        print("The system will use default configurations.")
        config_path = None
    else:
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"✅ Loaded field configuration: {config['field_type']} field")
            field_dims = config['real_world_dimensions'][config['field_type']]
            print(f"📏 Field dimensions: {field_dims['field_length']}m x {field_dims['field_width']}m")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            config_path = None

    print("\nTraining Options:")
    print("1. Train PPO only")
    print("2. Train DDPG only")
    print("3. Full pipeline (train both + evaluate)")
    print("4. Evaluate existing models")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("🚀 Training PPO agent with real-time plotting...")
        model = train_ppo_agent(config_path)
        if model:
            print("✅ PPO training completed successfully!")
        
    elif choice == "2":
        print("🚀 Training DDPG agent with real-time plotting...")
        model, results = train_ddpg_agent(config_path)
        if model:
            print("✅ DDPG training completed successfully!")
            
    elif choice == "3":
        print("🚀 Running Full Training Pipeline...")
        print("This will take 20-40 minutes depending on your computer.")
        print("You can watch the real-time training plots - look for increasing rewards!")
        
        # Train both models
        print("\n" + "="*50)
        print("PHASE 1: TRAINING PPO AGENT")
        print("="*50)
        # ppo_model, ppo_results = train_ppo_agent(config_path)
        ppo_model = train_ppo_agent(config_path)

        
        print("\n" + "="*50)
        print("PHASE 2: TRAINING DDPG AGENT")
        print("="*50)
        # ddpg_model, ddpg_results = train_ddpg_agent(config_path)
        ddpg_model = train_ddpg_agent(config_path)

        print("\n" + "="*50)
        print("PHASE 3: EVALUATION AND COMPARISON")
        print("="*50)
        
        # Evaluate and compare if both succeeded
        if ppo_model and ddpg_model:
            results = evaluate_and_compare(config_path)
            
            # Create comprehensive comparison plots
            if results.get("PPO") and results.get("DDPG"):
                create_comparison_plots(results["PPO"], results["DDPG"])
        
        print("\n🎉 TRAINING PIPELINE COMPLETE!")
        print("Models saved:")
        if ppo_model:
            print("- soccer_rl_ppo_final.zip")
        if ddpg_model:
            print("- soccer_rl_ddpg_final.zip")
        print("\nPlots saved:")
        print("- ppo_training_progress.png")
        print("- ddpg_training_progress.png")
        print("- final_model_comparison.png")
        print("\nNext step: Use test_trained_model.py to watch them play!")
        
    elif choice == "4":
        print("📊 Evaluating existing models...")
        results = evaluate_and_compare(config_path)
        
        # Create comparison plots if both models exist
        if results.get("PPO") and results.get("DDPG"):
            create_comparison_plots(results["PPO"], results["DDPG"])
        
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()