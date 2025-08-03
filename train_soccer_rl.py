# train_soccer_rl.py - For RL training
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from SoccerEnv.soccerenv import SoccerEnv
import os
import torch

class ProgressTracker(BaseCallback):
    """Track learning progress and switch difficulty for both training and evaluation"""
    
    def __init__(self, train_env_wrapper, eval_env_wrapper):
        super().__init__()
        self.train_env_wrapper = train_env_wrapper
        self.eval_env_wrapper = eval_env_wrapper
        self.episode_rewards = []
        self.last_20_rewards = []
        self.step_count = 0
        self.current_difficulty = "easy"
        
    def _on_step(self):
        self.step_count += 1
        
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            episode_reward = self.locals.get('infos', [{}])[0].get('episode', {}).get('r', 0)
            if episode_reward != 0:
                self.episode_rewards.append(episode_reward)
                self.last_20_rewards.append(episode_reward)
                if len(self.last_20_rewards) > 20:
                    self.last_20_rewards.pop(0)
        
        # Print progress every 10000 steps
        if self.step_count % 10000 == 0 and len(self.last_20_rewards) >= 10:
            avg_reward = np.mean(self.last_20_rewards)
            print(f"\n📈 Step {self.step_count}: Recent avg reward = {avg_reward:.2f}")
            
            # Curriculum progression
            if self.current_difficulty == "easy" and avg_reward > 6:
                print("🎯 PROGRESSING TO MEDIUM DIFFICULTY!")
                self.current_difficulty = "medium"
                
                # Update BOTH environments
                self.train_env_wrapper.env.difficulty = "medium"
                self.train_env_wrapper.env.max_steps = 200
                self.train_env_wrapper.env.possession_distance = 40.0
                self.train_env_wrapper.env.collision_distance = 25.0
                
                self.eval_env_wrapper.env.difficulty = "medium"
                self.eval_env_wrapper.env.max_steps = 200
                self.eval_env_wrapper.env.possession_distance = 40.0
                self.eval_env_wrapper.env.collision_distance = 25.0
            elif self.current_difficulty == "medium" and avg_reward > 10:
                print("🎯 PROGRESSING TO HARD DIFFICULTY!")  
                self.current_difficulty = "hard"
                
                # Update BOTH environments
                self.train_env_wrapper.env.difficulty = "hard"
                self.train_env_wrapper.env.max_steps = 250
                self.train_env_wrapper.env.possession_distance = 35.0
                self.train_env_wrapper.env.collision_distance = 30.0
                
                self.eval_env_wrapper.env.difficulty = "hard"
                self.eval_env_wrapper.env.max_steps = 250
                self.eval_env_wrapper.env.possession_distance = 35.0
                self.eval_env_wrapper.env.collision_distance = 30.0
        
        return True
    
def create_monitored_env(difficulty="easy"):
    """Create environment with monitoring wrapper"""
    env = SoccerEnv(difficulty=difficulty)
    env = Monitor(env)  # This wrapper logs episode rewards and lengths
    return env

def train_ppo_agent():
    """Train PPO agent with proper hyperparameters"""
    print("🚀 Training PPO Agent...")
    
    # Create training and evaluation environments starting with "easy" difficulty
    train_env = create_monitored_env("easy")
    eval_env = create_monitored_env("easy")
    
    # Create model with hyperparameters for continuous control
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,      # Learning rate for PPO (First I put 3e-4 but too slow to improve then 1e-3 but still too long)

        # Training stability
        n_steps=2048,            # Number of steps to run for each environment per update (USed to be 2048)
        batch_size=64,           # Minibatch size
        n_epochs=5,              # Number of epochs when optimising the surrogate loss (used to be 10)
        
        # Exploration VS Exploitation
        gamma=0.97,              # Discount factor (Used to be 0.99)
        gae_lambda=0.95,         # Factor for trade-off of bias vs variance for GAE
        clip_range=0.15,          # Clipping parameter (Used to be 0.2)
        ent_coef=0.1,           # Entropy coefficient for exploration (used to be 0.05)
        
        # Stability settings
        vf_coef=0.5,             # Value function coefficient
        max_grad_norm=0.5,       # Gradient clipping

        # Network architecture for stability
        policy_kwargs=dict(
            net_arch=[64, 64],           # Smaller, more stable network
            activation_fn=torch.nn.Tanh, # More stable than ReLU
        ),

        verbose=1,               # Print training progress
        device="cpu"             # Use GPU (CPU more stable for small envs but trying out GPU if possible)
    )
    
    # Progress tracking
    progress_cb = ProgressTracker(train_env, eval_env)

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=15000,          # Evaluate every 15000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    print("Starting PPO training... This may take 5-15 minutes depending on your computer.")
    print("Watch for increasing episode rewards - that means learning is happening!")
    
    #TODO: Once the model is proving effective on small batches, increase the number of timesteps for both algos PPO and DDPG
    # Train the model - (50000 for quick testing) but 100000 timesteps for better learning but to check if things are going in the right direction
    total_timesteps = 250000
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_cb, eval_callback],
            progress_bar=True
        )
        
        # Save the final model
        model.save("soccer_rl_ppo_final")
        print("✅ PPO training completed! Model saved as 'soccer_rl_ppo_final'")

        
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
        easy_score = final_results["easy"][0]
        hard_score = final_results["hard"][0]
        
        if easy_score > hard_score:
            print("✅ Curriculum working! Robot performs better on easier difficulties")
        else:
            print("⚠️  Curriculum might need tuning - similar performance across difficulties")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        # Save partial progress
        model.save("soccer_rl_ppo_partial")
        print("💾 Partial model saved")
    
    finally:
        train_env.close()
        eval_env.close()
    
    return model

def train_ddpg_agent():
    """Train DDPG agent with proper hyperparameters"""
    print("🚀 Training DDPG Agent...")
    
    # Create environments
    train_env = create_monitored_env("medium")
    eval_env = create_monitored_env("medium")
    
    # Create DDPG model
    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,      # Learning rate for DDPG
        buffer_size=100000,      # Size of the replay buffer
        learning_starts=2000,    # How many steps to collect before training, to encourage more initial exploration
        batch_size=128,          # Batch size for training
        tau=0.005,               # Soft update coefficient for target networks
        gamma=0.97,              # Discount factor (Used to be 0.99)
        action_noise=None,       # Let DDPG handle exploration
        
        # Stability settings
        policy_kwargs=dict(
            net_arch=[64, 64],   # Smaller network
        ),

        verbose=1,
        device="cpu"
    )
    
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
    
    print("Starting DDPG training...")
    
    # Train the model
    try:
        total_timesteps = 250000
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save the final model
        model.save("soccer_rl_ddpg_final")
        print("✅ DDPG training completed! Model saved as 'soccer_rl_ddpg_final'")
        # Quick evaluation
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"📊 Final DDPG Performance: {mean_reward:.2f} ± {std_reward:.2f}")
    
    except Exception as e:
        print(f"❌ DDPG training error: {e}")
        model.save("soccer_rl_ddpg_partial")

    finally:
        train_env.close()
        eval_env.close()
    
    return model

def evaluate_and_compare():
    """Evaluate trained models and compare with random baseline"""
    print("📊 Evaluating All Models...")
    
    # Test environment
    eval_env = create_monitored_env()
    
    results = {}
    
    # 1. Test Random Baseline
    print("\n🎲 Testing Random Baseline...")
    random_rewards = []
    for episode in range(20):
        obs, _ = eval_env.reset()
        episode_reward = 0
        for step in range(1000):
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)
    
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    results["Random"] = (random_mean, random_std)
    print(f"Random Policy: {random_mean:.2f} ± {random_std:.2f}")
    
    # 2. Test PPO Model
    try:
        print("\n🧠 Testing PPO Model...")
        ppo_model = PPO.load("soccer_rl_ppo_final")
        mean_reward, std_reward = evaluate_policy(ppo_model, eval_env, n_eval_episodes=20)
        results["PPO"] = (mean_reward, std_reward)
        print(f"PPO Model: {mean_reward:.2f} ± {std_reward:.2f}")
        
        improvement = ((mean_reward - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else 0
        print(f"PPO Improvement over Random: {improvement:.1f}%")
        
    except FileNotFoundError:
        print("❌ PPO model not found!")
        results["PPO"] = None
    
    # 3. Test DDPG Model
    try:
        print("\n🧠 Testing DDPG Model...")
        ddpg_model = DDPG.load("soccer_rl_ddpg_final")
        mean_reward, std_reward = evaluate_policy(ddpg_model, eval_env, n_eval_episodes=20)
        results["DDPG"] = (mean_reward, std_reward)
        print(f"DDPG Model: {mean_reward:.2f} ± {std_reward:.2f}")
        
        improvement = ((mean_reward - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else 0
        print(f"DDPG Improvement over Random: {improvement:.1f}%")
        
    except FileNotFoundError:
        print("❌ DDPG model not found!")
        results["DDPG"] = None
    
    eval_env.close()
    
    # Plot results
    plot_comparison(results)
    
    return results

def plot_comparison(results):
    """Create comparison plot"""
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) == 0:
        print("❌ No valid results to plot!")
        return
    
    algorithms = list(valid_results.keys())
    means = [valid_results[algo][0] for algo in algorithms]
    stds = [valid_results[algo][1] for algo in algorithms]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    bars = plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=colors[:len(algorithms)])
    
    plt.title("RL Algorithm Performance Comparison\nSoccer Environment", 
              fontsize=14, fontweight='bold')
    plt.ylabel("Average Episode Reward", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("soccer_rl_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Comparison plot saved as 'soccer_rl_comparison.png'")

def main():
    """Main training function"""
    print("🤖 Soccer RL Training System")
    print("=" * 50)
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\nTraining Options:")
    print("1. Train PPO only")
    print("2. Train DDPG only")
    print("3. Train both algorithms")
    print("4. Evaluate existing models")
    print("5. Full pipeline (train both + evaluate)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        train_ppo_agent()
        
    elif choice == "2":
        train_ddpg_agent()
        
    elif choice == "3":
        print("Training both algorithms...")
        train_ppo_agent()
        print("\n" + "="*50)
        train_ddpg_agent()
        
    elif choice == "4":
        evaluate_and_compare()
        
    elif choice == "5":
        print("🚀 Running Full Training Pipeline...")
        print("This will take 20-40 minutes depending on your computer.")
        print("You can watch the training progress - look for increasing rewards!")
        
        # Train both models
        train_ppo_agent()
        print("\n" + "="*50)
        train_ddpg_agent()
        
        print("\n" + "="*50)
        print("Training complete! Now evaluating...")
        
        # Evaluate and compare
        results = evaluate_and_compare()
        
        print("\n🎉 TRAINING PIPELINE COMPLETE!")
        print("Models saved:")
        print("- soccer_rl_ppo_final.zip")
        print("- soccer_rl_ddpg_final.zip")
        print("\nNext step: Use test_trained_model.py to watch them play!")
        
    else:
        print("Invalid choice. Training PPO by default...")
        train_ppo_agent()

if __name__ == "__main__":
    main()