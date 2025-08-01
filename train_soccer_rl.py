# train_soccer_rl.py - For RL training
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from SoccerEnv.soccerenv import SoccerEnv
import os

def create_monitored_env():
    """Create environment with monitoring wrapper"""
    env = SoccerEnv()
    env = Monitor(env)  # This wrapper logs episode rewards and lengths
    return env

def train_ppo_agent():
    """Train PPO agent with proper hyperparameters"""
    print("🚀 Training PPO Agent...")
    
    # Create training and evaluation environments
    train_env = create_monitored_env()
    eval_env = create_monitored_env()
    
    # Create model with hyperparameters for continuous control
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,      # Standard learning rate for PPO (First I put 3e-4 but too slow to improve then 1e-3 but still too long)
        n_steps=1024,            # Number of steps to run for each environment per update (USed to be 2048)
        batch_size=64,           # Minibatch size
        n_epochs=5,              # Number of epochs when optimising the surrogate loss (used to be 10)
        gamma=0.97,              # Discount factor (Used to be 0.99)
        gae_lambda=0.95,         # Factor for trade-off of bias vs variance for GAE
        clip_range=0.15,          # Clipping parameter (Used to be 0.2)
        ent_coef=0.1,           # Entropy coefficient for exploration (used to be 0.05)
        vf_coef=0.5,             # Value function coefficient
        max_grad_norm=0.5,       # Gradient clipping
        verbose=1,               # Print training progress
        device="cpu"             # Use GPU (CPU more stable for small envs but trying out GPU if possible)
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,          # Evaluate every 5000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    print("Starting PPO training... This may take 5-15 minutes depending on your computer.")
    print("Watch for increasing episode rewards - that means learning is happening!")
    
    #TODO: Once the model is proving effective on small batches, increase the number of timesteps for both algos PPO and DDPG
    # Train the model - (50000 for quick testing) but 100000 timesteps for better learning but to check if things are going in the right direction
    total_timesteps = 50000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("soccer_rl_ppo_final")
    print("✅ PPO training completed! Model saved as 'soccer_rl_ppo_final'")
    
    # Evaluation on randomised scenarios
    print("\n📊 Evaluating on randomised scenarios...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Performance on random scenarios: {mean_reward:.2f} ± {std_reward:.2f}")
    
    train_env.close()
    eval_env.close()
    
    return model

def train_ddpg_agent():
    """Train DDPG agent with proper hyperparameters"""
    print("🚀 Training DDPG Agent...")
    
    # Create environments
    train_env = create_monitored_env()
    eval_env = create_monitored_env()
    
    # Create DDPG model
    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,      # Learning rate for DDPG
        buffer_size=100000,      # Size of the replay buffer
        learning_starts=1000,    # How many steps to collect before training
        batch_size=128,          # Batch size for training
        tau=0.005,               # Soft update coefficient for target networks
        gamma=0.97,              # Discount factor (Used to be 0.99)
        action_noise=None,       # Let DDPG handle exploration
        verbose=1,
        device="cpu"
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    print("Starting DDPG training...")
    
    # Train the model
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
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"📊 Final DDPG Performance: {mean_reward:.2f} ± {std_reward:.2f}")
    
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

# # train_soccer_rl.py - Train RL agents on the simple soccer environment
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3 import PPO, DDPG
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback
# from soccerenv import SoccerEnv
# import gymnasium as gym

# def create_monitored_env():
#     """Create environment with monitoring wrapper"""
#     env = SoccerEnv()
#     env = Monitor(env)  # This wrapper logs episode rewards and lengths
#     return env

# def train_and_compare_algorithms():
#     """Train multiple RL algorithms and compare performance"""
    
#     print("🚀 Starting RL Training for Soccer Environment...")
    
#     # Create environments
#     def make_env():
#         return SoccerEnv()
    
#     # Results storage
#     results = {}
#     models = {}
    
#     # Training parameters
#     total_timesteps = 50000
#     n_eval_episodes = 20
    
#     # Algorithms to test (as per your FYP requirements)
#     algorithms = {
#         'PPO': PPO,
#         'DDPG': DDPG
#     }
    
#     for algo_name, algo_class in algorithms.items():
#         print(f"\n🧠 Training {algo_name}...")
        
#         # Create environment
#         env = make_env()
        
#         # Create model
#         if algo_name == 'DQN':
#             # DQN requires discrete actions, so we'll skip it for now
#             # since our environment has continuous actions
#             print(f"⚠️ Skipping {algo_name} - requires discrete actions")
#             continue
        
#         if algo_name == 'DDPG':
#             model = algo_class("MlpPolicy", env, verbose=1, learning_rate=0.001)
#         else:  # PPO
#             model = algo_class("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        
#         # Train model
#         model.learn(total_timesteps=total_timesteps)
        
#         # Evaluate model
#         print(f"📊 Evaluating {algo_name}...")
#         mean_reward, std_reward = evaluate_policy(
#             model, env, n_eval_episodes=n_eval_episodes, deterministic=True
#         )
        
#         # Store results
#         results[algo_name] = (mean_reward, std_reward)
#         models[algo_name] = model
        
#         print(f"✅ {algo_name} Results: {mean_reward:.2f} ± {std_reward:.2f}")
        
#         # Save model
#         model.save(f"soccer_rl_{algo_name.lower()}")
        
#         env.close()
    
#     # Plot comparison
#     if results:
#         plot_results(results)
    
#     return results, models

# def plot_results(results):
#     """Plot algorithm comparison"""
#     algorithms = list(results.keys())
#     means = [results[algo][0] for algo in algorithms]
#     stds = [results[algo][1] for algo in algorithms]
    
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7)
#     plt.title("RL Algorithm Performance Comparison\nSimple Soccer Environment")
#     plt.ylabel("Mean Episode Reward")
#     plt.grid(True, alpha=0.3)
    
#     # Add value labels on bars
#     for bar, mean, std in zip(bars, means, stds):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
#                 f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig("soccer_rl_comparison.png", dpi=300, bbox_inches='tight')
#     plt.show()

# def test_trained_model(model_name="soccer_rl_ppo"):
#     """Test a trained model"""
#     print(f"🎮 Testing trained model: {model_name}")
    
#     try:
#         # Load model
#         if "ppo" in model_name.lower():
#             model = PPO.load(model_name)
#         elif "ddpg" in model_name.lower():
#             model = DDPG.load(model_name)
#         else:
#             print("❌ Unknown model type")
#             return
        
#         # Create environment with rendering
#         env = SimpleSoccerEnv(render_mode="human")
        
#         # Test episodes
#         for episode in range(5):
#             obs, _ = env.reset()
#             total_reward = 0
#             steps = 0
            
#             print(f"\n🎯 Episode {episode + 1}")
            
#             for step in range(1000):
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, reward, terminated, truncated, _ = env.step(action)
#                 total_reward += reward
#                 steps += 1
                
#                 if terminated or truncated:
#                     break
            
#             print(f"Episode {episode + 1}: {total_reward:.2f} reward, {steps} steps")
            
#         env.close()
        
#     except FileNotFoundError:
#         print(f"❌ Model {model_name} not found. Train a model first!")

# def benchmark_performance():
#     """Benchmark performance metrics for FYP evaluation"""
#     print("📈 Running Performance Benchmark...")
    
#     # Create environment
#     env = SimpleSoccerEnv()
    
#     # Test random policy (baseline)
#     print("Testing random policy...")
#     random_rewards = []
#     random_success_rate = 0
    
#     for episode in range(100):
#         obs, _ = env.reset()
#         episode_reward = 0
#         success = False
        
#         for step in range(1000):
#             action = env.action_space.sample()
#             obs, reward, terminated, truncated, _ = env.step(action)
#             episode_reward += reward
            
#             if terminated:
#                 # Check if it's a success (ball in goal)
#                 if env.ball_pos[0] > 340 and 160 < env.ball_pos[1] < 240:
#                     success = True
#                 break
            
#             if truncated:
#                 break
        
#         random_rewards.append(episode_reward)
#         if success:
#             random_success_rate += 1
    
#     random_success_rate /= 100
    
#     print(f"🎲 Random Policy Results:")
#     print(f"   Mean reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
#     print(f"   Success rate: {random_success_rate:.1%}")
    
#     # Try to test trained model if available
#     try:
#         model = PPO.load("soccer_rl_ppo")
#         print("\nTesting trained PPO model...")
        
#         trained_rewards = []
#         trained_success_rate = 0
        
#         for episode in range(100):
#             obs, _ = env.reset()
#             episode_reward = 0
#             success = False
            
#             for step in range(1000):
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, reward, terminated, truncated, _ = env.step(action)
#                 episode_reward += reward
                
#                 if terminated:
#                     if env.ball_pos[0] > 340 and 160 < env.ball_pos[1] < 240:
#                         success = True
#                     break
                
#                 if truncated:
#                     break
            
#             trained_rewards.append(episode_reward)
#             if success:
#                 trained_success_rate += 1
        
#         trained_success_rate /= 100
        
#         print(f"🧠 Trained PPO Results:")
#         print(f"   Mean reward: {np.mean(trained_rewards):.2f} ± {np.std(trained_rewards):.2f}")
#         print(f"   Success rate: {trained_success_rate:.1%}")
        
#         # Calculate improvement
#         improvement = (np.mean(trained_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100
#         print(f"   Improvement: {improvement:.1f}%")
        
#         # Check FYP success criteria
#         print(f"\n📊 FYP Success Criteria Check:")
#         print(f"   ✅ Success rate > 60%: {trained_success_rate > 0.6}")
#         print(f"   ✅ Improvement > 15%: {improvement > 15}")
        
#     except FileNotFoundError:
#         print("No trained model found. Run training first!")
    
#     env.close()

# if __name__ == "__main__":
#     # Run training
#     results, models = train_and_compare_algorithms()
    
#     # Run benchmark
#     benchmark_performance()
    
#     print("\n🎉 Training complete! Next steps:")
#     print("1. Run: python train_soccer_rl.py")
#     print("2. Test model: test_trained_model()")
#     print("3. Convert to ONNX for deployment")