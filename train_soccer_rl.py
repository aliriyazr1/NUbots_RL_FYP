# train_soccer_rl.py - Train RL agents on the simple soccer environment
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from simplesoccerenv import SimpleSoccerEnv
import gymnasium as gym

def train_and_compare_algorithms():
    """Train multiple RL algorithms and compare performance"""
    
    print("🚀 Starting RL Training for Soccer Environment...")
    
    # Create environments
    def make_env():
        return SimpleSoccerEnv()
    
    # Results storage
    results = {}
    models = {}
    
    # Training parameters
    total_timesteps = 50000
    n_eval_episodes = 20
    
    # Algorithms to test (as per your FYP requirements)
    algorithms = {
        'PPO': PPO,
        'DQN': DQN,
        'DDPG': DDPG
    }
    
    for algo_name, algo_class in algorithms.items():
        print(f"\n🧠 Training {algo_name}...")
        
        # Create environment
        env = make_env()
        
        # Create model
        if algo_name == 'DQN':
            # DQN requires discrete actions, so we'll skip it for now
            # since our environment has continuous actions
            print(f"⚠️ Skipping {algo_name} - requires discrete actions")
            continue
        
        if algo_name == 'DDPG':
            model = algo_class("MlpPolicy", env, verbose=1, learning_rate=0.001)
        else:  # PPO
            model = algo_class("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        
        # Train model
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate model
        print(f"📊 Evaluating {algo_name}...")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        
        # Store results
        results[algo_name] = (mean_reward, std_reward)
        models[algo_name] = model
        
        print(f"✅ {algo_name} Results: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Save model
        model.save(f"soccer_rl_{algo_name.lower()}")
        
        env.close()
    
    # Plot comparison
    if results:
        plot_results(results)
    
    return results, models

def plot_results(results):
    """Plot algorithm comparison"""
    algorithms = list(results.keys())
    means = [results[algo][0] for algo in algorithms]
    stds = [results[algo][1] for algo in algorithms]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title("RL Algorithm Performance Comparison\nSimple Soccer Environment")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("soccer_rl_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def test_trained_model(model_name="soccer_rl_ppo"):
    """Test a trained model"""
    print(f"🎮 Testing trained model: {model_name}")
    
    try:
        # Load model
        if "ppo" in model_name.lower():
            model = PPO.load(model_name)
        elif "ddpg" in model_name.lower():
            model = DDPG.load(model_name)
        else:
            print("❌ Unknown model type")
            return
        
        # Create environment with rendering
        env = SimpleSoccerEnv(render_mode="human")
        
        # Test episodes
        for episode in range(5):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\n🎯 Episode {episode + 1}")
            
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1}: {total_reward:.2f} reward, {steps} steps")
            
        env.close()
        
    except FileNotFoundError:
        print(f"❌ Model {model_name} not found. Train a model first!")

def benchmark_performance():
    """Benchmark performance metrics for FYP evaluation"""
    print("📈 Running Performance Benchmark...")
    
    # Create environment
    env = SimpleSoccerEnv()
    
    # Test random policy (baseline)
    print("Testing random policy...")
    random_rewards = []
    random_success_rate = 0
    
    for episode in range(100):
        obs, _ = env.reset()
        episode_reward = 0
        success = False
        
        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated:
                # Check if it's a success (ball in goal)
                if env.ball_pos[0] > 340 and 160 < env.ball_pos[1] < 240:
                    success = True
                break
            
            if truncated:
                break
        
        random_rewards.append(episode_reward)
        if success:
            random_success_rate += 1
    
    random_success_rate /= 100
    
    print(f"🎲 Random Policy Results:")
    print(f"   Mean reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"   Success rate: {random_success_rate:.1%}")
    
    # Try to test trained model if available
    try:
        model = PPO.load("soccer_rl_ppo")
        print("\nTesting trained PPO model...")
        
        trained_rewards = []
        trained_success_rate = 0
        
        for episode in range(100):
            obs, _ = env.reset()
            episode_reward = 0
            success = False
            
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated:
                    if env.ball_pos[0] > 340 and 160 < env.ball_pos[1] < 240:
                        success = True
                    break
                
                if truncated:
                    break
            
            trained_rewards.append(episode_reward)
            if success:
                trained_success_rate += 1
        
        trained_success_rate /= 100
        
        print(f"🧠 Trained PPO Results:")
        print(f"   Mean reward: {np.mean(trained_rewards):.2f} ± {np.std(trained_rewards):.2f}")
        print(f"   Success rate: {trained_success_rate:.1%}")
        
        # Calculate improvement
        improvement = (np.mean(trained_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100
        print(f"   Improvement: {improvement:.1f}%")
        
        # Check FYP success criteria
        print(f"\n📊 FYP Success Criteria Check:")
        print(f"   ✅ Success rate > 60%: {trained_success_rate > 0.6}")
        print(f"   ✅ Improvement > 15%: {improvement > 15}")
        
    except FileNotFoundError:
        print("No trained model found. Run training first!")
    
    env.close()

if __name__ == "__main__":
    # Run training
    results, models = train_and_compare_algorithms()
    
    # Run benchmark
    benchmark_performance()
    
    print("\n🎉 Training complete! Next steps:")
    print("1. Run: python train_soccer_rl.py")
    print("2. Test model: test_trained_model()")
    print("3. Convert to ONNX for deployment")