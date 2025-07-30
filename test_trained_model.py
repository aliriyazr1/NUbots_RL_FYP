# test_trained_model.py - Watch your trained robot play!
import time
from stable_baselines3 import PPO, DDPG
from simplesoccerenv import SimpleSoccerEnv
import numpy as np

def watch_trained_robot(model_name="soccer_rl_ppo", episodes=5):
    """
    Watch your trained robot play soccer with visual rendering
    """
    print(f"🎮 Loading and testing model: {model_name}")
    
    try:
        # Load the trained model
        if "ppo" in model_name.lower():
            model = PPO.load(model_name)
            print("✅ PPO model loaded successfully!")
        elif "ddpg" in model_name.lower():
            model = DDPG.load(model_name)
            print("✅ DDPG model loaded successfully!")
        else:
            print("❌ Unknown model type. Use 'soccer_rl_ppo' or 'soccer_rl_ddpg'")
            return
        
        # Create environment with human rendering
        env = SimpleSoccerEnv(render_mode="human")
        print("🎯 Starting visual test...")
        print("Watch the blue robot (your AI) try to get the white ball to the yellow goal!")
        print("Red robot is the opponent. Close the window to stop.\n")
        
        successful_episodes = 0
        total_rewards = []
        
        for episode in range(episodes):
            print(f"\n🏁 Episode {episode + 1}/{episodes}")
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_successful = False
            
            # Track if robot gets ball and reaches goal
            had_ball = False
            reached_goal = False
            
            for step in range(1000):  # Max 1000 steps per episode
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                # Check progress
                has_ball_now = obs[11] > 0.5  # Ball possession flag
                if has_ball_now:
                    had_ball = True
                
                # Check if reached goal area
                robot_x = (obs[0] * 200) + 200  # Convert back to pixel coordinates
                if robot_x > 320:  # Near goal
                    reached_goal = True
                
                # Add small delay so you can see what's happening
                time.sleep(0.02)  # 50 FPS
                
                if terminated:
                    # Check if it's a successful termination (ball in goal)
                    ball_x = (obs[3] * 200) + 200
                    ball_y = (obs[4] * 200) + 200
                    if ball_x > 340 and 160 < ball_y < 240:
                        episode_successful = True
                        successful_episodes += 1
                        print(f"   🎉 SUCCESS! Robot scored a goal!")
                    else:
                        print(f"   💥 Episode ended (collision or failure)")
                    break
                
                if truncated:
                    print(f"   ⏰ Episode timeout (took too long)")
                    break
            
            total_rewards.append(total_reward)
            
            # Episode summary
            print(f"   📊 Reward: {total_reward:.2f}")
            print(f"   👟 Steps: {steps}")
            print(f"   ⚽ Had ball: {'Yes' if had_ball else 'No'}")
            print(f"   🥅 Reached goal area: {'Yes' if reached_goal else 'No'}")
            
            # Wait a moment between episodes
            time.sleep(1)
        
        env.close()
        
        # Final statistics
        print(f"\n📈 FINAL RESULTS:")
        print(f"   🏆 Successful episodes: {successful_episodes}/{episodes} ({successful_episodes/episodes*100:.1f}%)")
        print(f"   💯 Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   🎯 Best episode reward: {max(total_rewards):.2f}")
        print(f"   📉 Worst episode reward: {min(total_rewards):.2f}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        success_rate = successful_episodes/episodes*100
        avg_reward = np.mean(total_rewards)
        
        if success_rate >= 60:
            print(f"   ✅ EXCELLENT! Success rate {success_rate:.1f}% meets FYP requirement (>60%)")
        elif success_rate >= 30:
            print(f"   👍 GOOD! Success rate {success_rate:.1f}% is decent, might need more training")
        else:
            print(f"   ⚠️  NEEDS WORK! Success rate {success_rate:.1f}% is low, try more training or tune environment")
        
        if avg_reward > 10:
            print(f"   ✅ EXCELLENT! Average reward {avg_reward:.1f} is very good")
        elif avg_reward > 0:
            print(f"   👍 GOOD! Average reward {avg_reward:.1f} is positive")
        else:
            print(f"   ⚠️  NEEDS WORK! Average reward {avg_reward:.1f} is negative")
    
    except FileNotFoundError:
        print(f"❌ Model file '{model_name}.zip' not found!")
        print("Make sure you've run the training script first:")
        print("python train_soccer_rl.py")
        print("\nAvailable models should be:")
        print("- soccer_rl_ppo.zip")
        print("- soccer_rl_ddpg.zip")

def compare_models():
    """Compare PPO vs DDPG visually"""
    print("🥊 Comparing PPO vs DDPG models")
    print("Testing each model for 3 episodes...\n")
    
    models_to_test = ["soccer_rl_ppo", "soccer_rl_ddpg"]
    
    for model_name in models_to_test:
        print(f"{'='*50}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*50}")
        watch_trained_robot(model_name, episodes=3)
        
        input("\nPress Enter to test next model...")

def test_random_baseline():
    """Test random policy for comparison"""
    print("🎲 Testing random policy (baseline)...")
    
    env = SimpleSoccerEnv(render_mode="human")
    
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(1000000):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        time.sleep(0.02)  # Slow it down so you can watch
        
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"🎲 Random policy result:")
    print(f"   Reward: {total_reward:.2f}")
    print(f"   Steps: {steps}")
    print("This is what your trained model should beat!")

if __name__ == "__main__":
    print("🤖 Model Testing Options:")
    print("1. Test PPO model")
    print("2. Test DDPG model") 
    print("3. Compare both models")
    print("4. Test random baseline")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        watch_trained_robot("soccer_rl_ppo", episodes=5)
    elif choice == "2":
        watch_trained_robot("soccer_rl_ddpg", episodes=5)
    elif choice == "3":
        compare_models()
    elif choice == "4":
        test_random_baseline()
    else:
        print("Invalid choice. Testing PPO by default...")
        watch_trained_robot("soccer_rl_ppo", episodes=5)