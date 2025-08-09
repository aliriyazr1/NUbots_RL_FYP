"""
Author: Ali Riyaz
Student Number: C3412624
Last Updated: 09/08/2025
"""

# test_trained_model.py - Watch the trained robot play by loading the correct models
import time
from stable_baselines3 import PPO, DDPG
from SoccerEnv.soccerenv import SoccerEnv
import numpy as np

def watch_trained_robot(model_name="soccer_rl_ppo_final", episodes=5, difficulty="medium"):
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
            print("❌ Unknown model type. Use 'soccer_rl_ppo_final' or 'soccer_rl_ddpg_final'")
            return
        
        # Create environment with human rendering
        env = SoccerEnv(render_mode="human", difficulty=difficulty)
        print("🎯 Starting visual test...")
        print("Watch the blue robot (your AI) try to get the white ball to the yellow goal!")
        print("Red robot is the opponent. Close the window to stop.\n")
        
        successful_episodes = 0
        total_rewards = []
        got_ball_episodes = 0
        
        for episode in range(episodes):
            print(f"\n🏁 Episode {episode + 1}/{episodes}")
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_successful = False
            
            # Track if robot gets ball and reaches goal
            had_ball = False
            reached_goal = False
            max_progress_to_goal = 0
            ball_possession_time = 0
            
            for step in range(env.max_steps):  # Used to be Max 1000 steps per episode
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
                    ball_possession_time += 1
                
                # Check if reached goal area
                robot_x = (obs[0] * 200) + 200  # Convert normalised obs back to pixel coordinates
                progress_to_goal = max(0, robot_x - 100) / 300  # Progress from left side to goal
                max_progress_to_goal = max(max_progress_to_goal, progress_to_goal)

                if robot_x > 320:  # Near goal
                    reached_goal = True
                
                # Add small delay to see what is going on
                time.sleep(0.03)  # 33 FPS
                
                if terminated:
                    # Check if it's a successful termination (ball in goal)
                    if env._check_goal():
                        episode_successful = True
                        successful_episodes += 1
                        print(f"   🎉 SUCCESS! Robot scored a goal!")
                    else:
                        print(f"   💥 Episode ended (collision or failure)")
                    break
                
                if truncated:
                    print(f"   ⏰ Episode timeout (reached {env.max_steps} steps)")
                    break
            
            total_rewards.append(total_reward)
            if had_ball:
                got_ball_episodes += 1
            
            # Episode summary
            print(f"   📊 Reward: {total_reward:.2f}")
            print(f"   👟 Steps: {steps}")
            print(f"   ⚽ Had ball: {'Yes' if had_ball else 'No'}")
            print(f"   🥅 Max progress to goal: {max_progress_to_goal*100:.1f}%")
            print(f"   ⏱️  Ball possession time: {ball_possession_time} steps")            
            # Wait a moment between episodes
            time.sleep(1.5)
        
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
            result = watch_trained_robot(model_name, episodes=3, difficulty=difficulty)
            
            if result:
                model_results[difficulty] = result
                print(f"   Summary: {result['success_rate']:.1f}% goals, {result['avg_reward']:.1f} avg reward")
            
            if len(difficulties) > 1:  # Only pause if testing multiple difficulties
                input("   Press Enter to continue...")
        results[model_name] = model_results
        
        if len(models_to_test) > 1:  # Only pause if testing multiple models
            input(f"\n✅ {model_name} testing complete. Press Enter for next model...")

    # Final comparison summary
    print(f"\n{'='*60}")
    print("🏆 FINAL MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")

    for model_name, model_results in results.items():
        print(f"\n🧠 {model_name.upper()}:")
        for difficulty, result in model_results.items():
            print(f"   {difficulty.title()}: {result['success_rate']:.1f}% goals, {result['avg_reward']:.1f} reward")
    

def test_random_baseline():
    """Test random policy for comparison"""
    print("🎲 Testing random policy (baseline)...")
    
    env = SoccerEnv(render_mode="human")
    
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(env.max_steps): # Used to be 1000000
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

if __name__ == "__main__":
    print("🤖 Model Testing Options:")
    print("1. Test PPO model")
    print("2. Test DDPG model") 
    print("3. Compare both models")
    print("4. Test random baseline")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        watch_trained_robot("soccer_rl_ppo_final", episodes=5, difficulty="medium")
    elif choice == "2":
        watch_trained_robot("soccer_rl_ddpg_final", episodes=5, difficulty="medium")
    elif choice == "3":
        compare_models()
    elif choice == "4":
        test_random_baseline()
    # elif choice == "5":
        
    else:
        print("Invalid choice. Testing PPO by default...")
        watch_trained_robot("soccer_rl_ppo", episodes=5)