"""
Author: Ali Riyaz
Student Number: C3412624
Last Updated: 20/09/2025
"""

# test_trained_model.py - Watch the trained robot play by loading the correct models
import time, yaml
from stable_baselines3 import PPO, DDPG
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.environments.soccerenv import SoccerEnv
from src.training.train_GUI import TrainGUI
import numpy as np
import matplotlib.pyplot as plt
from src.environments.soccerenv import ActionSmoothingWrapper

import traceback
import sys

def load_field_config(config_path="configs/field_config.yaml"):
    """Load and display field configuration"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        field_type = config['field_type']
        field_dims = config['real_world_dimensions'][field_type]
        
        print(f"📏 Field Configuration Loaded:")
        print(f"   Type: {field_type.title()}")
        print(f"   Dimensions: {field_dims['field_length']}m x {field_dims['field_width']}m")
        print(f"   Goal: {field_dims['goal_width']}m wide")
        
        return config_path
        
    except FileNotFoundError:
        print(f"⚠️  Warning: {config_path} not found! Using default configuration.")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}. Using default configuration.")
        return None
    
def watch_trained_robot(model_name, model_type=None, episodes=5, difficulty="medium", config_path="SoccerEnv/field_config.yaml", testing_mode=False):
    """
    Watch your trained robot play soccer with visual rendering
    """
    print(f"🎮 Loading and testing model: {model_name}")
    if testing_mode:
        print(f"🚀 Testing mode enabled - faster gameplay for better visibility")
    
    model_types = {"PPO": PPO, "DDPG": DDPG}    
    try:
        if not model_name:
            print("Invalid and empty model name!")
            return
        if model_type not in model_types:
            print("Unknown model type!")
            return
        
        # Load the trained model        
        ModelName = model_types[model_type]
        model = ModelName.load(model_name)

        smooth_model = ActionSmoothingWrapper(model, smoothing_factor=0.6)

        # Create environment with human rendering
        env = SoccerEnv(render_mode="human", difficulty=difficulty, config_path=config_path, testing_mode=testing_mode)
        env.dt *= 4.0  # TODO: REMOVE THIS WHEN DONE WITH TESTING. THis is Just to speed up the visualisation
        env.set_show_velocities(True)

        # Display field information
        field_config = env.field_config
        print(f"🎯 Field Type: {field_config.config['field_type'].title()}")
        print(f"📐 Real Dimensions: {field_config.real_dims['field_length']}m x {field_config.real_dims['field_width']}m")
        print(f"🖥️  Display Size: {field_config.field_width_pixels}x{field_config.field_height_pixels} pixels")
        print(f"⚽ Goal Width: {field_config.real_dims['goal_width']}m")
        
        print("🎯 Starting visual test...")
        print("Watch the blue robot (your AI) try to get the white ball to the yellow goal!")
        print("Red robot is the opponent.")
        print("Press Ctrl+C to stop early.\n")
        
        successful_episodes = 0
        total_rewards = []
        episode_lengths = []
        got_ball_episodes = 0
        goals_scored = 0
        total_steps = 0
        
        for episode in range(episodes):
            print(f"\n🏁 Episode {episode + 1}/{episodes}")
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            # Track if robot gets ball and reaches goal
            had_ball = False
            max_progress_to_goal = 0
            ball_possession_time = 0

            start_time = time.time()
            prev_ball_pos = env.ball_pos.copy()
            
            for step in range(env.max_steps):  # Max 1000 steps per episode
                # Get action from trained model
                # action, _ = model.predict(obs, deterministic=True)

                action, _ = smooth_model.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                total_steps += 1
                
                # Check progress
                has_ball_now = obs[11] > 0.5  # Ball possession flag
                if has_ball_now:
                    had_ball = True
                    ball_possession_time += 1
                
                # Check if reached goal area
                robot_x = (obs[0] * 200) + 200  # Convert normalised obs back to pixel coordinates
                progress_to_goal = max(0, robot_x - 100) / 300  # Progress from left side to goal
                max_progress_to_goal = max(max_progress_to_goal, progress_to_goal)

                env.render()
                # HACK: Adaptive sleep based on testing mode (Whole reason for increasing sleep time was to decrease frame rate for my laptop to be able to render the environment but check again)
                if testing_mode:
                    time.sleep(0.02)  # Faster for testing
                else:
                    time.sleep(0.05)  # Slower for detailed observation

                if step % 20 == 0:
                    elapsed = time.time() - start_time
                    ball_distance_moved = np.linalg.norm(env.ball_pos - prev_ball_pos)
                    print(f"Step {step}: Real time {elapsed:.1f}s, Ball moved {ball_distance_moved:.1f} pixels")
                # # Add small delay to see what is going on
                # time.sleep(0.05)  # 20 FPS

                if terminated:
                    # Check if it's a successful termination (ball in goal)
                    if env._check_goal():
                        successful_episodes += 1
                        goals_scored += 1
                        print(f"   🎉 SUCCESS! Robot scored a goal!")
                    else:
                        print(f"   💥 Episode ended (collision or failure)")
                    break
                
                if truncated:
                    print(f"   ⏰ Episode timeout (reached {env.max_steps} steps)")
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
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
        
        # Calculate and display statistics
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = (goals_scored / episodes) * 100
        
        print(f"\n{'='*50}")
        print(f"📊 TEST RESULTS for {model_name}")
        print(f"{'='*50}")
        print(f"Field Configuration: {field_config.config['field_type'].title()}")
        print(f"Difficulty: {difficulty.title()}")
        print(f"Episodes: {episodes}")
        print(f"Goals Scored: {goals_scored}/{episodes}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f} steps")
        print(f"Total Steps: {total_steps}")
        print(f"Episodes with Ball Contact: {got_ball_episodes}/{episodes}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if success_rate >= 70:
            print(f"   🌟 EXCELLENT! Success rate {success_rate:.1f}% is very high")
        elif success_rate >= 40:
            print(f"   👍 GOOD! Success rate {success_rate:.1f}% is solid")
        elif success_rate >= 20:
            print(f"   📈 DECENT! Success rate {success_rate:.1f}% shows learning")
        else:
            print(f"   📚 LEARNING! Success rate {success_rate:.1f}% is low, try more training or tune environment")
        
        if avg_reward > 10:
            print(f"   ✅ EXCELLENT! Average reward {avg_reward:.1f} is very good")
        elif avg_reward > 0:
            print(f"   👍 GOOD! Average reward {avg_reward:.1f} is positive")
        else:
            print(f"   ⚠️  NEEDS WORK! Average reward {avg_reward:.1f} is negative")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'avg_length': avg_length,
            'goals_scored': goals_scored,
            'episodes': episodes,
            'ball_contact_episodes': got_ball_episodes,
            'total_steps': total_steps
        }
    
    except FileNotFoundError:
        print(f"❌ Model file '{model_name}.zip' not found!")
        print("Make sure you've run the training script first:")
        print("python train_soccer_rl.py")
        print("\nAvailable models should be:")
        print("- soccer_rl_ppo_final.zip")
        print("- soccer_rl_ddpg_final.zip")
        return None
    except KeyboardInterrupt:
        print("\n⏹️  Testing stopped by user")
        if 'env' in locals():
            env.close()
        return None
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        traceback.print_exc(file=sys.stdout) # Prints the traceback to standard output
        
        if 'env' in locals():
            env.close()
        return None

def main():
    """Main function for model evaluation"""
    # Load field configuration
    config_path = load_field_config()

    print("🤖 Model Testing Options:")
    print("1. Test PPO model")
    print("2. Test DDPG model")
    #TODO: FIX THE DATA COLLECTION AND COMPARISON
    print("3. WIP: Compare both models")
    print("4. WIP: Detailed model comparison with statistical analysis")
    print("5. WIP: Test random baseline")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        model_name = input("Enter the model name to load:").strip()
        print("\nSelect difficulty:")
        print("1. Easy")
        print("2. Medium")
        print("3. Hard")
        diff_choice = input("Enter choice (1-3): ").strip()

        difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        difficulty = difficulty_map.get(diff_choice, "medium")

        watch_trained_robot(model_name, model_type="PPO", episodes=5, difficulty=difficulty, config_path=config_path)

    elif choice == "2":
        app = TrainGUI()
        model_file_path = app.open_file_dialog()
        if model_file_path:
            app.close_window()

        # model_name = input("Enter the model name to load:").strip()


        print("\nSelect difficulty:")
        print("1. Easy")
        print("2. Medium")
        print("3. Hard")
        diff_choice = input("Enter choice (1-3): ").strip()

        difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        difficulty = difficulty_map.get(diff_choice, "medium")

        watch_trained_robot(model_file_path, model_type="DDPG", episodes=5, difficulty=difficulty, config_path=config_path)

    elif choice == "3":
        # compare_models(config_path)
        print("WIP, STILL NEED TODO")

    elif choice == "4":
        print("WIP, STILL NEED TODO")
        # print("\nSelect difficulty for detailed analysis:")
        # print("1. Easy")
        # print("2. Medium")
        # print("3. Hard")
        # diff_choice = input("Enter choice (1-3): ").strip()

        # difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        # difficulty = difficulty_map.get(diff_choice, "medium")

        # results = detailed_model_comparison(episodes=5, difficulty=difficulty, config_path=config_path)
        # detailed_statistical_analysis(results, episodes=5)

    elif choice == "5":
        # test_random_baseline(config_path)
        print("WIP, STILL NEED TODO")

    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()
