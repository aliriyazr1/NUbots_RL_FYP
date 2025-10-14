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
import argparse
import traceback
import sys

def load_field_config(config_path="configs/field_config.yaml"):
    """Load and display field configuration"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        field_type = config['field_type']
        field_dims = config['real_world_dimensions'][field_type]

        print(f"\nField Configuration Loaded:")
        print(f"  Type: {field_type.title()}")
        print(f"  Dimensions: {field_dims['field_length']}m x {field_dims['field_width']}m")
        print(f"  Goal: {field_dims['goal_width']}m wide")

        return config_path

    except FileNotFoundError:
        print(f"Warning: {config_path} not found! Using default configuration.")
        return None
    except Exception as e:
        print(f"Error loading config: {e}. Using default configuration.")
        return None
    
def watch_trained_robot(model_name, model_type=None, episodes=5, difficulty="medium", config_path="SoccerEnv/field_config.yaml", testing_mode=False, show_visualization=True, debug_display=False, slow_motion=False):
    """
    Evaluate trained robot with comprehensive performance metrics

    Args:
        model_name: Path to trained model
        model_type: Algorithm type (PPO or DDPG)
        episodes: Number of evaluation episodes
        difficulty: Environment difficulty level
        config_path: Path to field configuration
        testing_mode: Enable faster simulation
        show_visualization: Enable pygame rendering
        debug_display: Show debug overlay during rendering
        slow_motion: Slow down visualization for detailed observation
    """
    print(f"\n{'='*70}")
    print(f"MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Algorithm: {model_type}")
    print(f"Episodes: {episodes}")
    print(f"Difficulty: {difficulty.title()}")
    if testing_mode:
        print(f"Testing mode: ENABLED (faster simulation)")

    model_types = {"PPO": PPO, "DDPG": DDPG}
    try:
        if not model_name:
            print("Error: Invalid and empty model name!")
            return
        if model_type not in model_types:
            print("Error: Unknown model type!")
            return

        # Load the trained model
        ModelName = model_types[model_type]
        model = ModelName.load(model_name)

        smooth_model = ActionSmoothingWrapper(model, smoothing_factor=0.6)

        # Create environment with rendering based on show_visualization flag
        render_mode = "human" if show_visualization else None
        env = SoccerEnv(render_mode=render_mode, difficulty=difficulty, config_path=config_path, testing_mode=testing_mode)
        env.dt *= 4.0  # Speed up visualization
        if show_visualization:
            env.set_show_velocities(True)
            if debug_display:
                env.enable_debug_display()  # Will implement this in soccerenv.py

        # Display field information
        field_config = env.field_config
        print(f"Field Type: {field_config.config['field_type'].title()}")
        print(f"Real Dimensions: {field_config.real_dims['field_length']}m x {field_config.real_dims['field_width']}m")
        print(f"Display Size: {field_config.field_width_pixels}x{field_config.field_height_pixels} pixels")
        print(f"Goal Width: {field_config.real_dims['goal_width']}m")

        print(f"\n{'='*70}")
        print("STARTING EVALUATION")
        print(f"{'='*70}\n")

        # Comprehensive metrics tracking
        all_metrics = {
            'goals_scored': [],
            'ball_possession_timesteps': [],
            'ball_out_of_bounds_count': [],
            'robot_collisions_count': [],
            'episode_length': [],
            'cumulative_reward': [],
            'final_ball_distance': [],
            'goal_approaches': []
        }

        # Constants for metrics calculation
        POSSESSION_DISTANCE_THRESHOLD = env.field_config.meters_to_pixels(0.3)  # 0.3m in pixels
        COLLISION_DISTANCE_THRESHOLD = (env.robot_radius + env.robot_radius) * 1.1  # Slightly larger than sum of radii
        ATTACKING_THIRD_START = env.field_width * 0.66  # Right third of field

        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")

            obs, _ = env.reset()

            # Episode-specific metrics
            episode_goals = 0
            episode_possession_steps = 0
            episode_out_of_bounds = 0
            episode_collisions = 0
            episode_steps = 0
            episode_reward = 0
            episode_goal_approaches = 0

            start_time = time.time()
            in_attacking_third = False

            for step in range(env.max_steps):
                # Get action from trained model
                action, _ = smooth_model.predict(obs, deterministic=True)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # 1. Ball possession tracking (distance < 0.3m)
                robot_ball_distance = np.linalg.norm(env.robot_pos - env.ball_pos)
                if robot_ball_distance < POSSESSION_DISTANCE_THRESHOLD:
                    episode_possession_steps += 1

                # 2. Robot collision tracking (distance to opponent < sum of radii)
                robot_opponent_distance = np.linalg.norm(env.robot_pos - env.opponent_pos)
                if robot_opponent_distance < COLLISION_DISTANCE_THRESHOLD:
                    episode_collisions += 1

                # 3. Goal approaches tracking (ball enters attacking third)
                ball_in_attacking_third = env.ball_pos[0] > ATTACKING_THIRD_START
                if ball_in_attacking_third and not in_attacking_third:
                    episode_goal_approaches += 1
                in_attacking_third = ball_in_attacking_third

                # 4. Ball out of bounds tracking
                if env._check_ball_out_of_play():
                    episode_out_of_bounds += 1

                # 5. Goals scored tracking
                if env._check_goal():
                    episode_goals += 1

                # Rendering with optional debug overlay
                if show_visualization:
                    if debug_display:
                        # Pass current episode data to render for overlay display
                        env.set_debug_info({
                            'step': step,
                            'reward': reward,
                            'cumulative_reward': episode_reward,
                            'goals': episode_goals,
                            'possession_pct': (episode_possession_steps / (step + 1)) * 100,
                            'robot_pos': env.robot_pos,
                            'ball_pos': env.ball_pos,
                            'action': action,
                            'has_possession': robot_ball_distance < POSSESSION_DISTANCE_THRESHOLD,
                            'has_collision': robot_opponent_distance < COLLISION_DISTANCE_THRESHOLD,
                            'ball_out': env._check_ball_out_of_play()
                        })
                    env.render()

                    # Adaptive sleep based on flags
                    if slow_motion:
                        time.sleep(0.1)  # Very slow for detailed observation
                    elif testing_mode:
                        time.sleep(0.02)  # Faster for testing
                    else:
                        time.sleep(0.05)  # Normal speed

                if terminated:
                    # Check if it's a successful termination (ball in goal)
                    if env._check_goal():
                        print(f"  SUCCESS! Robot scored a goal!")
                    else:
                        print(f"  Episode ended (collision or failure)")
                    break

                if truncated:
                    print(f"  Episode timeout (reached {env.max_steps} steps)")
                    break

            # Calculate final ball distance (in meters)
            final_ball_distance_px = np.linalg.norm(env.robot_pos - env.ball_pos)
            final_ball_distance_m = env.field_config.pixels_to_meters(final_ball_distance_px)

            # Store episode metrics
            all_metrics['goals_scored'].append(episode_goals)
            all_metrics['ball_possession_timesteps'].append(episode_possession_steps)
            all_metrics['ball_out_of_bounds_count'].append(episode_out_of_bounds)
            all_metrics['robot_collisions_count'].append(episode_collisions)
            all_metrics['episode_length'].append(episode_steps)
            all_metrics['cumulative_reward'].append(episode_reward)
            all_metrics['final_ball_distance'].append(final_ball_distance_m)
            all_metrics['goal_approaches'].append(episode_goal_approaches)

            # Episode summary (brief, professional)
            print(f"  Reward: {episode_reward:.2f} | Steps: {episode_steps} | Goals: {episode_goals} | Possession: {episode_possession_steps} steps\n")

            # Wait between episodes if visualizing
            if show_visualization:
                time.sleep(1.0)
        
        env.close()

        # Calculate summary statistics for all metrics
        def calc_stats(data):
            """Calculate mean, std, median, min, max for a list of values"""
            arr = np.array(data)
            return {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'median': np.median(arr),
                'min': np.min(arr),
                'max': np.max(arr)
            }

        metrics_stats = {key: calc_stats(values) for key, values in all_metrics.items()}

        # Calculate traditional summary stats
        total_goals = sum(all_metrics['goals_scored'])
        avg_reward = metrics_stats['cumulative_reward']['mean']
        std_reward = metrics_stats['cumulative_reward']['std']
        avg_length = metrics_stats['episode_length']['mean']
        success_rate = (total_goals / episodes) * 100 if episodes > 0 else 0
        total_steps = sum(all_metrics['episode_length'])

        # Display comprehensive results
        print(f"\n{'='*70}")
        print(f"TEST RESULTS")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Field Configuration: {field_config.config['field_type'].title()}")
        print(f"Difficulty: {difficulty.title()}")
        print(f"Episodes: {episodes}")
        print(f"Total Steps: {total_steps}")
        print(f"\nPrimary Metrics:")
        print(f"  Goals Scored: {total_goals}/{episodes}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Episode Length: {avg_length:.1f} steps")

        # Detailed metrics table
        print(f"\nDetailed Performance Metrics:")
        print(f"{'  Metric':<37} {'Mean ± Std':>18} {'[Min, Max]':>15}")
        print(f"  {'-'*68}")

        # Ball possession
        s = metrics_stats['ball_possession_timesteps']
        poss_rate_mean = (s['mean'] / avg_length) * 100 if avg_length > 0 else 0
        print(f"  {'Ball Possession (steps)':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
        print(f"  {'Ball Possession Rate (%)':<35} {poss_rate_mean:>8.2f}%")

        # Goal approaches
        s = metrics_stats['goal_approaches']
        print(f"  {'Goal Approaches':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")

        # Final ball distance
        s = metrics_stats['final_ball_distance']
        print(f"  {'Final Ball Distance (m)':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.2f}, {s['max']:.2f}]")

        # Collisions
        s = metrics_stats['robot_collisions_count']
        collision_rate_mean = (s['mean'] / avg_length) * 100 if avg_length > 0 else 0
        print(f"  {'Robot Collisions (steps)':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
        print(f"  {'Collision Rate (%)':<35} {collision_rate_mean:>8.2f}%")

        # Out of bounds
        s = metrics_stats['ball_out_of_bounds_count']
        print(f"  {'Ball Out of Bounds':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")

        # Performance assessment
        print(f"\nPERFORMANCE ASSESSMENT:")
        if success_rate >= 70:
            print(f"  Status: EXCELLENT - Success rate {success_rate:.1f}% is very high")
        elif success_rate >= 40:
            print(f"  Status: GOOD - Success rate {success_rate:.1f}% is solid")
        elif success_rate >= 20:
            print(f"  Status: DECENT - Success rate {success_rate:.1f}% shows learning")
        else:
            print(f"  Status: NEEDS IMPROVEMENT - Success rate {success_rate:.1f}% is low")

        if avg_reward > 1000:
            print(f"  Reward Level: EXCELLENT - Average reward {avg_reward:.1f} is very good")
        elif avg_reward > 0:
            print(f"  Reward Level: POSITIVE - Average reward {avg_reward:.1f}")
        else:
            print(f"  Reward Level: NEGATIVE - Average reward {avg_reward:.1f}, needs work")

        print(f"{'='*70}\n")

        return {
            'metrics': all_metrics,
            'statistics': metrics_stats,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'avg_length': avg_length,
            'goals_scored': total_goals,
            'episodes': episodes,
            'total_steps': total_steps
        }
    
    except FileNotFoundError:
        print(f"Error: Model file '{model_name}' not found!")
        print("Make sure you've trained the model first or check the path.")
        return None
    except KeyboardInterrupt:
        print("\nTesting stopped by user")
        if 'env' in locals():
            env.close()
        return None
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc(file=sys.stdout)

        if 'env' in locals():
            env.close()
        return None

def main():
    """Main function for model evaluation with CLI argument support"""
    parser = argparse.ArgumentParser(description='Evaluate trained soccer RL models')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--model-type', type=str, choices=['PPO', 'DDPG'], help='Algorithm type')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'], default='medium', help='Environment difficulty')
    parser.add_argument('--config', type=str, default='configs/field_config.yaml', help='Path to field configuration')
    parser.add_argument('--no-visualization', action='store_true', help='Disable pygame rendering for faster evaluation')
    parser.add_argument('--debug-display', action='store_true', help='Show debug overlay during rendering')
    parser.add_argument('--slow-motion', action='store_true', help='Slow down visualization for detailed observation')
    parser.add_argument('--testing-mode', action='store_true', help='Enable faster simulation (3x speed)')

    args = parser.parse_args()

    # If CLI args provided, use them directly
    if args.model and args.model_type:
        config_path = load_field_config(args.config)
        watch_trained_robot(
            model_name=args.model,
            model_type=args.model_type,
            episodes=args.episodes,
            difficulty=args.difficulty,
            config_path=config_path,
            testing_mode=args.testing_mode,
            show_visualization=not args.no_visualization,
            debug_display=args.debug_display,
            slow_motion=args.slow_motion
        )
        return

    # Otherwise, use interactive menu
    config_path = load_field_config()

    print("\nModel Testing Options:")
    print("1. Test PPO model")
    print("2. Test DDPG model")
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
