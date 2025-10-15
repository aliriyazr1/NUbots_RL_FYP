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
import json
import datetime
from pathlib import Path

def save_test_metrics(results, model_name, model_type, difficulty, config_path, save_dir="test_results"):
    """
    Save test metrics to JSON file with metadata

    Args:
        results: Dictionary containing metrics and statistics
        model_name: Path to the model file
        model_type: Algorithm type (PPO or DDPG)
        difficulty: Environment difficulty level
        config_path: Path to field configuration
        save_dir: Directory to save metrics (default: test_results/)

    Returns:
        Path to saved metrics file
    """
    try:
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model name from path for cleaner filename
        model_basename = Path(model_name).stem if model_name else "unknown_model"

        # Create descriptive filename
        filename = f"test_metrics_{model_type}_{model_basename}_{difficulty}_{timestamp}.json"
        filepath = save_path / filename

        # Prepare data structure with metadata
        save_data = {
            'metadata': {
                'timestamp': timestamp,
                'datetime': datetime.datetime.now().isoformat(),
                'model_path': model_name,
                'model_type': model_type,
                'difficulty': difficulty,
                'config_path': config_path,
                'episodes': results.get('episodes', 0),
                'total_steps': results.get('total_steps', 0)
            },
            'summary_statistics': {
                'avg_reward': results.get('avg_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'success_rate': results.get('success_rate', 0),
                'avg_episode_length': results.get('avg_length', 0),
                'total_goals': results.get('goals_scored', 0)
            },
            'raw_metrics': {},
            'detailed_statistics': {}
        }

        # Add raw metrics (per-episode data)
        if 'metrics' in results:
            for key, values in results['metrics'].items():
                # Convert numpy arrays to lists for JSON serialization
                save_data['raw_metrics'][key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in values]

        # Add detailed statistics
        if 'statistics' in results:
            for key, stats in results['statistics'].items():
                save_data['detailed_statistics'][key] = {
                    stat_name: float(stat_value) if isinstance(stat_value, (np.integer, np.floating)) else stat_value
                    for stat_name, stat_value in stats.items()
                }

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\nMetrics saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        print(f"Error saving metrics: {e}")
        traceback.print_exc()
        return None

def create_test_visualizations(metrics_filepath, save_dir=None):
    """
    Create comprehensive visualization plots from saved test metrics

    Args:
        metrics_filepath: Path to saved metrics JSON file
        save_dir: Directory to save plots (default: same directory as metrics file)

    Returns:
        Dictionary of plot paths {'plot_name': 'path/to/plot.png'}
    """
    try:
        # Load metrics from file
        with open(metrics_filepath, 'r') as f:
            data = json.load(f)

        # Determine save directory
        if save_dir is None:
            save_dir = Path(metrics_filepath).parent / "plots"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        metadata = data.get('metadata', {})
        raw_metrics = data.get('raw_metrics', {})
        stats = data.get('detailed_statistics', {})

        # Set academic plotting style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9

        plot_paths = {}
        timestamp = metadata.get('timestamp', 'unknown')
        model_type = metadata.get('model_type', 'unknown')

        print(f"\nGenerating visualization plots...")

        # ===== PLOT 1: Game Performance Metrics (4 subplots) =====
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle(f'{model_type} Game Performance Analysis', fontsize=14, fontweight='bold')
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        # 1a: Goals Distribution
        if 'goals_scored' in raw_metrics:
            goals = raw_metrics['goals_scored']
            episodes = list(range(1, len(goals) + 1))
            ax1.bar(episodes, goals, color='#2E86C1', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Episode Number')
            ax1.set_ylabel('Goals Scored')
            ax1.set_title('Goals Per Episode')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(np.mean(goals), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(goals):.2f}')
            ax1.legend()

        # 1b: Ball Possession
        if 'ball_possession_timesteps' in raw_metrics and 'episode_length' in raw_metrics:
            possession_steps = np.array(raw_metrics['ball_possession_timesteps'])
            episode_lengths = np.array(raw_metrics['episode_length'])
            possession_pct = (possession_steps / episode_lengths) * 100
            ax2.plot(range(1, len(possession_pct) + 1), possession_pct, linewidth=2,
                    color='#27AE60', marker='o', markersize=4)
            ax2.set_xlabel('Episode Number')
            ax2.set_ylabel('Ball Possession (%)')
            ax2.set_title('Ball Possession Efficiency')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(np.mean(possession_pct), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(possession_pct):.1f}%')
            ax2.legend()

        # 1c: Collisions and Out of Bounds (binary per episode: 0 or 1)
        if 'robot_collisions_count' in raw_metrics and 'ball_out_of_bounds_count' in raw_metrics:
            episodes_range = range(1, len(raw_metrics['robot_collisions_count']) + 1)
            ax3.plot(episodes_range, raw_metrics['robot_collisions_count'],
                    linewidth=2, color='#E74C3C', marker='s', markersize=4, label='Collision Occurred')
            ax3.plot(episodes_range, raw_metrics['ball_out_of_bounds_count'],
                    linewidth=2, color='#F39C12', marker='^', markersize=4, label='Out of Bounds Occurred')
            ax3.set_xlabel('Episode Number')
            ax3.set_ylabel('Occurrence (0 or 1)')
            ax3.set_title('Collision and Out-of-Bounds Events per Episode')
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['No', 'Yes'])
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        # 1d: Goal Approaches
        if 'goal_approaches' in raw_metrics:
            approaches = raw_metrics['goal_approaches']
            ax4.hist(approaches, bins=max(5, max(approaches) + 1 if approaches else 5),
                    color='#8E44AD', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Goal Approaches per Episode')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Goal Approach Distribution')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axvline(np.mean(approaches), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(approaches):.1f}')
            ax4.legend()

        plot1_path = save_dir / f"game_performance_{model_type}_{timestamp}.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths['game_performance'] = str(plot1_path)
        print(f"  Created: {plot1_path.name}")

        # ===== PLOT 2: Learning Metrics (4 subplots) =====
        fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle(f'{model_type} Learning Analysis', fontsize=14, fontweight='bold')
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        # 2a: Rewards Distribution
        if 'cumulative_reward' in raw_metrics:
            rewards = raw_metrics['cumulative_reward']
            ax5.hist(rewards, bins=30, color='#16A085', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Cumulative Reward')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Reward Distribution')
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(rewards):.1f}')
            ax5.legend()

        # 2b: Rewards Box Plot
        if 'cumulative_reward' in raw_metrics:
            ax6.boxplot(rewards, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#16A085', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax6.set_ylabel('Cumulative Reward')
            ax6.set_title('Reward Distribution (Box Plot)')
            ax6.grid(True, alpha=0.3, axis='y')

        # 2c: Rewards Over Episodes
        if 'cumulative_reward' in raw_metrics:
            episodes_range = range(1, len(rewards) + 1)
            ax7.plot(episodes_range, rewards, linewidth=1.5, color='#16A085',
                    marker='o', markersize=3, alpha=0.7, label='Episode Reward')
            # Moving average
            if len(rewards) > 5:
                window = min(5, len(rewards) // 3)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax7.plot(range(window, len(rewards) + 1), moving_avg, linewidth=2.5,
                        color='#E74C3C', label=f'{window}-Episode Moving Avg')
            ax7.set_xlabel('Episode Number')
            ax7.set_ylabel('Cumulative Reward')
            ax7.set_title('Reward Progression')
            ax7.grid(True, alpha=0.3)
            ax7.legend()

        # 2d: Episode Length Distribution
        if 'episode_length' in raw_metrics:
            lengths = raw_metrics['episode_length']
            ax8.hist(lengths, bins=20, color='#D35400', alpha=0.7, edgecolor='black')
            ax8.set_xlabel('Episode Length (steps)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Episode Length Distribution')
            ax8.grid(True, alpha=0.3, axis='y')
            ax8.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(lengths):.1f}')
            ax8.legend()

        plot2_path = save_dir / f"learning_metrics_{model_type}_{timestamp}.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths['learning_metrics'] = str(plot2_path)
        print(f"  Created: {plot2_path.name}")

        # ===== PLOT 3: Correlation Analysis (2 subplots) =====
        fig3, (ax9, ax10) = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle(f'{model_type} Correlation Analysis', fontsize=14, fontweight='bold')
        plt.subplots_adjust(wspace=0.3)

        # 3a: Reward vs Goals
        if 'cumulative_reward' in raw_metrics and 'goals_scored' in raw_metrics:
            rewards = raw_metrics['cumulative_reward']
            goals = raw_metrics['goals_scored']
            ax9.scatter(goals, rewards, alpha=0.6, s=100, c='#2E86C1', edgecolors='black')
            ax9.set_xlabel('Goals Scored')
            ax9.set_ylabel('Cumulative Reward')
            ax9.set_title('Reward vs Goals Correlation')
            ax9.grid(True, alpha=0.3)
            # Add correlation coefficient
            if len(goals) > 1:
                corr = np.corrcoef(goals, rewards)[0, 1]
                ax9.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=ax9.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3b: Possession vs Success
        if 'ball_possession_timesteps' in raw_metrics and 'goals_scored' in raw_metrics:
            possession_steps = np.array(raw_metrics['ball_possession_timesteps'])
            episode_lengths = np.array(raw_metrics['episode_length'])
            possession_pct = (possession_steps / episode_lengths) * 100
            goals = raw_metrics['goals_scored']
            ax10.scatter(possession_pct, goals, alpha=0.6, s=100, c='#27AE60', edgecolors='black')
            ax10.set_xlabel('Ball Possession (%)')
            ax10.set_ylabel('Goals Scored')
            ax10.set_title('Possession vs Goals Correlation')
            ax10.grid(True, alpha=0.3)
            # Add correlation coefficient
            if len(possession_pct) > 1:
                corr = np.corrcoef(possession_pct, goals)[0, 1]
                ax10.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                         transform=ax10.transAxes, fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plot3_path = save_dir / f"correlation_analysis_{model_type}_{timestamp}.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths['correlation_analysis'] = str(plot3_path)
        print(f"  Created: {plot3_path.name}")

        # ===== PLOT 4: Episode Length Over Time (matching rewards plot style) =====
        if 'episode_length' in raw_metrics:
            fig4, ax = plt.subplots(1, 1, figsize=(12, 6))
            fig4.suptitle(f'{model_type} Episode Length Progression', fontsize=14, fontweight='bold')

            lengths = raw_metrics['episode_length']
            episodes_range = range(1, len(lengths) + 1)

            # Plot episode lengths
            ax.plot(episodes_range, lengths, linewidth=1.5, color='#D35400',
                   marker='o', markersize=3, alpha=0.7, label='Episode Length')

            # Add moving average (matching rewards plot logic)
            if len(lengths) > 5:
                window = min(5, len(lengths) // 3)
                moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(lengths) + 1), moving_avg, linewidth=2.5,
                       color='#E74C3C', label=f'{window}-Episode Moving Avg')

            # Add mean line
            mean_length = np.mean(lengths)
            ax.axhline(mean_length, color='green', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_length:.1f} steps', alpha=0.7)

            ax.set_xlabel('Episode Number', fontsize=11)
            ax.set_ylabel('Episode Length (steps)', fontsize=11)
            ax.set_title('Episode Length vs Episode Number', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            # Add statistics text box
            stats_text = (
                f'Statistics:\n'
                f'Mean: {np.mean(lengths):.1f}\n'
                f'Std: {np.std(lengths):.1f}\n'
                f'Min: {np.min(lengths):.0f}\n'
                f'Max: {np.max(lengths):.0f}'
            )
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plot4_path = save_dir / f"episode_lengths_over_time_{model_type}_{timestamp}.png"
            plt.savefig(plot4_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_paths['episode_lengths'] = str(plot4_path)
            print(f"  Created: {plot4_path.name}")

        print(f"\nAll plots saved to: {save_dir}")
        return plot_paths

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
        return None

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
    
def watch_trained_robot(model_name, model_type=None, episodes=5, difficulty="medium", config_path="SoccerEnv/field_config.yaml", testing_mode=False, show_visualization=True, debug_display=False, slow_motion=False, save_metrics=True, create_plots=True):
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
        save_metrics: Save metrics to JSON file (default: True)
        create_plots: Generate visualization plots (default: True)
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

        # Use collision distance from environment (loaded from field_config.yaml, varies by difficulty)
        COLLISION_DISTANCE_THRESHOLD = env.collision_distance
        print(f"\nDifficulty: {difficulty}")
        print(f"Collision threshold: {env.field_config.pixels_to_meters(COLLISION_DISTANCE_THRESHOLD):.3f}m ({COLLISION_DISTANCE_THRESHOLD:.1f} pixels)")

        ATTACKING_THIRD_START = env.field_width * 0.66  # Right third of field

        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")

            obs, _ = env.reset()

            # Episode-specific metrics
            episode_goals = 0
            episode_possession_steps = 0
            episode_out_of_bounds = 0  # Binary: 0 or 1 (did out-of-bounds occur this episode?)
            episode_collisions = 0  # Binary: 0 or 1 (did collision occur this episode?)
            episode_steps = 0
            episode_reward = 0
            episode_goal_approaches = 0

            # State tracking flags to prevent double-counting
            # Since episodes terminate on first collision/out-of-bounds,
            # these should never be set more than once per episode
            collision_occurred = False
            out_of_bounds_occurred = False

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

                # 2. Robot collision tracking (detect first occurrence only)
                robot_opponent_distance = np.linalg.norm(env.robot_pos - env.opponent_pos)
                if not collision_occurred and robot_opponent_distance < COLLISION_DISTANCE_THRESHOLD:
                    episode_collisions = 1
                    collision_occurred = True

                # 3. Goal approaches tracking (ball enters attacking third)
                ball_in_attacking_third = env.ball_pos[0] > ATTACKING_THIRD_START
                if ball_in_attacking_third and not in_attacking_third:
                    episode_goal_approaches += 1
                in_attacking_third = ball_in_attacking_third

                # 4. Ball out of bounds tracking (detect first occurrence only)
                if not out_of_bounds_occurred and env._check_ball_out_of_play():
                    episode_out_of_bounds = 1
                    out_of_bounds_occurred = True

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
                    # Check termination reason
                    if env._check_goal():
                        print(f"  SUCCESS! Robot scored a goal!")
                    elif collision_occurred:
                        print(f"  Episode ended: COLLISION with opponent")
                    elif out_of_bounds_occurred:
                        print(f"  Episode ended: Ball OUT OF BOUNDS")
                    elif hasattr(env, '_check_opponent_goal') and env._check_opponent_goal():
                        print(f"  Episode ended: Opponent scored")
                    else:
                        print(f"  Episode ended: Other termination reason")
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

        # Collisions (binary: 0 or 1 per episode)
        s = metrics_stats['robot_collisions_count']
        collision_rate_pct = (s['mean']) * 100  # Percentage of episodes with collisions
        print(f"  {'Robot Collision Occurrences':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
        print(f"  {'Episodes with Collisions (%)':<35} {collision_rate_pct:>8.2f}%")

        # Out of bounds (binary: 0 or 1 per episode)
        s = metrics_stats['ball_out_of_bounds_count']
        out_of_bounds_pct = (s['mean']) * 100  # Percentage of episodes with out-of-bounds
        print(f"  {'Ball Out of Bounds Occurrences':<35} {s['mean']:>8.2f} ± {s['std']:<7.2f} [{s['min']:.0f}, {s['max']:.0f}]")
        print(f"  {'Episodes with Out-of-Bounds (%)':<35} {out_of_bounds_pct:>8.2f}%")

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

        results = {
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

        # Save metrics to file if requested
        metrics_filepath = None
        if save_metrics:
            metrics_filepath = save_test_metrics(
                results=results,
                model_name=model_name,
                model_type=model_type,
                difficulty=difficulty,
                config_path=config_path
            )
            if metrics_filepath:
                results['metrics_file'] = metrics_filepath

        # Create visualization plots if requested
        if create_plots and metrics_filepath:
            plot_paths = create_test_visualizations(metrics_filepath)
            if plot_paths:
                results['plot_files'] = plot_paths

        return results
    
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
    parser.add_argument('--no-save-metrics', action='store_true', help='Disable saving metrics to file')
    parser.add_argument('--no-plots', action='store_true', help='Disable generation of visualization plots')

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
            slow_motion=args.slow_motion,
            save_metrics=not args.no_save_metrics,
            create_plots=not args.no_plots
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
