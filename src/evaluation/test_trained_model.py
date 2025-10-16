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

# Import 3-way comparison functions from extended_train_script.py
from src.training.extended_train_script import (
    HandCodedPolicy,
    compare_three_policies
)

def create_timestamped_results_dir(base_name="evaluation"):
    """
    Create a timestamped results directory to prevent overwriting.

    Args:
        base_name: Descriptive name for the evaluation type
            Examples: "ddpg_evaluation", "ppo_evaluation", "three_way_comparison"

    Returns:
        Path object to the created directory

    Example:
        >>> results_dir = create_timestamped_results_dir("ddpg_evaluation")
        >>> # Creates: ./test_results/ddpg_evaluation_20251016_143022/
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'./test_results/{base_name}_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

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

        # Extract model name from path for cleaner filename
        model_basename = Path(model_name).stem if model_name else "unknown_model"

        # Create descriptive filename (no timestamp - directory is timestamped)
        filename = f"test_metrics_{model_type}_{difficulty}.json"
        filepath = save_path / filename

        # Get timestamp for metadata
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

def create_test_visualizations(metrics_data, save_dir=None, model_name="model"):
    """
    Create comprehensive visualization plots from test metrics.

    Args:
        metrics_data: Either a dictionary containing metrics or path to JSON file
        save_dir: Directory to save plots (required if metrics_data is dict)
        model_name: Name for plot titles (default: "model")

    Returns:
        Dictionary of plot paths {'plot_name': 'path/to/plot.png'}
    """
    try:
        # Handle both dict and filepath inputs
        if isinstance(metrics_data, (str, Path)):
            # Load metrics from file
            with open(metrics_data, 'r') as f:
                data = json.load(f)
            # Determine save directory from file path if not provided
            if save_dir is None:
                save_dir = Path(metrics_data).parent / "plots"
        else:
            # metrics_data is already a dictionary
            data = metrics_data
            if save_dir is None:
                raise ValueError("save_dir must be provided when metrics_data is a dictionary")

        # Ensure save_dir is a Path object
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Extract data - handle both new format (dict) and old format (JSON with nested structure)
        if 'metadata' in data:
            # Old format from JSON file
            metadata = data.get('metadata', {})
            raw_metrics = data.get('raw_metrics', {})
            stats = data.get('detailed_statistics', {})
            model_type = metadata.get('model_type', model_name)
        else:
            # New format from direct dict
            metadata = {}
            raw_metrics = data.get('metrics', {})
            stats = data.get('statistics', {})
            model_type = model_name

        # Set academic plotting style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9

        plot_paths = {}

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

        # 1d: Goal Approaches Over Time (much better than histogram)
        if 'goal_approaches' in raw_metrics and 'goals_scored' in raw_metrics:
            approaches = raw_metrics['goal_approaches']
            goals = raw_metrics['goals_scored']
            episodes_range = range(1, len(approaches) + 1)

            # Plot goal approaches as a line chart showing progression
            ax4.plot(episodes_range, approaches, linewidth=2, color='#8E44AD',
                    marker='o', markersize=4, alpha=0.8, label='Approaches per Episode')

            # Add moving average to show trend
            if len(approaches) >= 5:
                window = min(5, len(approaches))
                moving_avg = np.convolve(approaches, np.ones(window)/window, mode='valid')
                ax4.plot(range(window, len(approaches) + 1), moving_avg,
                        linewidth=2.5, color='#E74C3C', linestyle='--',
                        label=f'{window}-Episode Avg', alpha=0.9)

            # Highlight episodes where approaches led to goals
            goal_episodes = [i+1 for i, g in enumerate(goals) if g > 0]
            goal_approach_values = [approaches[i] for i in range(len(approaches)) if goals[i] > 0]
            if goal_episodes:
                ax4.scatter(goal_episodes, goal_approach_values, color='#27AE60',
                           s=100, marker='*', edgecolors='black', linewidths=1.5,
                           label='Goals Scored', zorder=5)

            ax4.set_xlabel('Episode Number')
            ax4.set_ylabel('Goal Approaches')
            ax4.set_title('Attacking Progression Over Time\n(Stars = Successful Goals)')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=9)

            # Add summary text
            success_rate = len(goal_episodes) / len(approaches) * 100 if approaches else 0
            avg_approaches = np.mean(approaches)
            ax4.text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%\nAvg Approaches: {avg_approaches:.1f}',
                    transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plot1_path = save_dir / f"game_performance_{model_type}.png"
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

        plot2_path = save_dir / f"learning_metrics_{model_type}.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths['learning_metrics'] = str(plot2_path)
        print(f"  Created: {plot2_path.name}")

        # ===== PLOT 3: Success Rate Analysis (2 subplots) =====
        # More intuitive than correlation - shows success distribution and episode outcomes
        fig3, (ax9, ax10) = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle(f'{model_type} Episode Outcomes Analysis', fontsize=14, fontweight='bold')
        plt.subplots_adjust(wspace=0.3)

        # 3a: Success vs Failure Episodes (easier to understand than correlation)
        if 'goals_scored' in raw_metrics:
            goals = np.array(raw_metrics['goals_scored'])
            successful_episodes = np.sum(goals > 0)
            failed_episodes = len(goals) - successful_episodes

            # Pie chart showing success rate
            sizes = [successful_episodes, failed_episodes]
            labels = [f'Goals Scored\n({successful_episodes} episodes)',
                     f'No Goals\n({failed_episodes} episodes)']
            colors = ['#27AE60', '#E74C3C']
            explode = (0.05, 0)  # Slightly separate the success slice

            ax9.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
            ax9.set_title('Goal-Scoring Success Rate', fontsize=13, fontweight='bold')

            # Add text summary
            success_rate = (successful_episodes / len(goals)) * 100
            ax9.text(0, -1.4, f'Overall: {success_rate:.1f}% of episodes had at least 1 goal',
                    ha='center', fontsize=10, style='italic')

        # 3b: Average Metrics - Bar Chart Comparison
        if raw_metrics:
            metrics_to_plot = []
            metric_values = []
            metric_colors = []

            # Calculate averages for key metrics
            if 'goals_scored' in raw_metrics:
                avg_goals = np.mean(raw_metrics['goals_scored'])
                metrics_to_plot.append(f'Avg Goals\n({avg_goals:.2f})')
                metric_values.append(avg_goals)
                metric_colors.append('#2E86C1')

            if 'ball_possession_timesteps' in raw_metrics and 'episode_length' in raw_metrics:
                possession_steps = np.array(raw_metrics['ball_possession_timesteps'])
                episode_lengths = np.array(raw_metrics['episode_length'])
                avg_possession = np.mean((possession_steps / episode_lengths) * 100)
                metrics_to_plot.append(f'Avg Possession\n({avg_possession:.1f}%)')
                metric_values.append(avg_possession)
                metric_colors.append('#27AE60')

            if 'cumulative_reward' in raw_metrics:
                avg_reward = np.mean(raw_metrics['cumulative_reward'])
                # Normalize reward to 0-100 scale for visual comparison
                reward_normalized = min(100, max(0, avg_reward / 100))  # Assuming ~10000 is good
                metrics_to_plot.append(f'Avg Reward\n({avg_reward:.0f})')
                metric_values.append(avg_reward / 100)  # Scale down for visualization
                metric_colors.append('#8E44AD')

            if metrics_to_plot:
                bars = ax10.bar(range(len(metrics_to_plot)), metric_values,
                               color=metric_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
                ax10.set_xticks(range(len(metrics_to_plot)))
                ax10.set_xticklabels(metrics_to_plot, fontsize=10)
                ax10.set_ylabel('Value', fontsize=12)
                ax10.set_title('Key Performance Metrics', fontsize=13, fontweight='bold')
                ax10.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar, val in zip(bars, metric_values):
                    height = bar.get_height()
                    ax10.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plot3_path = save_dir / f"episode_outcomes_{model_type}.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths['episode_outcomes'] = str(plot3_path)
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

            plot4_path = save_dir / f"episode_lengths_over_time_{model_type}.png"
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

        # Create timestamped results directory (all files go here)
        if save_metrics or create_plots:
            results_dir = create_timestamped_results_dir(f"{model_type.lower()}_evaluation")
            print(f"\n{'='*70}")
            print(f"RESULTS SAVED TO: {results_dir}")
            print(f"{'='*70}")
        else:
            results_dir = None

        # Save metrics to file if requested
        metrics_filepath = None
        if save_metrics and results_dir:
            metrics_filepath = save_test_metrics(
                results=results,
                model_name=model_name,
                model_type=model_type,
                difficulty=difficulty,
                config_path=config_path,
                save_dir=str(results_dir)  # Use timestamped directory
            )
            if metrics_filepath:
                results['metrics_file'] = metrics_filepath
                print(f"  - Metrics JSON: {Path(metrics_filepath).name}")

        # Create visualization plots if requested
        if create_plots and results_dir:
            # Pass the results data and directory directly
            plot_paths = create_test_visualizations(
                metrics_data=results,  # Pass data dict instead of filepath
                save_dir=results_dir,
                model_name=model_type
            )
            if plot_paths:
                results['plot_files'] = plot_paths
                print(f"  - Plots: {len(plot_paths)} visualization files")
                for plot_name, plot_path in plot_paths.items():
                    print(f"      • {Path(plot_path).name}")

        if results_dir:
            print(f"{'='*70}\n")

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

def compare_pretrained_models(ddpg_model_path, ppo_model_path,
                             difficulty="medium", n_episodes=50,
                             config_path="configs/field_config.yaml",
                             render_demo=False, debug_display=False):
    """
    Compare pretrained DDPG vs PPO models.

    This wraps the existing compare_three_policies() function from
    extended_train_script.py to work with pretrained model files.

    Args:
        ddpg_model_path: Path to trained DDPG model (.zip)
        ppo_model_path: Path to trained PPO model (.zip)
        difficulty: Environment difficulty ("easy", "medium", "hard")
        n_episodes: Number of evaluation episodes per model
        config_path: Path to field configuration YAML
        render_demo: If True, render episodes in human mode
        debug_display: If True, show debug overlay during evaluation

    Returns:
        Dictionary with comparison results and file paths
    """
    print(f"\n{'='*70}")
    print("DDPG vs PPO MODEL COMPARISON")
    print(f"{'='*70}\n")

    # Load pretrained models
    print(f"Loading models...")
    ddpg_model = DDPG.load(ddpg_model_path)
    ppo_model = PPO.load(ppo_model_path)
    print(f"  DDPG: {ddpg_model_path}")
    print(f"  PPO: {ppo_model_path}")

    # Create timestamped output directory to prevent overwriting
    output_dir = create_timestamped_results_dir("three_way_comparison")
    print(f"\nResults will be saved to: {output_dir}")

    # Run the comparison (reuses existing function)
    print(f"\nRunning 3-way comparison with {n_episodes} episodes per model...")
    if debug_display:
        print(f"Debug display: ENABLED")
    results = compare_three_policies(
        ppo_model=ppo_model,
        ddpg_model=ddpg_model,
        config_path=config_path,
        difficulty=difficulty,
        n_episodes=n_episodes,
        output_dir=str(output_dir),
        logger=None,  # test_trained_model.py doesn't have logger
        debug_display=debug_display  # Pass debug_display flag
    )

    # Optional: Demonstrate with rendering
    if render_demo:
        print(f"\n{'='*70}")
        print("DEMONSTRATION MODE: Rendering sample episodes")
        print(f"{'='*70}\n")
        # Use min(n_episodes, 5) for demo to avoid too many rendered episodes
        demo_episodes = min(n_episodes, 5)
        print(f"Rendering {demo_episodes} episodes per model (limited for demo mode)")
        demonstrate_models(ddpg_model, ppo_model, config_path, difficulty, episodes_per_model=demo_episodes, debug_display=debug_display)

    # Print summary
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - JSON data: {results.get('json_path', 'N/A')}")
    print(f"  - Text summary: {results.get('summary_report', 'N/A')}")
    print(f"  - Markdown table: {results.get('markdown_path', 'N/A')}")
    print(f"  - Bar charts: {results.get('plots', {}).get('bar_charts', 'N/A')}")
    print(f"  - Histograms: {results.get('plots', {}).get('histograms', 'N/A')}")
    print(f"  - Goals plot: {results.get('plots', {}).get('goals_over_time', 'N/A')}")

    return results

def demonstrate_models(ddpg_model, ppo_model, config_path, difficulty, episodes_per_model=2, debug_display=False):
    """
    Render a few episodes of each model for visual demonstration.

    Args:
        ddpg_model: Loaded DDPG model
        ppo_model: Loaded PPO model
        config_path: Path to field configuration
        difficulty: Environment difficulty
        episodes_per_model: Number of episodes to render per model
        debug_display: Enable debug overlay with live metrics
    """
    # Create environment first
    env = SoccerEnv(
        render_mode="human",
        config_path=config_path,
        difficulty=difficulty,
        reward_type="standard",
        testing_mode=True  # Speed up simulation
    )

    # Speed up visualization (match watch_trained_robot behavior)
    env.dt *= 4.0

    # Enable debug display if requested
    if debug_display:
        if hasattr(env, 'enable_debug_display'):
            env.enable_debug_display()
        if hasattr(env, 'set_show_velocities'):
            env.set_show_velocities(True)

    # Wrap models with ActionSmoothingWrapper (match watch_trained_robot behavior)
    ddpg_smooth = ActionSmoothingWrapper(ddpg_model, smoothing_factor=0.6)
    ppo_smooth = ActionSmoothingWrapper(ppo_model, smoothing_factor=0.6)

    models = [
        ("DDPG", ddpg_smooth),
        ("PPO", ppo_smooth)
    ]

    # Get possession threshold for metrics
    POSSESSION_THRESHOLD = env.field_config.meters_to_pixels(0.3)
    COLLISION_THRESHOLD = env.collision_distance

    for model_name, model in models:
        print(f"\n{'='*50}")
        print(f"Demonstrating: {model_name}")
        print(f"{'='*50}\n")

        # Reset environment for each model (reuse same env instance)
        if model_name != "DDPG":  # Skip reset for first model
            env.close()
            env = SoccerEnv(
                render_mode="human",
                config_path=config_path,
                difficulty=difficulty,
                reward_type="standard",
                testing_mode=True  # Speed up simulation
            )
            # Speed up visualization
            env.dt *= 4.0

            # Re-enable debug display
            if debug_display:
                if hasattr(env, 'enable_debug_display'):
                    env.enable_debug_display()
                if hasattr(env, 'set_show_velocities'):
                    env.set_show_velocities(True)

            # Re-get thresholds
            POSSESSION_THRESHOLD = env.field_config.meters_to_pixels(0.3)
            COLLISION_THRESHOLD = env.collision_distance

        for episode in range(episodes_per_model):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            episode_goals = 0
            episode_possession_steps = 0

            print(f"Episode {episode + 1}/{episodes_per_model}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                # Track metrics for debug display
                if debug_display:
                    robot_ball_distance = np.linalg.norm(env.robot_pos - env.ball_pos)
                    robot_opponent_distance = np.linalg.norm(env.robot_pos - env.opponent_pos)

                    if robot_ball_distance < POSSESSION_THRESHOLD:
                        episode_possession_steps += 1

                    if env._check_goal():
                        episode_goals += 1

                    # Update debug info with current state
                    if hasattr(env, 'set_debug_info'):
                        env.set_debug_info({
                            'policy': model_name,
                            'step': steps,
                            'reward': reward,
                            'cumulative_reward': episode_reward,
                            'goals': episode_goals,
                            'possession_pct': (episode_possession_steps / steps) * 100 if steps > 0 else 0,
                            'robot_pos': env.robot_pos,
                            'ball_pos': env.ball_pos,
                            'action': action,
                            'has_possession': robot_ball_distance < POSSESSION_THRESHOLD,
                            'has_collision': robot_opponent_distance < COLLISION_THRESHOLD,
                            'ball_out': env._check_ball_out_of_play()
                        })

                env.render()
                # Match watch_trained_robot speed
                time.sleep(0.05)  # Normal viewing speed

                done = terminated or truncated

            print(f"  Reward: {episode_reward:.1f}, Steps: {steps}")
            time.sleep(2.0)  # Pause between episodes

        env.close()
        time.sleep(1.0)  # Pause between models

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
    parser.add_argument('--no-plots', action='store_true', help='Disable generation of visualisation plots')

    # 3-way comparison arguments
    parser.add_argument('--compare', action='store_true', help='Run 3-way comparison (DDPG vs PPO vs Hand-Coded)')
    parser.add_argument('--ddpg-model', type=str, help='Path to DDPG model for comparison')
    parser.add_argument('--ppo-model', type=str, help='Path to PPO model for comparison')
    parser.add_argument('--demo', action='store_true', help='Render demonstration episodes after comparison')

    args = parser.parse_args()

    # 3-way comparison mode
    if args.compare and args.ddpg_model and args.ppo_model:
        config_path = load_field_config(args.config)
        compare_pretrained_models(
            ddpg_model_path=args.ddpg_model,
            ppo_model_path=args.ppo_model,
            difficulty=args.difficulty,
            n_episodes=args.episodes,
            config_path=config_path,
            render_demo=args.demo,
            debug_display=args.debug_display  # Pass debug_display flag
        )
        return

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
    print("3. 3-Way Comparison (DDPG vs PPO vs Hand-Coded)")
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
        # Get model paths
        print("\nEnter paths to models:")
        ddpg_path = input("DDPG model path: ").strip()
        ppo_path = input("PPO model path: ").strip()

        # Get difficulty
        print("\nSelect difficulty:")
        print("1. Easy")
        print("2. Medium")
        print("3. Hard")
        diff_choice = input("Enter choice (1-3): ").strip()
        difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        difficulty = difficulty_map.get(diff_choice, "medium")

        # Ask about demonstration
        demo_choice = input("\nRender demonstration episodes? (y/n): ").strip().lower()
        render_demo = demo_choice == 'y'

        # Ask about debug display
        debug_choice = input("Enable debug overlay? (y/n): ").strip().lower()
        debug_display = debug_choice == 'y'

        # Run comparison
        compare_pretrained_models(
            ddpg_model_path=ddpg_path,
            ppo_model_path=ppo_path,
            difficulty=difficulty,
            n_episodes=50,
            config_path=config_path,
            render_demo=render_demo,
            debug_display=debug_display
        )

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
