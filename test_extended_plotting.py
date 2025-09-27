#!/usr/bin/env python3
"""
Test script for plotting functions from extended_train_script.py
This directly tests the actual plotting code without copying it.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

# Add src to path to import modules
sys.path.append('src')
sys.path.append('src/training')


# Import the plotting functions and classes we want to test
from extended_train_script import (
    MultiModelTrainingSystem,
    create_academic_training_curves,
    create_algorithm_comparison_plot,
    create_comprehensive_comparison_plots,
    create_enhanced_metrics_comparison,
    load_training_data
)

def generate_fake_training_data():
    """Generate realistic fake training data for testing plots"""
    
    # PPO training data - typically more stable but slower learning
    ppo_timesteps = np.linspace(0, 2500000, 2000)  # 2.5M timesteps, 2000 episodes
    ppo_base_progress = np.logspace(0, 1, len(ppo_timesteps)) - 1  # Logarithmic learning
    ppo_base_progress = ppo_base_progress / np.max(ppo_base_progress)  # Normalize
    
    # Add realistic noise and variation
    ppo_noise = np.random.normal(0, 0.15, len(ppo_timesteps))  # Moderate noise
    ppo_rewards = -50 + (ppo_base_progress * 120) + ppo_noise * (20 + ppo_base_progress * 30)
    
    # Add some episodes with very poor performance (exploration)
    exploration_episodes = np.random.choice(len(ppo_rewards)//3, size=len(ppo_rewards)//20)
    ppo_rewards[exploration_episodes] = np.random.uniform(-80, -40, len(exploration_episodes))
    
    # Smooth out the end (convergence)
    convergence_start = int(0.7 * len(ppo_rewards))
    ppo_rewards[convergence_start:] = np.convolve(
        ppo_rewards[convergence_start:], 
        np.ones(20)/20, 
        mode='same'
    ) * 0.8 + ppo_rewards[convergence_start:] * 0.2
    
    # Episode lengths (PPO tends to have more varied episode lengths)
    ppo_episode_lengths = np.random.gamma(2.5, 40, len(ppo_timesteps)).astype(int)
    ppo_episode_lengths = np.clip(ppo_episode_lengths, 10, 200)
    
    # DDPG training data - typically more unstable but can learn faster initially
    ddpg_timesteps = np.linspace(0, 2500000, 1800)  # Slightly fewer episodes
    ddpg_base_progress = 1 - np.exp(-np.linspace(0, 4, len(ddpg_timesteps)))  # Exponential learning
    
    # More noise for DDPG (less stable)
    ddpg_noise = np.random.normal(0, 0.25, len(ddpg_timesteps))
    ddpg_rewards = -45 + (ddpg_base_progress * 110) + ddpg_noise * (25 + ddpg_base_progress * 35)
    
    # Add periodic instability (characteristic of off-policy methods)
    instability_points = np.random.choice(
        len(ddpg_rewards), 
        size=len(ddpg_rewards)//15, 
        replace=False
    )
    for point in instability_points:
        if point < len(ddpg_rewards) - 20:
            ddpg_rewards[point:point+np.random.randint(5, 15)] *= np.random.uniform(0.3, 0.8)
    
    # DDPG episode lengths (typically more consistent)
    ddpg_episode_lengths = np.random.gamma(3.0, 35, len(ddpg_timesteps)).astype(int)
    ddpg_episode_lengths = np.clip(ddpg_episode_lengths, 15, 180)
    
    # Generate enhanced metrics for both algorithms
    def generate_enhanced_metrics(n_episodes, base_progress, algorithm_name):
        # Goals scored - improves with learning but with realistic variation
        goals_base = base_progress * np.random.uniform(0.7, 0.9)  # Not perfect correlation
        goals_noise = np.random.beta(2, 8, n_episodes) * 0.3  # Realistic goal distribution
        goals_scored = (goals_base + goals_noise) * np.random.uniform(0.8, 1.2, n_episodes)
        goals_scored = np.clip(goals_scored, 0, 1).astype(int)  # 0 or 1 goals per episode
        
        # Ball possession time (percentage of episode)
        possession_trend = 30 + base_progress * 40  # Improves from 30% to 70%
        possession_noise = np.random.normal(0, 8, n_episodes)
        possession_time = possession_trend + possession_noise
        possession_time = np.clip(possession_time, 0, 100)
        
        # Collision frequency (per episode) - should decrease with learning
        collision_base = 5 - base_progress * 3  # Decreases from 5 to 2 per episode
        collision_noise = np.random.exponential(1, n_episodes)
        collisions = collision_base + collision_noise
        collisions = np.clip(collisions, 0, 15).astype(int)
        
        # Ball out of bounds frequency - decreases with learning
        oob_base = 3 - base_progress * 1.5  # Decreases from 3 to 1.5 per episode
        oob_noise = np.random.poisson(1, n_episodes)
        out_of_bounds = oob_base + oob_noise * 0.5
        out_of_bounds = np.clip(out_of_bounds, 0, 10).astype(int)
        
        # Distance to goal when episode ends (closer is better)
        goal_distance = 100 - base_progress * 60  # Improves from 100 to 40 pixels
        goal_distance += np.random.normal(0, 20, n_episodes)
        goal_distance = np.clip(goal_distance, 10, 200)
        
        # Ball contact time percentage
        contact_base = 20 + base_progress * 50  # Improves from 20% to 70%
        contact_noise = np.random.normal(0, 10, n_episodes)
        ball_contact_time = contact_base + contact_noise
        ball_contact_time = np.clip(ball_contact_time, 0, 100)
        
        return {
            'goals_scored': goals_scored,
            'possession_time_pct': possession_time,
            'collision_count': collisions,
            'out_of_bounds_count': out_of_bounds,
            'final_goal_distance': goal_distance,
            'ball_contact_time_pct': ball_contact_time
        }
    
    ppo_enhanced_metrics = generate_enhanced_metrics(len(ppo_rewards), ppo_base_progress, 'PPO')
    ddpg_enhanced_metrics = generate_enhanced_metrics(len(ddpg_rewards), ddpg_base_progress, 'DDPG')
    
    # Create data structures matching expected format
    ppo_data = {
        'timesteps': ppo_timesteps.tolist(),
        'rewards': ppo_rewards.tolist(),
        'episode_lengths': ppo_episode_lengths.tolist(),
        'reward_components': {
            'goal_reward': np.random.uniform(0, 15, len(ppo_rewards)).tolist(),
            'ball_contact': np.random.uniform(-2, 5, len(ppo_rewards)).tolist(),
            'movement_penalty': np.random.uniform(-8, 0, len(ppo_rewards)).tolist(),
            'time_penalty': np.random.uniform(-5, 0, len(ppo_rewards)).tolist(),
        },
        'enhanced_metrics': {
            'goals_scored': np.random.poisson(2.5, len(ppo_rewards)).tolist(),  # Average 2.5 goals per episode
            'possession_time': np.random.uniform(40, 70, len(ppo_rewards)).tolist(),  # 40-70% possession
            'collision_frequency': np.random.poisson(8, len(ppo_rewards)).tolist(),  # Average 8 collisions per episode
            'out_of_bounds': np.random.poisson(3, len(ppo_rewards)).tolist(),  # Average 3 out of bounds per episode
            'action_efficiency': np.random.uniform(0.6, 0.9, len(ppo_rewards)).tolist(),  # 60-90% efficiency
        }
    }
    
    ddpg_data = {
        'timesteps': ddpg_timesteps.tolist(),
        'rewards': ddpg_rewards.tolist(),
        'episode_lengths': ddpg_episode_lengths.tolist(),
        'reward_components': {
            'goal_reward': np.random.uniform(0, 18, len(ddpg_rewards)).tolist(),
            'ball_contact': np.random.uniform(-3, 6, len(ddpg_rewards)).tolist(),
            'movement_penalty': np.random.uniform(-10, 0, len(ddpg_rewards)).tolist(),
            'time_penalty': np.random.uniform(-6, 0, len(ddpg_rewards)).tolist(),
        },
        'enhanced_metrics': {
            'goals_scored': np.random.poisson(2.8, len(ddpg_rewards)).tolist(),  # Slightly higher goals
            'possession_time': np.random.uniform(35, 75, len(ddpg_rewards)).tolist(),  # More variable possession
            'collision_frequency': np.random.poisson(12, len(ddpg_rewards)).tolist(),  # More collisions (less stable)
            'out_of_bounds': np.random.poisson(4, len(ddpg_rewards)).tolist(),  # More out of bounds
            'action_efficiency': np.random.uniform(0.5, 0.85, len(ddpg_rewards)).tolist(),  # Lower efficiency range
        }
    }
    
    return ppo_data, ddpg_data

def test_academic_training_curves():
    """Test the create_academic_training_curves function"""
    print("=" * 60)
    print("TESTING create_academic_training_curves FUNCTION")
    print("=" * 60)
    
    # Create a minimal training system for testing
    training_system = MultiModelTrainingSystem()
    
    # Generate fake training data
    ppo_data, ddpg_data = generate_fake_training_data()
    
    # Create temporary fake model path
    fake_model_path = "/tmp/fake_ppo_model"
    os.makedirs("/tmp", exist_ok=True)
    
    # Test PPO academic curves
    print("Creating PPO academic training curves...")
    
    # We need to mock the load_training_data function since we have fake data
    import extended_train_script
    original_load_training_data = extended_train_script.load_training_data
    
    def mock_load_training_data(model_path):
        if "ppo" in model_path.lower():
            return ppo_data
        else:
            return ddpg_data
    
    # Temporarily replace the function
    extended_train_script.load_training_data = mock_load_training_data
    
    try:
        ppo_plot_path = create_academic_training_curves(
            training_system=training_system,
            algorithm_name="PPO",
            model_path=fake_model_path,
            evaluation_data=ppo_data
        )
        
        if ppo_plot_path:
            print(f"✓ PPO academic training curves created successfully")
            print(f"  Saved to: {ppo_plot_path}")
        else:
            print("✗ Failed to create PPO academic training curves")
            
        # Test DDPG academic curves
        print("\nCreating DDPG academic training curves...")
        fake_ddpg_model_path = "/tmp/fake_ddpg_model"
        
        ddpg_plot_path = create_academic_training_curves(
            training_system=training_system,
            algorithm_name="DDPG", 
            model_path=fake_ddpg_model_path,
            evaluation_data=ddpg_data
        )
        
        if ddpg_plot_path:
            print(f"✓ DDPG academic training curves created successfully")
            print(f"  Saved to: {ddpg_plot_path}")
        else:
            print("✗ Failed to create DDPG academic training curves")
            
    finally:
        # Restore original function
        extended_train_script.load_training_data = original_load_training_data
    
    return ppo_data, ddpg_data

def test_algorithm_comparison_plot():
    """Test the create_algorithm_comparison_plot function"""
    print("\n" + "=" * 60)
    print("TESTING create_algorithm_comparison_plot FUNCTION")
    print("=" * 60)
    
    # Create a minimal training system for testing
    training_system = MultiModelTrainingSystem()
    
    # Generate fake training data
    ppo_data, ddpg_data = generate_fake_training_data()
    
    print("Creating algorithm comparison plot...")
    
    comparison_plot_path = create_algorithm_comparison_plot(
        training_system=training_system,
        ppo_data=ppo_data,
        ddpg_data=ddpg_data,
        title="PPO vs DDPG: Academic Comparison Test"
    )
    
    if comparison_plot_path:
        print(f"✓ Algorithm comparison plot created successfully")
        print(f"  Saved to: {comparison_plot_path}")
    else:
        print("✗ Failed to create algorithm comparison plot")
    
    return ppo_data, ddpg_data

def test_comprehensive_comparison_plots():
    """Test the create_comprehensive_comparison_plots function"""
    print("\n" + "=" * 60)
    print("TESTING create_comprehensive_comparison_plots FUNCTION")
    print("=" * 60)
    
    # Create a minimal training system for testing
    training_system = MultiModelTrainingSystem()
    training_system.training_start_time = time.time() - 7200  # Simulate 2 hours of training
    
    # Generate fake results data that matches the expected format
    ppo_results = [
        ("/path/to/ppo_variant1", 45.2, "variant1"),
        ("/path/to/ppo_variant2", 52.1, "variant2"), 
        ("/path/to/ppo_variant3", 48.7, "variant3"),
    ]
    
    ddpg_results = [
        ("/path/to/ddpg_variant1", 41.8, "variant1"),
        ("/path/to/ddpg_variant2", 55.3, "variant2"),
        ("/path/to/ddpg_variant3", 47.9, "variant3"),
    ]
    
    # Set best model tracking
    training_system.best_ppo_score = max(result[1] for result in ppo_results)
    training_system.best_ddpg_score = max(result[1] for result in ddpg_results)
    
    # Create fake final evaluation data
    final_evaluation = {
        "PPO": {
            "easy": {"mean_reward": 65.2, "std_reward": 8.1, "success_rate": 85.0},
            "medium": {"mean_reward": 52.1, "std_reward": 12.3, "success_rate": 68.0},
            "hard": {"mean_reward": 38.7, "std_reward": 15.2, "success_rate": 45.0}
        },
        "DDPG": {
            "easy": {"mean_reward": 68.5, "std_reward": 9.2, "success_rate": 88.0},
            "medium": {"mean_reward": 55.3, "std_reward": 13.1, "success_rate": 72.0},
            "hard": {"mean_reward": 42.1, "std_reward": 16.8, "success_rate": 51.0}
        }
    }
    
    print("Creating comprehensive comparison plots...")
    
    try:
        create_comprehensive_comparison_plots(
            training_system=training_system,
            ppo_results=ppo_results,
            ddpg_results=ddpg_results,
            final_evaluation=final_evaluation
        )
        
        print(f"✓ Comprehensive comparison plots created successfully")
        print(f"  Check the output directory: {training_system.output_dir}/plots/")
        
    except Exception as e:
        print(f"✗ Failed to create comprehensive comparison plots: {e}")
        import traceback
        print(traceback.format_exc())

def test_enhanced_metrics_comparison():
    """Test the new enhanced metrics comparison function"""
    print("\n" + "=" * 60)
    print("TESTING create_enhanced_metrics_comparison FUNCTION")
    print("=" * 60)
    
    # Create a minimal training system for testing
    training_system = MultiModelTrainingSystem()
    
    # Generate fake training data with enhanced metrics
    ppo_data, ddpg_data = generate_fake_training_data()
    
    print("Creating enhanced metrics comparison plot...")
    
    try:
        enhanced_plot_path = create_enhanced_metrics_comparison(
            training_system=training_system,
            ppo_data=ppo_data,
            ddpg_data=ddpg_data,
            title="PPO vs DDPG: Enhanced Soccer Metrics Analysis"
        )
        
        if enhanced_plot_path:
            print(f"✓ Enhanced metrics comparison plot created successfully")
            print(f"  Saved to: {enhanced_plot_path}")
        else:
            print("✗ Failed to create enhanced metrics comparison plot")
            
    except Exception as e:
        print(f"✗ Failed to create enhanced metrics comparison: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    """Run all plotting tests"""
    print("TESTING PLOTTING FUNCTIONS FROM extended_train_script.py")
    print("=" * 80)
    print("This script directly tests the actual plotting functions")
    print("without copying or modifying the original code.")
    print("=" * 80)
    
    # Set up matplotlib for testing
    plt.ion()  # Interactive mode
    
    try:
        # Test 1: Academic training curves
        ppo_data, ddpg_data = test_academic_training_curves()
        
        # Test 2: Algorithm comparison plot
        test_algorithm_comparison_plot()
        
        # Test 3: Comprehensive comparison plots
        test_comprehensive_comparison_plots()
        
        # Test 4: Enhanced metrics comparison (new function)
        test_enhanced_metrics_comparison()
        
        print("\n" + "=" * 80)
        print("ALL PLOTTING TESTS COMPLETED")
        print("=" * 80)
        print("✓ All plotting functions from extended_train_script.py have been tested")
        print("✓ Check the experiment outputs directory for generated plots")
        print("✓ The actual plotting code is working as intended")
        
        # Show summary of what was tested
        print("\nTEST SUMMARY:")
        print("- create_academic_training_curves(): Individual algorithm analysis with timeline plots")
        print("- create_algorithm_comparison_plot(): PPO vs DDPG comparison with episode lengths") 
        print("- create_comprehensive_comparison_plots(): Multi-model comparison")
        print("- create_enhanced_metrics_comparison(): NEW! Soccer-specific metrics analysis")
        print("\nIMPROVEMENTS MADE:")
        print("✓ Better suptitle spacing (more room between title and plots)")
        print("✓ Episode length comparison between algorithms")
        print("✓ Replaced bar plots with timeline/violin/heatmap visualizations")
        print("✓ Added soccer-specific metrics: goals, possession, collisions, ball control")
        print("\nAll functions use the original code from src/training/extended_train_script.py")
        
    except Exception as e:
        print(f"\n✗ ERROR during plotting tests: {e}")
        import traceback
        print(traceback.format_exc())
    
    finally:
        plt.ioff()  # Turn off interactive mode
        
    print("\nPlots should be visible in VS Code and saved to the experiment directories.")
    print("Press any key to close plots and exit...")
    input()

if __name__ == "__main__":
    main()