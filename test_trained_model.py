"""
Author: Ali Riyaz
Student Number: C3412624
Last Updated: 10/08/2025
"""

# test_trained_model.py - Watch the trained robot play by loading the correct models
import time, yaml
from stable_baselines3 import PPO, DDPG
from SoccerEnv.soccerenv import SoccerEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy import stats

import traceback
import sys

def load_field_config(config_path="SoccerEnv/field_config.yaml"):
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
    
def watch_trained_robot(model_name="soccer_rl_ppo_final", episodes=5, difficulty="medium", config_path="SoccerEnv/field_config.yaml", testing_mode=False):
    """
    Watch your trained robot play soccer with visual rendering
    """
    print(f"🎮 Loading and testing model: {model_name}")
    if testing_mode:
        print(f"🚀 Testing mode enabled - faster gameplay for better visibility")
    
    
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
                action, _ = model.predict(obs, deterministic=True)
                
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

def compare_models(config_path="SoccerEnv/field_config.yaml"):
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
            result = watch_trained_robot(model_name, episodes=3, difficulty=difficulty, config_path=config_path)
            
            if result:
                model_results[difficulty] = result
                print(f"   Summary: {result['success_rate']:.1f}% goals, {result['avg_reward']:.1f} avg reward")
            
            if len(difficulties) > 1:  # Only pause if testing multiple difficulties
                input("   Press Enter to continue...")
        results[model_name] = model_results
        
        if len(models_to_test) > 1:  # Only pause if testing multiple models
            input(f"\n✅ {model_name} testing complete. Press Enter for next model...")

    # Generate comparison summary
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for difficulty in difficulties:
        print(f"\n🎯 {difficulty.upper()} Difficulty:")
        print(f"{'Model':<15} {'Success Rate':<15} {'Avg Reward':<15} {'Avg Length':<12}")
        print("-" * 65)
        
        for model_name in models_to_test:
            if model_name in results and difficulty in results[model_name]:
                data = results[model_name][difficulty]
                model_short = model_name.replace("soccer_rl_", "").replace("_final", "").upper()
                print(f"{model_short:<15} {data['success_rate']:<15.1f}% {data['avg_reward']:<15.2f} {data['avg_length']:<12.1f}")
            else:
                model_short = model_name.replace("soccer_rl_", "").replace("_final", "").upper()
                print(f"{model_short:<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
    
    return results

def test_random_baseline(config_path="SoccerEnv/field_config.yaml"):
    """Test random policy for comparison"""
    print("🎲 Testing random policy (baseline)...")
    
    env = SoccerEnv(render_mode="human", config_path=config_path)
    
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(env.max_steps):
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

def detailed_model_comparison(episodes=5, difficulty="medium", config_path="SoccerEnv/field_config.yaml", testing_mode=True):
    """
    Test both PPO and DDPG models for detailed comparison with plots
    """
    print(f"🔬 DETAILED MODEL COMPARISON")
    print(f"Testing both models for {episodes} episodes each on {difficulty} difficulty")
    print("="*60)
    
    # Store results for plotting
    results = {
        'PPO': {'rewards': [], 'episode_lengths': [], 'timesteps': []},
        'DDPG': {'rewards': [], 'episode_lengths': [], 'timesteps': []}
    }
    
    models_to_test = [
        ('PPO', 'soccer_rl_ppo_final'),
        ('DDPG', 'soccer_rl_ddpg_final')
    ]
    
    for model_type, model_name in models_to_test:
        print(f"\n🤖 Testing {model_type} Model ({model_name})")
        print("-" * 40)
        
        try:
            # Load model
            if model_type == "PPO":
                model = PPO.load(model_name)
            else:
                model = DDPG.load(model_name)
            print(f"✅ {model_type} model loaded successfully!")
            
            # Test episodes
            episode_rewards = []
            episode_lengths = []
            cumulative_timesteps = []
            total_timesteps = 0
            goals_scored = 0
            
            env = SoccerEnv(render_mode="human", difficulty=difficulty, config_path=config_path, testing_mode=testing_mode)  # With rendering to see what's happening
            
            for episode in range(episodes):
                print(f"  Episode {episode + 1}/{episodes}...", end=" ")
                
                obs, _ = env.reset()
                episode_reward = 0
                steps = 0
                
                for step in range(env.max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    steps += 1
                    total_timesteps += 1
                    
                    if terminated:
                        if env._check_goal():
                            goals_scored += 1
                        break
                    elif truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                cumulative_timesteps.append(total_timesteps)
                
                print(f"Reward: {episode_reward:.2f}, Steps: {steps}")
            
            env.close()
            
            # Store results
            results[model_type]['rewards'] = episode_rewards
            results[model_type]['episode_lengths'] = episode_lengths
            results[model_type]['timesteps'] = cumulative_timesteps
            results[model_type]['goals_scored'] = goals_scored
            results[model_type]['success_rate'] = (goals_scored / episodes) * 100
            
            # Print summary
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)
            
            print(f"  📊 {model_type} Summary:")
            print(f"    Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"    Average Episode Length: {mean_length:.1f} ± {std_length:.1f} steps")
            print(f"    Goals Scored: {goals_scored}/{episodes} ({(goals_scored/episodes)*100:.1f}%)")
            
        except FileNotFoundError:
            print(f"❌ {model_type} model file '{model_name}' not found!")
            results[model_type] = None
    
    # Create plots - THIS IS THE KEY PART THAT WAS MISSING!
    print(f"\n📈 Creating comprehensive comparison plots...")
    create_comparison_plots(results, episodes, difficulty)
    
    return results

def create_comparison_plots(results, episodes, difficulty):
    """
    Create research-style comparison plots
    """
    print(f"\n📈 Creating comparison plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for consistency with research paper style
    colors = {'PPO': '#1f77b4', 'DDPG': '#2ca02c'}  # Blue, Green
    
    # Plot 1: Episode Rewards vs Episodes (not cumulative timesteps)
    ax1.set_title(f'Episode Rewards Comparison\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            episodes_range = range(1, episodes + 1)
            rewards = results[model_type]['rewards']
            
            # Plot individual episode rewards as line with markers
            ax1.plot(episodes_range, rewards, 
                    color=colors[model_type], 
                    marker='o', 
                    markersize=6,
                    linewidth=2.5,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add trend line
            z = np.polyfit(episodes_range, rewards, 1)
            p = np.poly1d(z)
            ax1.plot(episodes_range, p(episodes_range), 
                    color=colors[model_type], 
                    linestyle='--', 
                    alpha=0.5,
                    linewidth=1)
    
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(range(1, episodes + 1))
    
    # Plot 2: Episode Length vs Episodes
    ax2.set_title(f'Episode Length Comparison\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Length (steps)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            episodes_range = range(1, episodes + 1)
            lengths = results[model_type]['episode_lengths']
            
            # Plot episode lengths as line with markers
            ax2.plot(episodes_range, lengths, 
                    color=colors[model_type], 
                    marker='s', 
                    markersize=6,
                    linewidth=2.5,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add trend line
            z = np.polyfit(episodes_range, lengths, 1)
            p = np.poly1d(z)
            ax2.plot(episodes_range, p(episodes_range), 
                    color=colors[model_type], 
                    linestyle='--', 
                    alpha=0.5,
                    linewidth=1)
    
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xticks(range(1, episodes + 1))
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{difficulty}_{episodes}episodes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional cumulative performance plot (similar to research paper)
    create_cumulative_performance_plot(results, episodes, difficulty)

def create_cumulative_performance_plot(results, episodes, difficulty):
    """
    Create cumulative performance plots similar to the research paper
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'PPO': '#1f77b4', 'DDPG': '#2ca02c'}
    
    # Plot 1: Cumulative Average Reward
    ax1.set_title(f'Cumulative Average Reward\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Cumulative Average Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            rewards = results[model_type]['rewards']
            cumulative_avg_rewards = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
            episodes_range = range(1, episodes + 1)
            
            ax1.plot(episodes_range, cumulative_avg_rewards, 
                    color=colors[model_type], 
                    linewidth=3,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add confidence band (using standard error)
            cumulative_std = [np.std(rewards[:i+1]) / np.sqrt(i+1) for i in range(len(rewards))]
            ax1.fill_between(episodes_range, 
                           np.array(cumulative_avg_rewards) - np.array(cumulative_std),
                           np.array(cumulative_avg_rewards) + np.array(cumulative_std),
                           color=colors[model_type], alpha=0.2)
    
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(range(1, episodes + 1))
    
    # Plot 2: Cumulative Average Episode Length
    ax2.set_title(f'Cumulative Average Episode Length\n({episodes} episodes, {difficulty} difficulty)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Cumulative Average Episode Length (steps)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            lengths = results[model_type]['episode_lengths']
            cumulative_avg_lengths = [np.mean(lengths[:i+1]) for i in range(len(lengths))]
            episodes_range = range(1, episodes + 1)
            
            ax2.plot(episodes_range, cumulative_avg_lengths, 
                    color=colors[model_type], 
                    linewidth=3,
                    label=f'{model_type}',
                    alpha=0.8)
            
            # Add confidence band
            cumulative_std = [np.std(lengths[:i+1]) / np.sqrt(i+1) for i in range(len(lengths))]
            ax2.fill_between(episodes_range, 
                           np.array(cumulative_avg_lengths) - np.array(cumulative_std),
                           np.array(cumulative_avg_lengths) + np.array(cumulative_std),
                           color=colors[model_type], alpha=0.2)
    
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xticks(range(1, episodes + 1))
    
    plt.tight_layout()
    plt.savefig(f'cumulative_performance_{difficulty}_{episodes}episodes.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_statistical_analysis(results, episodes=5):
    """
    Provide detailed statistical analysis of the comparison
    """
    print(f"\n📊 DETAILED STATISTICAL ANALYSIS")
    print("="*50)
    
    for model_type in ['PPO', 'DDPG']:
        if results[model_type] is not None:
            rewards = results[model_type]['rewards']
            lengths = results[model_type]['episode_lengths']
            
            print(f"\n🤖 {model_type} Statistics:")
            print(f"  Rewards:")
            print(f"    Mean: {np.mean(rewards):.3f}")
            print(f"    Std:  {np.std(rewards):.3f}")
            print(f"    Min:  {np.min(rewards):.3f}")
            print(f"    Max:  {np.max(rewards):.3f}")
            print(f"    Median: {np.median(rewards):.3f}")
            
            print(f"  Episode Lengths:")
            print(f"    Mean: {np.mean(lengths):.1f}")
            print(f"    Std:  {np.std(lengths):.1f}")
            print(f"    Min:  {np.min(lengths)}")
            print(f"    Max:  {np.max(lengths)}")
            print(f"    Median: {np.median(lengths):.1f}")
    
    # Compare models if both exist
    if results['PPO'] is not None and results['DDPG'] is not None:
        ppo_rewards = results['PPO']['rewards']
        ddpg_rewards = results['DDPG']['rewards']
        
        print(f"\n🔄 Model Comparison:")
        print(f"  Reward Difference (PPO - DDPG):")
        print(f"    Mean: {np.mean(ppo_rewards) - np.mean(ddpg_rewards):.3f}")
        print(f"    PPO better episodes: {sum(1 for p, d in zip(ppo_rewards, ddpg_rewards) if p > d)}/{episodes}")
        
        # Simple t-test equivalent
        try:
            t_stat, p_value = stats.ttest_ind(ppo_rewards, ddpg_rewards)
            print(f"    Statistical significance (t-test): p={p_value:.4f}")
            if p_value < 0.05:
                print(f"    ✅ Significant difference detected (p < 0.05)")
            else:
                print(f"    ⚠️  No significant difference (p >= 0.05)")
        except:
            print(f"    ⚠️  Could not perform t-test (scipy not available)")

        # Create comparison plots
        create_comparison_plots(results, episodes, "analysis")
        
        # Create visualization
        try:
            plt.figure(figsize=(15, 5))
            
            # Reward comparison
            plt.subplot(1, 3, 1)
            plt.boxplot([ppo_rewards, ddpg_rewards], labels=['PPO', 'DDPG'])
            plt.title('Reward Distribution')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.3)
            
            # Success rate comparison
            plt.subplot(1, 3, 2)
            models = ['PPO', 'DDPG']
            success_rates = [results['PPO']['success_rate'], results['DDPG']['success_rate']]
            colors = ['skyblue', 'lightcoral']
            plt.bar(models, success_rates, color=colors)
            plt.title('Success Rate Comparison')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            
            # Episode length comparison
            ppo_lengths = results['PPO']['episode_lengths']
            ddpg_lengths = results['DDPG']['episode_lengths']
            plt.subplot(1, 3, 3)
            plt.boxplot([ppo_lengths, ddpg_lengths], labels=['PPO', 'DDPG'])
            plt.title('Episode Length Distribution')
            plt.ylabel('Steps per Episode')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('detailed_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📈 Detailed analysis visualization saved as 'detailed_model_comparison.png'")
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":

    # Load field configuration
    config_path = load_field_config()

    print("🤖 Model Testing Options:")
    print("1. Test PPO model")
    print("2. Test DDPG model") 
    #TODO: FIX THE DATA COLLECTION AND COMPARISON
    print("3. WIP: Compare both models")
    print("4. WIP: Detailed model comparison with statistical analysis")
    print("5. Test random baseline")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        print("\nSelect difficulty:")
        print("1. Easy")
        print("2. Medium") 
        print("3. Hard")
        diff_choice = input("Enter choice (1-3): ").strip()
        
        difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        difficulty = difficulty_map.get(diff_choice, "medium")
        
        watch_trained_robot("soccer_rl_ppo_final", episodes=5, difficulty=difficulty, config_path=config_path)
        
    elif choice == "2":
        print("\nSelect difficulty:")
        print("1. Easy")
        print("2. Medium") 
        print("3. Hard")
        diff_choice = input("Enter choice (1-3): ").strip()
        
        difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        difficulty = difficulty_map.get(diff_choice, "medium")
        
        watch_trained_robot("soccer_rl_ddpg_final", episodes=5, difficulty=difficulty, config_path=config_path)
        
    elif choice == "3":
        compare_models(config_path)
        
    elif choice == "4":
        print("\nSelect difficulty for detailed analysis:")
        print("1. Easy")
        print("2. Medium") 
        print("3. Hard")
        diff_choice = input("Enter choice (1-3): ").strip()
        
        difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
        difficulty = difficulty_map.get(diff_choice, "medium")
        
        results = detailed_model_comparison(episodes=5, difficulty=difficulty, config_path=config_path)
        detailed_statistical_analysis(results, episodes=5)
        
    elif choice == "5":
        test_random_baseline(config_path)
        
    else:
        print("Invalid choice. Testing PPO by default...")
        watch_trained_robot("soccer_rl_ppo_final", episodes=5, config_path=config_path)
