"""
Complete Data Explorer for evaluations.npz
Shows all data in your training logs for FYP analysis
"""

import numpy as np
import matplotlib.pyplot as plt

def explore_evaluation_data(file_path="logs/evaluations.npz"):
    """Complete analysis of evaluations.npz file"""
    
    print("=" * 60)
    print("🔍 COMPLETE EVALUATION DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    data = np.load(file_path)
    
    print("\n📋 FILE CONTENTS:")
    print(f"Keys available: {list(data.keys())}")
    print(f"File size: {file_path}")
    
    # Extract arrays
    timesteps = data['timesteps']
    results = data['results']  # Shape: (evaluations, episodes_per_eval)
    ep_lengths = data['ep_lengths']
    
    print(f"\n📊 DATA SHAPES:")
    print(f"Timesteps: {timesteps.shape} - {timesteps.dtype}")
    print(f"Results: {results.shape} - {results.dtype}")
    print(f"Episode lengths: {ep_lengths.shape} - {ep_lengths.dtype}")
    
    print(f"\n📈 TRAINING OVERVIEW:")
    print(f"Total evaluations: {len(timesteps)}")
    print(f"Episodes per evaluation: {results.shape[1]}")
    print(f"Training timesteps: {timesteps[0]:,} → {timesteps[-1]:,}")
    print(f"Total training duration: {timesteps[-1] - timesteps[0]:,} steps")
    
    # Calculate statistics for each evaluation
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    min_rewards = np.min(results, axis=1)
    max_rewards = np.max(results, axis=1)
    
    print(f"\n🎯 REWARD STATISTICS:")
    print(f"Initial mean reward: {mean_rewards[0]:.2f}")
    print(f"Final mean reward: {mean_rewards[-1]:.2f}")
    print(f"Best ever mean reward: {np.max(mean_rewards):.2f}")
    print(f"Worst ever mean reward: {np.min(mean_rewards):.2f}")
    print(f"Total improvement: {mean_rewards[-1] - mean_rewards[0]:+.2f}")
    print(f"Standard deviation (final): {std_rewards[-1]:.2f}")
    
    # Episode length analysis
    mean_ep_lengths = np.mean(ep_lengths, axis=1)
    print(f"\n⏱️ EPISODE LENGTH ANALYSIS:")
    print(f"Initial mean episode length: {mean_ep_lengths[0]:.1f}")
    print(f"Final mean episode length: {mean_ep_lengths[-1]:.1f}")
    print(f"Max episode length seen: {np.max(ep_lengths):.1f}")
    print(f"Min episode length seen: {np.min(ep_lengths):.1f}")
    
    print(f"\n📋 DETAILED EVALUATION BREAKDOWN:")
    print("Eval# | Timestep | Mean Reward | Std | Min | Max | Avg Episode Length")
    print("-" * 80)
    
    # Show every 5th evaluation + first and last few
    indices_to_show = []
    indices_to_show.extend([0, 1, 2])  # First 3
    indices_to_show.extend(range(4, len(timesteps)-3, 5))  # Every 5th
    indices_to_show.extend(range(max(len(timesteps)-3, 0), len(timesteps)))  # Last 3
    indices_to_show = sorted(list(set(indices_to_show)))  # Remove duplicates
    
    for i in indices_to_show:
        if i < len(timesteps):
            print(f"{i+1:4d}  | {timesteps[i]:8,} | {mean_rewards[i]:10.2f} | "
                  f"{std_rewards[i]:6.2f} | {min_rewards[i]:6.2f} | {max_rewards[i]:6.2f} | "
                  f"{mean_ep_lengths[i]:8.1f}")
    
    print(f"\n🔢 RAW DATA SAMPLES:")
    print("First evaluation episodes (individual rewards):")
    print(f"  {results[0]}")
    print("Last evaluation episodes (individual rewards):")
    print(f"  {results[-1]}")
    
    print(f"\n📊 LEARNING PROGRESS INDICATORS:")
    
    # Calculate moving averages
    window = min(5, len(mean_rewards))
    if len(mean_rewards) >= window:
        recent_avg = np.mean(mean_rewards[-window:])
        early_avg = np.mean(mean_rewards[:window])
        print(f"Early training average (first {window}): {early_avg:.2f}")
        print(f"Recent training average (last {window}): {recent_avg:.2f}")
        print(f"Improvement ratio: {recent_avg/early_avg:.2f}x")
    
    # Stability analysis
    if len(mean_rewards) >= 10:
        recent_stability = np.std(mean_rewards[-10:])
        early_stability = np.std(mean_rewards[:10])
        print(f"Training stability (recent 10 std): {recent_stability:.2f}")
        print(f"Training stability (early 10 std): {early_stability:.2f}")
    
    # Learning rate estimation
    if len(mean_rewards) > 1:
        improvements = np.diff(mean_rewards)
        positive_improvements = improvements[improvements > 0]
        print(f"Positive improvement steps: {len(positive_improvements)}/{len(improvements)}")
        if len(positive_improvements) > 0:
            print(f"Average improvement when learning: {np.mean(positive_improvements):.2f}")
    
    print(f"\n📈 QUICK VISUALIZATION:")
    
    # Simple text-based progress bar
    normalized_rewards = (mean_rewards - np.min(mean_rewards)) / (np.max(mean_rewards) - np.min(mean_rewards))
    print("Training Progress (each . = one evaluation):")
    progress_bar = ""
    for norm_reward in normalized_rewards[::max(1, len(normalized_rewards)//50)]:
        if norm_reward < 0.2:
            progress_bar += "."
        elif norm_reward < 0.4:
            progress_bar += "o"
        elif norm_reward < 0.6:
            progress_bar += "O"
        elif norm_reward < 0.8:
            progress_bar += "@"
        else:
            progress_bar += "#"
    print(f"Low: {progress_bar} :High")
    
    # Plot if matplotlib available
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Reward progress
        ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                        alpha=0.3, label='±1 Std Dev')
        ax1.set_xlabel('Training Timesteps')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        ax2.plot(timesteps, mean_ep_lengths, 'g-', linewidth=2)
        ax2.set_xlabel('Training Timesteps')
        ax2.set_ylabel('Mean Episode Length')
        ax2.set_title('Episode Length Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_overview.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Plot saved as 'evaluation_overview.png'")
        
    except ImportError:
        print("📊 Matplotlib not available - install for plots: pip install matplotlib")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"💡 Key takeaway: Your model improved from {mean_rewards[0]:.0f} to {mean_rewards[-1]:.0f} reward")
    print(f"🎯 For FYP report: This shows {'successful' if mean_rewards[-1] > mean_rewards[0] else 'unsuccessful'} learning!")
    
    return {
        'timesteps': timesteps,
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'episode_lengths': mean_ep_lengths,
        'total_improvement': mean_rewards[-1] - mean_rewards[0],
        'final_performance': mean_rewards[-1],
        'training_duration': timesteps[-1] - timesteps[0]
    }

if __name__ == "__main__":
    # Run the analysis
    results = explore_evaluation_data("logs/evaluations.npz")
    
    # Export summary for report
    print(f"\n📄 EXPORT FOR FYP REPORT:")
    print(f"Training Duration: {results['training_duration']:,} timesteps")
    print(f"Final Performance: {results['final_performance']:.1f} ± {results['std_rewards'][-1]:.1f}")
    print(f"Total Improvement: {results['total_improvement']:+.1f}")
    print(f"Learning Achieved: {'Yes' if results['total_improvement'] > 0 else 'No'}")