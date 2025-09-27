#!/usr/bin/env python3
"""
Academic Training Script for Soccer RL FYP

This script runs the academic-quality training pipeline with publication-ready
analysis and plots suitable for academic reports and presentations.

Usage:
    python run_academic_training.py [--timesteps 2500000] [--reward smooth]

Author: Ali Riyaz
Date: 2025
"""

import sys
import os
sys.path.append('src')

from training.extended_train_script import run_academic_training_pipeline

def main():
    print("="*80)
    print("ACADEMIC SOCCER RL TRAINING PIPELINE")
    print("="*80)
    print("This pipeline will:")
    print("• Train one optimal PPO model")
    print("• Train one optimal DDPG model")
    print("• Generate publication-quality training curves")
    print("• Create comprehensive algorithm comparison plots")
    print("• Provide detailed statistical analysis")
    print("• Save all artifacts for your academic report")
    print("="*80)

    # Configuration
    total_timesteps = 2500000  # 2.5M steps for robust performance
    reward_type = "smooth"     # Use the enhanced smooth reward function

    print(f"Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Reward function: {reward_type}")
    print(f"  Expected duration: ~5 hours")
    print("="*80)

    # Confirm before starting
    confirm = input("Start academic training? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Training cancelled.")
        return

    try:
        # Run the academic training pipeline
        results = run_academic_training_pipeline(
            total_timesteps=total_timesteps,
            reward_type=reward_type
        )

        # Display results summary
        if results and results.get('training_summary'):
            print("\n" + "="*80)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)

            summary = results['training_summary']
            experiment = summary['experiment_details']

            print(f"Training Duration: {experiment['training_duration_hours']:.2f} hours")
            print(f"Device Used: {experiment['device_used']}")
            print(f"Environment: {experiment['environment']}")

            print("\nGenerated Artifacts:")
            if results.get('comparison_plot_path'):
                print(f"  📊 Algorithm Comparison: {results['comparison_plot_path']}")
            if results.get('ppo_analysis_path'):
                print(f"  📈 PPO Analysis: {results['ppo_analysis_path']}")
            if results.get('ddpg_analysis_path'):
                print(f"  📈 DDPG Analysis: {results['ddpg_analysis_path']}")

            print("\nModel Performance Summary:")
            perf_summary = summary.get('performance_summary', {})

            for algo in ['PPO', 'DDPG']:
                if algo in perf_summary:
                    print(f"\n{algo} Results:")
                    for difficulty, metrics in perf_summary[algo].items():
                        print(f"  {difficulty.capitalize()}: "
                              f"{metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f} reward, "
                              f"{metrics['success_rate']:.1f}% success")

            # Statistical Analysis
            stats_analysis = summary.get('statistical_analysis', {})
            if stats_analysis:
                print("\nStatistical Analysis:")
                for difficulty, stats in stats_analysis.items():
                    significance = "✓ Significant" if stats['significant_difference'] else "~ No difference"
                    print(f"  {difficulty.capitalize()}: {stats['better_algorithm']} better "
                          f"(p={stats['p_value']:.4f}) {significance}")

            print("\n" + "="*80)
            print("All artifacts are ready for your academic report!")
            print("="*80)

        else:
            print("❌ Training completed but results may be incomplete.")
            print("Check the logs for details.")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)")
        print("Partial results may be available in the output directory.")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Check the logs for detailed error information.")

if __name__ == "__main__":
    main()