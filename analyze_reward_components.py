#!/usr/bin/env python3
"""
Analyze reward component contributions to verify the 84% goal progress claim.
Runs 100 episodes and collects detailed reward component statistics.
"""

from src.environments.soccerenv import SoccerEnv
import numpy as np
from collections import defaultdict

print('='*70)
print('REWARD COMPONENT ANALYSIS')
print('='*70)

# Initialize environment
env = SoccerEnv(difficulty='easy')

# Track component statistics
component_totals = defaultdict(list)
episode_count = 100

print(f'\nRunning {episode_count} episodes to collect reward component data...\n')

for episode in range(episode_count):
    obs, info = env.reset()
    episode_components = defaultdict(float)
    episode_reward = 0
    steps = 0

    while True:
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

        # Collect component data
        if hasattr(env, 'reward_components'):
            for component_name, component_value in env.reward_components.items():
                episode_components[component_name] += component_value

        if terminated or truncated:
            break

    # Store episode totals
    for component_name, total_value in episode_components.items():
        component_totals[component_name].append(total_value)

    if (episode + 1) % 20 == 0:
        print(f'  Completed {episode + 1}/{episode_count} episodes...')

print('\n' + '='*70)
print('REWARD COMPONENT STATISTICS')
print('='*70)

# Calculate statistics
all_components = []
for component_name, values in sorted(component_totals.items()):
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    all_components.append({
        'name': component_name,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'count': len(values)
    })

    print(f'\n{component_name}:')
    print(f'  Mean:  {mean_val:>10.2f}')
    print(f'  Std:   {std_val:>10.2f}')
    print(f'  Min:   {min_val:>10.2f}')
    print(f'  Max:   {max_val:>10.2f}')
    print(f'  Count: {len(values):>10} episodes')

# Calculate total contribution and percentages
print('\n' + '='*70)
print('COMPONENT CONTRIBUTION ANALYSIS')
print('='*70)

# Sum of all positive contributions
positive_contributions = {}
for comp in all_components:
    if comp['mean'] > 0:
        positive_contributions[comp['name']] = comp['mean']

total_positive = sum(positive_contributions.values())

print(f'\nTotal positive reward contribution: {total_positive:.2f}')
print('\nPercentage breakdown of positive components:')
print('-'*70)

# Sort by percentage descending
sorted_components = sorted(positive_contributions.items(), key=lambda x: x[1], reverse=True)

for component_name, mean_value in sorted_components:
    percentage = (mean_value / total_positive) * 100 if total_positive > 0 else 0
    print(f'  {component_name:30s}: {mean_value:>10.2f}  ({percentage:>6.2f}%)')

# Check if goal_progress dominates
if 'goal_progress' in positive_contributions:
    goal_progress_percentage = (positive_contributions['goal_progress'] / total_positive) * 100
    print('\n' + '='*70)
    print('THESIS CLAIM VERIFICATION')
    print('='*70)
    print(f'\nGoal Progress Component: {goal_progress_percentage:.2f}%')

    if goal_progress_percentage > 80:
        print(f'✅ CONFIRMED: Goal progress dominates reward signal ({goal_progress_percentage:.2f}% > 80%)')
    elif goal_progress_percentage > 50:
        print(f'⚠️  PARTIAL: Goal progress is significant but not dominant ({goal_progress_percentage:.2f}%)')
    else:
        print(f'❌ NOT CONFIRMED: Goal progress does not dominate ({goal_progress_percentage:.2f}% < 50%)')
else:
    print('\n❌ WARNING: goal_progress component not found in tracking data')

print('\n' + '='*70)

# Save results
with open('reward_component_analysis_results.txt', 'w') as f:
    f.write('='*70 + '\n')
    f.write('REWARD COMPONENT ANALYSIS RESULTS\n')
    f.write('='*70 + '\n\n')
    f.write(f'Episodes analyzed: {episode_count}\n')
    f.write(f'Environment difficulty: easy\n\n')

    f.write('COMPONENT STATISTICS:\n')
    f.write('-'*70 + '\n')
    for comp in sorted(all_components, key=lambda x: x['mean'], reverse=True):
        f.write(f"\n{comp['name']}:\n")
        f.write(f"  Mean: {comp['mean']:.2f}\n")
        f.write(f"  Std:  {comp['std']:.2f}\n")
        f.write(f"  Min:  {comp['min']:.2f}\n")
        f.write(f"  Max:  {comp['max']:.2f}\n")

    f.write('\n' + '='*70 + '\n')
    f.write('PERCENTAGE BREAKDOWN:\n')
    f.write('-'*70 + '\n')
    for component_name, mean_value in sorted_components:
        percentage = (mean_value / total_positive) * 100 if total_positive > 0 else 0
        f.write(f'{component_name:30s}: {percentage:6.2f}%\n')

    if 'goal_progress' in positive_contributions:
        f.write('\n' + '='*70 + '\n')
        f.write(f'Goal Progress Dominance: {goal_progress_percentage:.2f}%\n')
        f.write('='*70 + '\n')

print(f'\n📝 Results saved to: reward_component_analysis_results.txt')
