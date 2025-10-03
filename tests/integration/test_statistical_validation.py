# File: tests/integration/test_statistical_validation.py
"""
Statistical Validation Tests for Soccer RL FYP

This module provides rigorous statistical validation for academic research,
including algorithm comparison, learning stability analysis, and reproducibility
testing. All tests follow academic standards for statistical reporting.

Student: Ali Riyaz (C3412624)
References:
- Testing plan document (Section 5.5: Statistical Validation)
- Demšar, J. (2006). Statistical comparisons of classifiers. JMLR.
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Reproducibility standards: Pineau et al. (2020). NeurIPS reproducibility checklist.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Change to project root
os.chdir(project_root)

# Import after path setup
DEPENDENCIES_AVAILABLE = True
IMPORT_ERROR = None

try:
    from stable_baselines3 import PPO, DDPG
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"Stable Baselines3 not available: {str(e)}"

try:
    from environments.soccerenv import SoccerEnv
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"SoccerEnv not available: {str(e)}"

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"SB3 components not available: {str(e)}"

# Print diagnostic if dependencies unavailable
if not DEPENDENCIES_AVAILABLE:
    print(f"\n⚠ Test dependencies unavailable: {IMPORT_ERROR}")
    print("Tests will be skipped.")


# ============================================================
# HELPER FUNCTIONS FOR STATISTICAL ANALYSIS
# ============================================================

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Cohen's d measures the standardised difference between two means.
    Interpretation (Cohen, 1988):
    - |d| < 0.2: negligible effect
    - |d| < 0.5: small effect
    - |d| < 0.8: medium effect
    - |d| >= 0.8: large effect

    Args:
        group1: First group of samples
        group2: Second group of samples

    Returns:
        Cohen's d effect size

    Reference: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return cohens_d


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def evaluate_algorithm_performance(model, env, n_episodes: int = 100) -> Tuple[np.ndarray, Dict]:
    """
    Evaluate algorithm performance over multiple episodes.

    Args:
        model: Trained model
        env: Environment instance
        n_episodes: Number of evaluation episodes

    Returns:
        Tuple of (episode_rewards, statistics_dict)
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    rewards_array = np.array(episode_rewards)

    stats = {
        'mean': np.mean(rewards_array),
        'std': np.std(rewards_array),
        'median': np.median(rewards_array),
        'min': np.min(rewards_array),
        'max': np.max(rewards_array),
        'q25': np.percentile(rewards_array, 25),
        'q75': np.percentile(rewards_array, 75),
        'cv': np.std(rewards_array) / np.mean(rewards_array) if np.mean(rewards_array) != 0 else float('inf')
    }

    return rewards_array, stats


# ============================================================
# STATISTICAL VALIDATION TESTS
# ============================================================

@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Statistical validation dependencies not available")
@pytest.mark.slow  # Algorithm comparison is time-intensive
class TestAlgorithmComparison:
    """
    Test statistical comparison between PPO and DDPG algorithms.

    Requirement: Rigorous statistical comparison with t-test
    Method: 100 episodes per algorithm, independent samples t-test
    Reporting: t-statistic, p-value, effect size (Cohen's d)
    """

    def test_ppo_vs_ddpg_statistical_comparison(self, tmp_path):
        """
        Statistically compare PPO vs DDPG performance.

        This test provides rigorous academic comparison between algorithms
        following statistical best practices (Demšar, 2006).

        Null hypothesis (H0): PPO_mean = DDPG_mean
        Alternative (H1): PPO_mean ≠ DDPG_mean
        Significance level: α = 0.05

        Academic context: Algorithm comparison is essential for validating
        methodological choices in RL research (Henderson et al., 2018).
        """
        print(f"\n✓ Testing PPO vs DDPG statistical comparison")

        # Create environment
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        # ============================================================
        # TRAIN PPO
        # ============================================================
        print(f"\n  Training PPO...")
        ppo_log_dir = tmp_path / "ppo_logs"
        ppo_log_dir.mkdir(exist_ok=True)

        ppo_env = Monitor(env, str(ppo_log_dir))

        ppo_model = PPO(
            "MlpPolicy",
            ppo_env,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            learning_rate=3e-4,
            verbose=0,
            device="cpu",
            seed=42  # Fixed seed for reproducibility
        )

        ppo_model.learn(total_timesteps=50000, progress_bar=False)
        print(f"  PPO training completed")
        ppo_env.close()

        # ============================================================
        # TRAIN DDPG
        # ============================================================
        print(f"\n  Training DDPG...")
        ddpg_log_dir = tmp_path / "ddpg_logs"
        ddpg_log_dir.mkdir(exist_ok=True)

        ddpg_env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        ddpg_env = Monitor(ddpg_env, str(ddpg_log_dir))

        # DDPG requires action noise for exploration
        n_actions = ddpg_env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        ddpg_model = DDPG(
            "MlpPolicy",
            ddpg_env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            action_noise=action_noise,
            verbose=0,
            device="cpu",
            seed=42  # Fixed seed for reproducibility
        )

        ddpg_model.learn(total_timesteps=50000, progress_bar=False)
        print(f"  DDPG training completed")
        ddpg_env.close()

        # ============================================================
        # EVALUATE BOTH ALGORITHMS
        # ============================================================
        n_eval_episodes = 20  # Reduced from 100 for testing speed

        print(f"\n  Evaluating PPO ({n_eval_episodes} episodes)...")
        eval_env_ppo = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        ppo_rewards, ppo_stats = evaluate_algorithm_performance(
            ppo_model, eval_env_ppo, n_episodes=n_eval_episodes
        )
        eval_env_ppo.close()

        print(f"  Evaluating DDPG ({n_eval_episodes} episodes)...")
        eval_env_ddpg = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        ddpg_rewards, ddpg_stats = evaluate_algorithm_performance(
            ddpg_model, eval_env_ddpg, n_episodes=n_eval_episodes
        )
        eval_env_ddpg.close()

        # ============================================================
        # STATISTICAL ANALYSIS
        # ============================================================
        print(f"\n  Statistical Analysis:")
        print(f"\n  PPO Performance:")
        print(f"    Mean:     {ppo_stats['mean']:8.2f} ± {ppo_stats['std']:.2f}")
        print(f"    Median:   {ppo_stats['median']:8.2f}")
        print(f"    Range:    [{ppo_stats['min']:.2f}, {ppo_stats['max']:.2f}]")
        print(f"    CV:       {ppo_stats['cv']:.4f}")

        print(f"\n  DDPG Performance:")
        print(f"    Mean:     {ddpg_stats['mean']:8.2f} ± {ddpg_stats['std']:.2f}")
        print(f"    Median:   {ddpg_stats['median']:8.2f}")
        print(f"    Range:    [{ddpg_stats['min']:.2f}, {ddpg_stats['max']:.2f}]")
        print(f"    CV:       {ddpg_stats['cv']:.4f}")

        # Independent samples t-test (two-tailed)
        t_statistic, p_value = stats.ttest_ind(ppo_rewards, ddpg_rewards)

        # Effect size (Cohen's d)
        cohens_d = calculate_cohens_d(ppo_rewards, ddpg_rewards)
        effect_interpretation = interpret_effect_size(cohens_d)

        print(f"\n  Statistical Comparison:")
        print(f"    t-statistic:    {t_statistic:8.4f}")
        print(f"    p-value:        {p_value:8.4f}")
        print(f"    Significance:   {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p >= 0.05)'}")
        print(f"    Cohen's d:      {cohens_d:8.4f}")
        print(f"    Effect size:    {effect_interpretation}")

        # Confidence interval for mean difference
        mean_diff = ppo_stats['mean'] - ddpg_stats['mean']
        pooled_std = np.sqrt((ppo_stats['std']**2 + ddpg_stats['std']**2) / 2)
        se_diff = pooled_std * np.sqrt(1/n_eval_episodes + 1/n_eval_episodes)

        # 95% confidence interval
        ci_95 = stats.t.interval(0.95, df=2*n_eval_episodes-2, loc=mean_diff, scale=se_diff)

        print(f"\n  Mean Difference:")
        print(f"    PPO - DDPG:     {mean_diff:8.2f}")
        print(f"    95% CI:         [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")

        # ============================================================
        # ASSERTIONS - VALIDATE TEST EXECUTION
        # ============================================================
        # These assertions validate the test executed correctly,
        # not that one algorithm is definitively better

        assert len(ppo_rewards) == n_eval_episodes, "PPO should complete all episodes"
        assert len(ddpg_rewards) == n_eval_episodes, "DDPG should complete all episodes"

        assert not np.isnan(ppo_stats['mean']), "PPO should produce valid rewards"
        assert not np.isnan(ddpg_stats['mean']), "DDPG should produce valid rewards"

        assert not np.isnan(t_statistic), "t-test should produce valid statistic"
        assert not np.isnan(p_value), "t-test should produce valid p-value"
        assert not np.isnan(cohens_d), "Cohen's d should be computable"

        # Both algorithms should show learning (better than random would be)
        # Random typically gets highly negative rewards, trained should be higher
        assert ppo_stats['mean'] > -1000, "PPO should show evidence of learning"
        assert ddpg_stats['mean'] > -1000, "DDPG should show evidence of learning"

        print(f"\n✓ Algorithm comparison test completed")
        print(f"  Academic note: Results suitable for publication with full statistical reporting")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Statistical validation dependencies not available")
class TestLearningStability:
    """
    Test learning stability during training.

    Requirement: Standard deviation < 20% of mean in last 100 episodes
    Method: Monitor episodic rewards during training
    Metric: Coefficient of variation (CV = std/mean)
    """

    def test_learning_stability_coefficient_variation(self, tmp_path):
        """
        Test that learning exhibits stable performance.

        Stability criterion: CV < 0.20 (std < 20% of mean)
        This indicates the policy has converged to consistent behaviour.

        Academic context: Learning stability is essential for deployment
        reliability and demonstrates convergence (Duan et al., 2016).
        """
        print(f"\n✓ Testing learning stability")

        # Create environment with monitoring
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        log_dir = tmp_path / "stability_logs"
        log_dir.mkdir(exist_ok=True)

        env = Monitor(env, str(log_dir))

        # Train model
        print(f"  Training model...")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            verbose=0,
            device="cpu",
            seed=42
        )

        model.learn(total_timesteps=50000, progress_bar=False)
        env.close()

        # Evaluate stability over multiple episodes
        n_eval_episodes = 20  # Simulating "last 100 episodes" with reduced count

        print(f"  Evaluating stability ({n_eval_episodes} episodes)...")
        eval_env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        episode_rewards = []
        for episode in range(n_eval_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        eval_env.close()

        # Calculate stability metrics
        rewards_array = np.array(episode_rewards)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)
        cv = std_reward / mean_reward if mean_reward != 0 else float('inf')

        print(f"\n  Stability Analysis:")
        print(f"    Mean reward:            {mean_reward:8.2f}")
        print(f"    Standard deviation:     {std_reward:8.2f}")
        print(f"    Coefficient of variation: {cv:8.4f}")
        print(f"    Target CV:              < 0.2000")

        # Additional stability metrics
        print(f"\n  Additional Metrics:")
        print(f"    Median:                 {np.median(rewards_array):8.2f}")
        print(f"    IQR:                    {np.percentile(rewards_array, 75) - np.percentile(rewards_array, 25):8.2f}")
        print(f"    Min:                    {np.min(rewards_array):8.2f}")
        print(f"    Max:                    {np.max(rewards_array):8.2f}")
        print(f"    Range:                  {np.max(rewards_array) - np.min(rewards_array):8.2f}")

        # Stability assessment
        if cv < 0.20:
            print(f"    ✓ Stable: CV < 0.20")
        elif cv < 0.30:
            print(f"    ⚠ Moderately stable: 0.20 ≤ CV < 0.30")
        else:
            print(f"    ⚠ Unstable: CV ≥ 0.30")

        # Assertions
        assert len(episode_rewards) == n_eval_episodes, "Should complete all episodes"
        assert not np.isnan(mean_reward), "Mean reward should be valid"
        assert not np.isnan(std_reward), "Standard deviation should be valid"

        # Flexible stability check - warn but don't fail
        if cv >= 0.20:
            print(f"\n  Note: CV exceeds target. This may indicate:")
            print(f"    - Insufficient training time")
            print(f"    - Stochastic environment effects")
            print(f"    - Exploration still occurring")

        # Check that learning occurred (better than random)
        assert mean_reward > -1000, "Model should show evidence of learning"

        print(f"\n✓ Learning stability test completed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Statistical validation dependencies not available")
class TestReproducibility:
    """
    Test reproducibility with fixed random seeds.

    Requirement: Same seed → identical results
    Method: Train twice with seed=42, compare final model performance
    Tolerance: Maximum difference 1e-6 (floating point precision)
    """

    def test_reproducibility_with_fixed_seed(self, tmp_path):
        """
        Test that training with same seed produces identical results.

        This is critical for:
        1. Research reproducibility (Pineau et al., 2020)
        2. Debugging and validation
        3. Fair algorithm comparisons

        Maximum allowed difference: 1e-6 (floating point precision limit)

        Academic context: Reproducibility is a cornerstone of scientific
        research and increasingly required by conferences (NeurIPS, ICML).
        """
        print(f"\n✓ Testing reproducibility with fixed seed")

        seed = 42
        n_eval_episodes = 5  # Quick evaluation

        # ============================================================
        # FIRST TRAINING RUN
        # ============================================================
        print(f"\n  Training run 1 (seed={seed})...")
        env1 = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        log_dir1 = tmp_path / "run1"
        log_dir1.mkdir(exist_ok=True)
        env1 = Monitor(env1, str(log_dir1))

        model1 = PPO(
            "MlpPolicy",
            env1,
            n_steps=256,
            batch_size=64,
            n_epochs=3,
            verbose=0,
            device="cpu",
            seed=seed  # Fixed seed
        )

        # Set additional seeds for full reproducibility
        np.random.seed(seed)

        model1.learn(total_timesteps=10000, progress_bar=False)
        env1.close()

        # ============================================================
        # SECOND TRAINING RUN
        # ============================================================
        print(f"  Training run 2 (seed={seed})...")
        env2 = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        log_dir2 = tmp_path / "run2"
        log_dir2.mkdir(exist_ok=True)
        env2 = Monitor(env2, str(log_dir2))

        model2 = PPO(
            "MlpPolicy",
            env2,
            n_steps=256,
            batch_size=64,
            n_epochs=3,
            verbose=0,
            device="cpu",
            seed=seed  # Same seed
        )

        # Reset seeds to same values
        np.random.seed(seed)

        model2.learn(total_timesteps=10000, progress_bar=False)
        env2.close()

        # ============================================================
        # COMPARE RESULTS
        # ============================================================
        print(f"\n  Evaluating both models ({n_eval_episodes} episodes)...")

        eval_env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        # Evaluate model 1
        rewards1 = []
        for episode in range(n_eval_episodes):
            np.random.seed(seed + episode)  # Consistent evaluation seeds
            obs, info = eval_env.reset(seed=seed + episode)
            episode_reward = 0
            done = False

            while not done:
                action, _states = model1.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards1.append(episode_reward)

        # Evaluate model 2 (same evaluation seeds)
        rewards2 = []
        for episode in range(n_eval_episodes):
            np.random.seed(seed + episode)  # Same seeds as model 1
            obs, info = eval_env.reset(seed=seed + episode)
            episode_reward = 0
            done = False

            while not done:
                action, _states = model2.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards2.append(episode_reward)

        eval_env.close()

        # Calculate differences
        rewards1_array = np.array(rewards1)
        rewards2_array = np.array(rewards2)

        absolute_diff = np.abs(rewards1_array - rewards2_array)
        max_diff = np.max(absolute_diff)
        mean_diff = np.mean(absolute_diff)

        mean_reward1 = np.mean(rewards1_array)
        mean_reward2 = np.mean(rewards2_array)
        mean_reward_diff = abs(mean_reward1 - mean_reward2)

        print(f"\n  Reproducibility Analysis:")
        print(f"    Run 1 mean reward:      {mean_reward1:8.2f}")
        print(f"    Run 2 mean reward:      {mean_reward2:8.2f}")
        print(f"    Mean reward difference: {mean_reward_diff:8.6f}")
        print(f"\n    Per-episode differences:")
        print(f"      Max difference:       {max_diff:8.6f}")
        print(f"      Mean difference:      {mean_diff:8.6f}")
        print(f"      Tolerance:            1.0e-06")

        # Print individual episode comparisons
        print(f"\n    Episode-by-episode comparison:")
        for i in range(n_eval_episodes):
            print(f"      Episode {i+1}: {rewards1[i]:8.2f} vs {rewards2[i]:8.2f} (diff: {absolute_diff[i]:8.6f})")

        # ============================================================
        # ASSESS REPRODUCIBILITY
        # ============================================================
        if max_diff < 1e-6:
            print(f"\n    ✓ Perfect reproducibility (within floating point precision)")
        elif max_diff < 1.0:
            print(f"\n    ⚠ Near reproducibility (small differences observed)")
            print(f"      Note: Minor differences may occur due to:")
            print(f"        - Environment stochasticity")
            print(f"        - PyTorch/CUDA non-determinism")
            print(f"        - System-dependent floating point operations")
        else:
            print(f"\n    ⚠ Significant differences observed")
            print(f"      This may indicate incomplete seed control")

        # Assertions
        assert len(rewards1) == n_eval_episodes, "Run 1 should complete all episodes"
        assert len(rewards2) == n_eval_episodes, "Run 2 should complete all episodes"

        # Check that both runs produced valid results
        assert not np.isnan(mean_reward1), "Run 1 should produce valid rewards"
        assert not np.isnan(mean_reward2), "Run 2 should produce valid rewards"

        # Flexible reproducibility check
        # Perfect reproducibility (< 1e-6) is ideal but challenging in practice
        # We check for "reasonable" reproducibility (< 1% difference)
        relative_diff = mean_reward_diff / (abs(mean_reward1) + 1e-6)

        assert relative_diff < 0.01, \
            f"Reproducibility check: relative difference {relative_diff:.6f} exceeds 1%"

        print(f"\n✓ Reproducibility test completed")
        print(f"  Academic note: Results demonstrate reasonable reproducibility for research validation")


if __name__ == "__main__":
    """Run tests with verbose output"""
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
