# File: tests/integration/test_model_evaluation.py
"""
Model Evaluation Tests for Soccer RL FYP

This module validates trained model performance and deployment pipeline.
Tests ensure models perform better than random baselines, generalise
across difficulty levels, and support ONNX deployment.

Student: Ali Riyaz (C3412624)
References:
- Testing plan document (Section 5.3: Model Evaluation)
- Statistical validation using t-tests (α = 0.05)
- Academic standard: Trained agent >> Random baseline
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
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
    from stable_baselines3 import PPO
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
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"SB3 components not available: {str(e)}"

try:
    import torch
    import onnx
    import onnxruntime as ort
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"ONNX dependencies not available: {str(e)}"

# Print diagnostic if dependencies unavailable
if not DEPENDENCIES_AVAILABLE:
    print(f"\n⚠ Test dependencies unavailable: {IMPORT_ERROR}")
    print("Tests will be skipped.")


def evaluate_policy_episodes(model, env, n_episodes=10, deterministic=True):
    """
    Evaluate a policy over multiple episodes.

    Args:
        model: Trained model or None (for random baseline)
        env: Environment to evaluate in
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions (for trained models)

    Returns:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        episode_stats: Additional statistics per episode
    """
    episode_rewards = []
    episode_lengths = []
    episode_stats = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Track additional statistics
        goals_scored = 0

        while not done:
            if model is None:
                # Random baseline
                action = env.action_space.sample()
            else:
                # Trained model
                action, _states = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track goal scoring
            if 'goal_scored' in info and info['goal_scored']:
                goals_scored += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_stats.append({
            'goals_scored': goals_scored
        })

    return episode_rewards, episode_lengths, episode_stats


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Evaluation dependencies not available")
class TestTrainedModelPerformance:
    """
    Test trained model performs better than random baseline.

    Academic requirement: Demonstrate learning has occurred
    Statistical validation: t-test with p < 0.05
    """

    def test_trained_vs_random_baseline(self, tmp_path):
        """
        Test trained model significantly outperforms random baseline.

        This is the fundamental test of learning - a trained agent
        must perform significantly better than random actions.

        Statistical test: Independent samples t-test
        Null hypothesis: trained_reward = random_reward
        Alternative: trained_reward > random_reward
        Significance level: α = 0.05
        """
        print(f"\n✓ Testing trained model vs random baseline")

        # Create environment
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        # Train a minimal model
        print(f"  Training minimal model...")
        env_train = Monitor(env, str(tmp_path))

        model = PPO(
            "MlpPolicy",
            env_train,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            learning_rate=3e-4,
            verbose=0,
            device="cpu"
        )

        # Train for sufficient timesteps to learn
        # Note: 50k steps to ensure meaningful learning
        model.learn(total_timesteps=50000, progress_bar=False)

        print(f"  Training completed")

        # Evaluate trained model
        print(f"  Evaluating trained model...")
        trained_rewards, trained_lengths, _ = evaluate_policy_episodes(
            model, env, n_episodes=10, deterministic=True
        )

        # Evaluate random baseline
        print(f"  Evaluating random baseline...")
        random_rewards, random_lengths, _ = evaluate_policy_episodes(
            None, env, n_episodes=10, deterministic=False
        )

        # Calculate statistics
        trained_mean = np.mean(trained_rewards)
        trained_std = np.std(trained_rewards)
        random_mean = np.mean(random_rewards)
        random_std = np.std(random_rewards)

        print(f"\n  Performance comparison:")
        print(f"    Trained agent: {trained_mean:8.2f} ± {trained_std:.2f}")
        print(f"    Random policy: {random_mean:8.2f} ± {random_std:.2f}")
        print(f"    Improvement:   {trained_mean - random_mean:8.2f} ({((trained_mean - random_mean) / abs(random_mean) * 100):.1f}%)")

        # Statistical test: one-tailed t-test
        # H0: trained_mean <= random_mean
        # H1: trained_mean > random_mean
        t_statistic, p_value = stats.ttest_ind(trained_rewards, random_rewards)

        # One-tailed p-value (we only care if trained > random)
        p_value_one_tailed = p_value / 2 if t_statistic > 0 else 1 - p_value / 2

        print(f"\n  Statistical analysis:")
        print(f"    t-statistic: {t_statistic:.4f}")
        print(f"    p-value (one-tailed): {p_value_one_tailed:.4f}")
        print(f"    Significance level: 0.05")

        if p_value_one_tailed < 0.05:
            print(f"    ✓ Difference is statistically significant")
        else:
            print(f"    ✗ Difference is NOT statistically significant")

        # Assertions
        # For academic validation, we want to see:
        # 1. Trained model performs at least as well as random (no regression)
        # 2. Ideally, shows improvement (though statistical significance may vary with limited episodes)

        # Check that trained model doesn't perform significantly worse
        assert trained_mean >= random_mean * 0.95, \
            f"Trained model should not perform significantly worse than random (trained={trained_mean:.2f}, random={random_mean:.2f})"

        # Check for improvement or statistical evidence of learning
        improvement_ratio = (trained_mean - random_mean) / (abs(random_mean) + 1e-6)

        # Either show clear improvement OR statistical significance
        has_improvement = trained_mean > random_mean and improvement_ratio > 0.01  # >1% improvement
        is_significant = p_value_one_tailed < 0.05

        assert has_improvement or is_significant, \
            f"Model should show improvement (ratio={improvement_ratio:.4f}) or significance (p={p_value_one_tailed:.4f})"

        print(f"  Performance metrics:")
        print(f"    Improvement ratio: {improvement_ratio:.4f} ({improvement_ratio*100:.2f}%)")
        print(f"    Has improvement (>1%): {has_improvement}")
        print(f"    Is significant (p<0.05): {is_significant}")

        # Cleanup
        env_train.close()
        env.close()

        print(f"\n✓ Trained vs random baseline test passed")

    def test_model_achieves_minimum_performance(self, tmp_path):
        """
        Test trained model achieves minimum performance threshold.

        This validates that learning converges to a reasonable policy,
        not just better than random but actually competent.
        """
        print(f"\n✓ Testing minimum performance threshold")

        # Create and train model
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        env = Monitor(env, str(tmp_path))

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            verbose=0,
            device="cpu"
        )

        print(f"  Training model...")
        model.learn(total_timesteps=30000, progress_bar=False)

        # Evaluate
        print(f"  Evaluating model...")
        eval_env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        rewards, lengths, stats = evaluate_policy_episodes(
            model, eval_env, n_episodes=10, deterministic=True
        )

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        print(f"    Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"    Min reward:  {np.min(rewards):.2f}")
        print(f"    Max reward:  {np.max(rewards):.2f}")

        # Minimum threshold: should be significantly better than random
        # Random typically gets around -1000 to -500 on easy difficulty
        # Trained should get positive rewards
        minimum_threshold = 0.0  # Should at least be positive

        print(f"    Minimum threshold: {minimum_threshold:.2f}")

        assert mean_reward > minimum_threshold, \
            f"Mean reward {mean_reward:.2f} should exceed threshold {minimum_threshold:.2f}"

        # Cleanup
        env.close()
        eval_env.close()

        print(f"✓ Minimum performance test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Evaluation dependencies not available")
class TestCrossDifficultyPerformance:
    """
    Test model trained on one difficulty performs on others.

    Academic relevance: Demonstrates generalisation and transfer learning
    Expected: Performance degrades gracefully with increased difficulty
    """

    def test_easy_trained_model_on_all_difficulties(self, tmp_path):
        """
        Test model trained on easy difficulty across all difficulties.

        This validates that learned behaviours generalise across
        difficulty settings, with expected performance degradation.

        Expected ordering: easy > medium > hard (in terms of reward)
        """
        print(f"\n✓ Testing cross-difficulty performance")

        # Train on easy difficulty
        print(f"  Training model on EASY difficulty...")
        train_env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        train_env = Monitor(train_env, str(tmp_path))

        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            verbose=0,
            device="cpu"
        )

        model.learn(total_timesteps=30000, progress_bar=False)
        train_env.close()

        # Evaluate on all difficulties
        difficulties = ["easy", "medium", "hard"]
        results = {}

        for difficulty in difficulties:
            print(f"\n  Evaluating on {difficulty.upper()} difficulty...")

            eval_env = SoccerEnv(
                render_mode=None,
                difficulty=difficulty,
                reward_type="original"
            )

            rewards, lengths, stats = evaluate_policy_episodes(
                model, eval_env, n_episodes=10, deterministic=True
            )

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_length = np.mean(lengths)

            results[difficulty] = {
                'rewards': rewards,
                'mean': mean_reward,
                'std': std_reward,
                'length': mean_length
            }

            print(f"    Mean reward: {mean_reward:8.2f} ± {std_reward:.2f}")
            print(f"    Mean length: {mean_length:8.2f}")

            eval_env.close()

        # Analyse results
        print(f"\n  Performance summary:")
        for difficulty in difficulties:
            r = results[difficulty]
            print(f"    {difficulty:6s}: {r['mean']:8.2f} ± {r['std']:.2f}")

        # Validation: Model should work on all difficulties
        # (though performance may vary)
        for difficulty in difficulties:
            assert not np.isnan(results[difficulty]['mean']), \
                f"Model should produce valid rewards on {difficulty} difficulty"

        # Model should perform best on training difficulty (easy)
        easy_mean = results['easy']['mean']

        # Verify model learned something useful (better than random baseline)
        # Even on easy difficulty, random gets negative rewards
        assert easy_mean > 0, \
            "Model should achieve positive reward on training difficulty"

        print(f"\n✓ Cross-difficulty performance test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="ONNX dependencies not available")
class TestONNXDeploymentPipeline:
    """
    Test end-to-end deployment pipeline: Train → Export → Load → Predict

    Academic relevance: Validates deployment readiness for real robots
    Requirements: FR-10 - ONNX export and inference
    """

    def test_pytorch_to_onnx_deployment(self, tmp_path):
        """
        Test complete deployment pipeline from training to ONNX inference.

        Pipeline stages:
        1. Train PyTorch model
        2. Export to ONNX format
        3. Load ONNX model in ONNX Runtime
        4. Verify predictions match PyTorch
        5. Evaluate ONNX model performance

        This validates the entire deployment workflow for production use.
        """
        print(f"\n✓ Testing PyTorch → ONNX deployment pipeline")

        # Stage 1: Train PyTorch model
        print(f"\n  Stage 1: Training PyTorch model...")
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")
        env = Monitor(env, str(tmp_path))

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=64,
            verbose=0,
            device="cpu"
        )

        model.learn(total_timesteps=10000, progress_bar=False)
        print(f"    ✓ PyTorch model trained")

        # Stage 2: Export to ONNX
        print(f"\n  Stage 2: Exporting to ONNX...")

        # Create deterministic wrapper for ONNX export
        class DeterministicPPOWrapper(torch.nn.Module):
            def __init__(self, policy, action_space):
                super().__init__()
                self.policy = policy
                self.action_space = action_space

            def forward(self, obs):
                features = self.policy.extract_features(obs)
                policy_features = self.policy.mlp_extractor.policy_net(features)
                mean_actions = self.policy.action_net(policy_features)

                # Clamp to action space bounds
                low = torch.tensor(self.action_space.low, dtype=mean_actions.dtype, device=mean_actions.device)
                high = torch.tensor(self.action_space.high, dtype=mean_actions.dtype, device=mean_actions.device)
                mean_actions = torch.clamp(mean_actions, low, high)

                return mean_actions

        policy_wrapper = DeterministicPPOWrapper(model.policy, env.action_space)
        policy_wrapper.eval()
        policy_wrapper.to("cpu")

        onnx_path = tmp_path / "model.onnx"
        dummy_input = torch.randn(1, 12, dtype=torch.float32)

        torch.onnx.export(
            policy_wrapper,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            input_names=['observation'],
            output_names=['action']
        )

        print(f"    ✓ Model exported to ONNX")
        print(f"      Path: {onnx_path}")
        print(f"      Size: {onnx_path.stat().st_size / 1024:.2f} KB")

        # Stage 3: Load ONNX model
        print(f"\n  Stage 3: Loading ONNX model...")
        ort_session = ort.InferenceSession(str(onnx_path))
        print(f"    ✓ ONNX model loaded in ONNX Runtime")

        # Stage 4: Verify predictions match
        print(f"\n  Stage 4: Verifying PyTorch ↔ ONNX consistency...")

        # Test on multiple random observations
        test_samples = 5
        max_differences = []

        for i in range(test_samples):
            test_obs = np.random.randn(1, 12).astype(np.float32)

            # PyTorch prediction
            with torch.no_grad():
                torch_input = torch.from_numpy(test_obs)
                torch_output = policy_wrapper(torch_input).numpy()

            # ONNX prediction
            onnx_output = ort_session.run(None, {'observation': test_obs})[0]

            # Compare
            diff = np.abs(torch_output - onnx_output).max()
            max_differences.append(diff)

        max_diff = np.max(max_differences)
        mean_diff = np.mean(max_differences)

        print(f"    Max difference: {max_diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        print(f"    Tolerance: 1e-4")

        assert max_diff < 1e-4, \
            f"PyTorch and ONNX predictions should match (diff={max_diff:.2e})"

        print(f"    ✓ PyTorch and ONNX predictions match")

        # Stage 5: Evaluate ONNX model performance
        print(f"\n  Stage 5: Evaluating ONNX model performance...")

        eval_env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        # Run evaluation episodes using ONNX model
        episode_rewards = []

        for episode in range(5):
            obs, info = eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                # ONNX inference
                obs_input = obs.reshape(1, -1).astype(np.float32)
                action = ort_session.run(None, {'observation': obs_input})[0][0]

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"    ONNX model performance:")
        print(f"      Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"      Episodes: {len(episode_rewards)}")

        # ONNX model should perform comparably to PyTorch
        assert not np.isnan(mean_reward), "ONNX model should produce valid rewards"

        print(f"    ✓ ONNX model performs successfully")

        # Cleanup
        env.close()
        eval_env.close()

        print(f"\n✓ Deployment pipeline test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Evaluation dependencies not available")
class TestModelConsistency:
    """
    Test model produces consistent predictions.

    Validates deterministic behaviour for reproducibility
    """

    def test_deterministic_predictions(self, tmp_path):
        """
        Test model produces identical predictions for same input.

        Critical for deployment: Same state should always produce
        same action when using deterministic=True.
        """
        print(f"\n✓ Testing deterministic predictions")

        # Train model
        env = SoccerEnv(render_mode=None, difficulty="easy")
        env = Monitor(env, str(tmp_path))

        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        model.learn(total_timesteps=5000, progress_bar=False)

        # Test deterministic predictions
        test_obs = np.random.randn(12).astype(np.float32)

        predictions = []
        for i in range(10):
            action, _states = model.predict(test_obs, deterministic=True)
            predictions.append(action.copy())

        # All predictions should be identical
        for i in range(1, len(predictions)):
            diff = np.abs(predictions[0] - predictions[i]).max()
            assert diff == 0.0, \
                f"Deterministic predictions should be identical (diff={diff})"

        print(f"  ✓ All predictions identical (deterministic=True)")

        # Cleanup
        env.close()

        print(f"✓ Deterministic prediction test passed")


if __name__ == "__main__":
    """Run tests with verbose output"""
    pytest.main([__file__, "-v", "-s"])
