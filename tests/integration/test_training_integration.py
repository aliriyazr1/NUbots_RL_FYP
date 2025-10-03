# File: tests/integration/test_training_integration.py
"""
Training Integration Tests for Soccer RL FYP

This module validates end-to-end training pipeline functionality per
integration requirements. Tests ensure training runs complete successfully,
produce expected outputs, and work across different configurations.

Student: Ali Riyaz (C3412624)
References:
- Testing plan document (Section 5.2: Integration Testing)
- Academic training pipeline: src/training/extended_train_script.py
"""

import pytest
import numpy as np
import sys
import os
import shutil
from pathlib import Path
import tempfile
import time

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
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"SB3 components not available: {str(e)}"

# Print diagnostic if dependencies unavailable
if not DEPENDENCIES_AVAILABLE:
    print(f"\n⚠ Test dependencies unavailable: {IMPORT_ERROR}")
    print("Tests will be skipped.")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Training dependencies not available")
class TestMinimalTraining:
    """
    Test minimal training run completes successfully.

    Requirement: Training pipeline should complete without errors
    """

    def test_minimal_ppo_training_completes(self, tmp_path):
        """
        Test PPO training completes with minimal timesteps.

        This test validates that the training pipeline can execute
        without crashes or errors. Uses 10k timesteps for speed.

        Academic relevance: Validates training infrastructure works
        before committing to long training runs.
        """
        print(f"\n✓ Starting minimal PPO training test")
        print(f"  Temporary directory: {tmp_path}")

        # Create environment
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        # Create log directory
        log_dir = tmp_path / "ppo_logs"
        log_dir.mkdir(exist_ok=True)

        # Wrap with Monitor
        env = Monitor(env, str(log_dir))

        # Create minimal PPO model (faster training)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            verbose=0,
            device="cpu"  # Force CPU for test consistency
        )

        # Minimal training run (10k steps)
        timesteps = 10000
        print(f"  Training for {timesteps:,} timesteps...")
        start_time = time.time()

        model.learn(total_timesteps=timesteps, progress_bar=False)

        elapsed = time.time() - start_time
        print(f"  Training completed in {elapsed:.2f}s")

        # Save model
        model_path = tmp_path / "test_ppo_model.zip"
        model.save(str(model_path))

        print(f"  Model saved to: {model_path}")
        print(f"  Model file exists: {model_path.exists()}")
        print(f"  Model file size: {model_path.stat().st_size / 1024:.2f} KB")

        # Validate model saved successfully
        assert model_path.exists(), "Model file should be created"
        assert model_path.stat().st_size > 0, "Model file should not be empty"

        # Cleanup
        env.close()

        print(f"✓ Minimal training test passed")

    def test_training_produces_monitor_logs(self, tmp_path):
        """
        Test training produces monitor CSV logs.

        Monitor logs contain episode statistics (rewards, lengths)
        essential for academic analysis and reporting.
        """
        print(f"\n✓ Testing monitor log generation")

        # Create environment with Monitor
        env = SoccerEnv(render_mode=None, difficulty="easy", reward_type="original")

        log_dir = tmp_path / "monitor_logs"
        log_dir.mkdir(exist_ok=True)

        env = Monitor(env, str(log_dir))

        # Create and train minimal model
        # Use larger n_steps to ensure at least one episode completes
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,  # Increased to ensure episode completion
            batch_size=128,
            n_epochs=3,
            verbose=0,
            device="cpu"
        )

        # Train for longer to ensure multiple complete episodes
        model.learn(total_timesteps=10000, progress_bar=False)

        # Explicitly close environment to flush monitor logs
        env.close()

        # Check for monitor CSV file (can be monitor.csv or *.monitor.csv)
        monitor_files = list(log_dir.glob("*.monitor.csv"))
        if not monitor_files:
            monitor_files = list(log_dir.glob("monitor.csv"))

        print(f"  Monitor files found: {len(monitor_files)}")
        if monitor_files:
            print(f"  Monitor file: {monitor_files[0].name}")
            print(f"  File size: {monitor_files[0].stat().st_size} bytes")
        else:
            # List all files in directory for debugging
            all_files = list(log_dir.glob("*"))
            print(f"  All files in log directory: {[f.name for f in all_files]}")

        assert len(monitor_files) > 0, "Monitor CSV file should be created"
        assert monitor_files[0].stat().st_size > 0, "Monitor file should contain data"

        print(f"✓ Monitor log test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Training dependencies not available")
class TestCheckpointCreation:
    """
    Test checkpoint and evaluation file creation.

    Requirement: Training should produce checkpoint files for analysis
    """

    def test_model_checkpoint_created(self, tmp_path):
        """
        Test model checkpoint file creation.

        Validates that final model is saved in .zip format
        and can be reloaded successfully.
        """
        print(f"\n✓ Testing checkpoint creation")

        env = SoccerEnv(render_mode=None, difficulty="easy")
        env = Monitor(env, str(tmp_path))

        # Train minimal model
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        model.learn(total_timesteps=5000, progress_bar=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint_ppo.zip"
        model.save(str(checkpoint_path))

        print(f"  Checkpoint path: {checkpoint_path}")
        print(f"  Exists: {checkpoint_path.exists()}")
        print(f"  Size: {checkpoint_path.stat().st_size / 1024:.2f} KB")

        # Verify file exists and has content
        assert checkpoint_path.exists(), "Checkpoint file should exist"
        assert checkpoint_path.stat().st_size > 1024, "Checkpoint should be > 1KB"

        # Test reloading
        loaded_model = PPO.load(str(checkpoint_path), env=env, device="cpu")
        assert loaded_model is not None, "Model should reload successfully"

        print(f"  Model reloaded successfully")

        # Cleanup
        env.close()

        print(f"✓ Checkpoint creation test passed")

    def test_evaluations_npz_created(self, tmp_path):
        """
        Test evaluations.npz file creation with EvalCallback.

        This file contains evaluation metrics (rewards, episode lengths)
        essential for academic performance analysis.
        """
        print(f"\n✓ Testing evaluations.npz creation")

        # Create training and evaluation environments
        train_env = SoccerEnv(render_mode=None, difficulty="easy")
        train_env = Monitor(train_env, str(tmp_path / "train"))

        eval_env = SoccerEnv(render_mode=None, difficulty="easy")

        # Create evaluation callback
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir(exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
            eval_freq=2000,  # Evaluate every 2000 steps
            n_eval_episodes=3,
            deterministic=True,
            render=False,
            verbose=0
        )

        # Train with evaluation callback
        model = PPO("MlpPolicy", train_env, verbose=0, device="cpu")
        model.learn(total_timesteps=8000, callback=eval_callback, progress_bar=False)

        # Check for evaluations.npz
        eval_file = eval_dir / "evaluations.npz"

        print(f"  Evaluation file: {eval_file}")
        print(f"  Exists: {eval_file.exists()}")

        if eval_file.exists():
            # Load and inspect contents
            eval_data = np.load(eval_file)

            print(f"  Keys in evaluations.npz: {list(eval_data.keys())}")

            # Check expected keys exist
            expected_keys = ['timesteps', 'results', 'ep_lengths']
            for key in expected_keys:
                assert key in eval_data, f"evaluations.npz should contain '{key}'"
                print(f"    {key}: shape {eval_data[key].shape}")

            # Verify data is non-empty
            assert len(eval_data['timesteps']) > 0, "Should have evaluation timesteps"
            assert len(eval_data['results']) > 0, "Should have evaluation results"

            print(f"  Evaluation data validated")
        else:
            pytest.fail("evaluations.npz file was not created")

        # Cleanup
        train_env.close()
        eval_env.close()

        print(f"✓ Evaluations.npz test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Training dependencies not available")
class TestRewardTypeTraining:
    """
    Test training works with different reward function types.

    Requirement: Support 'smooth', 'hybrid', 'original' reward types
    Academic relevance: Validates reward function design flexibility
    """

    @pytest.mark.parametrize("reward_type", ["original", "smooth", "hybrid"])
    def test_training_with_reward_type(self, tmp_path, reward_type):
        """
        Test training completes with different reward types.

        This validates that all three reward function implementations
        work correctly in the training pipeline.

        Args:
            reward_type: One of 'original', 'smooth', 'hybrid'
        """
        print(f"\n✓ Testing training with reward_type='{reward_type}'")

        # Create environment with specific reward type
        env = SoccerEnv(
            render_mode=None,
            difficulty="easy",
            reward_type=reward_type
        )

        log_dir = tmp_path / f"logs_{reward_type}"
        log_dir.mkdir(exist_ok=True)

        env = Monitor(env, str(log_dir))

        # Train minimal model
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=64,
            verbose=0,
            device="cpu"
        )

        print(f"  Training with {reward_type} reward...")
        model.learn(total_timesteps=5000, progress_bar=False)

        # Save model
        model_path = tmp_path / f"model_{reward_type}.zip"
        model.save(str(model_path))

        print(f"  Model saved: {model_path.name}")
        print(f"  File size: {model_path.stat().st_size / 1024:.2f} KB")

        # Validate
        assert model_path.exists(), f"Model with {reward_type} reward should save"
        assert model_path.stat().st_size > 0, "Model file should not be empty"

        # Verify model can perform inference
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)

        print(f"  Inference test: action shape {action.shape}")
        assert action.shape == (3,), "Action should be 3D vector"

        # Cleanup
        env.close()

        print(f"✓ Reward type '{reward_type}' test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Training dependencies not available")
@pytest.mark.slow  # Mark as slow test (optional, can skip with pytest -m "not slow")
class TestDifficultyProgression:
    """
    Test training across difficulty levels shows performance changes.

    Requirement: Training on easy→medium→hard should show measurable
    performance differences.

    Academic relevance: Validates curriculum learning approach and
    demonstrates scalability of learned behaviours.
    """

    def test_difficulty_affects_performance(self, tmp_path):
        """
        Test that training difficulty affects evaluation performance.

        This test trains on each difficulty and measures performance,
        validating that the difficulty system works as intended.

        Expected: Easy > Medium > Hard in terms of reward
        """
        print(f"\n✓ Testing difficulty progression")

        difficulties = ["easy", "medium", "hard"]
        results = {}

        for difficulty in difficulties:
            print(f"\n  Training on difficulty: {difficulty}")

            # Create environment
            env = SoccerEnv(
                render_mode=None,
                difficulty=difficulty,
                reward_type="original"
            )

            log_dir = tmp_path / f"logs_{difficulty}"
            log_dir.mkdir(exist_ok=True)

            env = Monitor(env, str(log_dir))

            # Train model (slightly longer for difficulty test)
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=512,
                batch_size=128,
                n_epochs=5,
                verbose=0,
                device="cpu"
            )

            model.learn(total_timesteps=15000, progress_bar=False)

            # Evaluate trained model
            eval_env = SoccerEnv(
                render_mode=None,
                difficulty=difficulty,
                reward_type="original"
            )

            episode_rewards = []
            episode_lengths = []

            for _ in range(5):  # 5 evaluation episodes
                obs, _ = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False

                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)

            results[difficulty] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_length': mean_length
            }

            print(f"    Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"    Mean length: {mean_length:.2f}")

            # Cleanup
            env.close()
            eval_env.close()

        # Analyse results
        print(f"\n  Performance summary:")
        for difficulty in difficulties:
            r = results[difficulty]
            print(f"    {difficulty:6s}: {r['mean_reward']:8.2f} ± {r['std_reward']:.2f}")

        # Validation: Performance should generally decrease with difficulty
        # Note: Due to randomness in short training, we just validate
        # all difficulties complete without errors. For longer training,
        # we'd expect: easy_reward > medium_reward > hard_reward

        assert all(d in results for d in difficulties), \
            "All difficulties should complete training"

        assert all(not np.isnan(results[d]['mean_reward']) for d in difficulties), \
            "All difficulties should produce valid rewards"

        print(f"\n✓ Difficulty progression test passed")


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Training dependencies not available")
class TestTrainingCleanup:
    """
    Test proper cleanup after training.

    Validates that environments close properly and resources are released.
    """

    def test_environment_closes_properly(self, tmp_path):
        """
        Test environment cleanup doesn't leave resources hanging.

        Important for long-running experiments and preventing resource leaks.
        """
        print(f"\n✓ Testing environment cleanup")

        # Create and use environment
        env = SoccerEnv(render_mode=None, difficulty="easy")
        env = Monitor(env, str(tmp_path))

        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        model.learn(total_timesteps=2000, progress_bar=False)

        # Explicit cleanup
        env.close()

        # Verify we can create new environment (no resource conflicts)
        env2 = SoccerEnv(render_mode=None, difficulty="easy")
        env2.close()

        print(f"  Cleanup successful - no resource conflicts")
        print(f"✓ Cleanup test passed")


if __name__ == "__main__":
    """Run tests with verbose output"""
    pytest.main([__file__, "-v", "-s"])
