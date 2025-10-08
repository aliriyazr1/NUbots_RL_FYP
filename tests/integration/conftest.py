"""
Pytest configuration and fixtures for integration tests.

Provides flexible model path specification via:
1. Command line: pytest --ppo-model=path/to/model.zip
2. Environment variables: PPO_MODEL_PATH=path/to/model.zip pytest
3. TrainGUI experiment selection: pytest --experiment=multi_model_training_20250910_213419
4. Latest experiment: pytest --use-latest-experiment
5. Default paths with automatic fallback to training if not found

Example usage:
    # Test with custom DDPG model (full path)
    pytest --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip

    # Test with experiment name (searches for best_model.zip)
    pytest --experiment=multi_model_training_20250910_213419

    # Test with latest experiment
    pytest --use-latest-experiment

    # List available experiments
    pytest --list-experiments

    # Test with environment variables
    export PPO_MODEL_PATH=custom/path/ppo_model.zip
    export DDPG_MODEL_PATH=custom/path/ddpg_model.zip
    pytest
"""

import os
import sys
import pytest
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO, DDPG
from src.environments.soccerenv import SoccerEnv


def pytest_addoption(parser):
    """Add command-line options for specifying model paths."""
    parser.addoption(
        "--ppo-model",
        action="store",
        default=None,
        help="Path to pre-trained PPO model (default: models/ppo/soccer_rl_ppo_final.zip)"
    )
    parser.addoption(
        "--ddpg-model",
        action="store",
        default=None,
        help="Path to pre-trained DDPG model (default: models/ddpg/soccer_rl_ddpg_final.zip)"
    )
    parser.addoption(
        "--experiment",
        action="store",
        default=None,
        help="TrainGUI experiment name (e.g., multi_model_training_20250910_213419). Automatically finds best models."
    )
    parser.addoption(
        "--use-latest-experiment",
        action="store_true",
        default=False,
        help="Use models from the most recent trainGUI experiment"
    )
    parser.addoption(
        "--list-experiments",
        action="store_true",
        default=False,
        help="List all available trainGUI experiments and exit"
    )


def pytest_configure(config):
    """
    Configure pytest - handle special flags like --list-experiments.
    """
    if config.getoption("--list-experiments"):
        print("\n" + "="*80)
        print("Available TrainGUI Experiments")
        print("="*80)

        experiments = list_available_experiments()

        if not experiments:
            print("\nNo experiments found in experiments/archives/")
            print("\nMake sure you have run trainGUI and saved experiments.")
        else:
            print(f"\nFound {len(experiments)} experiment(s):\n")

            for exp_name, has_ppo, has_ddpg, ppo_path, ddpg_path in experiments:
                print(f"📁 {exp_name}")

                if has_ppo:
                    rel_path = ppo_path.relative_to(Path("experiments/archives") / exp_name)
                    print(f"   ✓ PPO:  {rel_path}")
                else:
                    print(f"   ✗ PPO:  No model found")

                if has_ddpg:
                    rel_path = ddpg_path.relative_to(Path("experiments/archives") / exp_name)
                    print(f"   ✓ DDPG: {rel_path}")
                else:
                    print(f"   ✗ DDPG: No model found")

                print()

            print("\nUsage examples:")
            print(f"  pytest --experiment={experiments[0][0]}")
            print(f"  pytest --use-latest-experiment")

        print("="*80 + "\n")
        pytest.exit("Experiment list displayed", returncode=0)


@pytest.fixture(scope="session")
def ppo_model_path(request):
    """
    Get PPO model path from (in priority order):
    1. Command-line argument (--ppo-model)
    2. Environment variable (PPO_MODEL_PATH)
    3. GUI selection file (.test_model_selection.json)
    4. Experiment selection (--experiment or --use-latest-experiment)
    5. Default path (models/ppo/soccer_rl_ppo_final.zip)

    Returns Path object or None if no valid path found.
    """
    # Check command-line argument
    cli_path = request.config.getoption("--ppo-model")
    if cli_path:
        path = Path(cli_path)
        if path.exists():
            print(f"\n[PPO Model] Using command-line path: {path}")
            return path
        else:
            print(f"\n[PPO Model] Warning: Command-line path not found: {path}")

    # Check environment variable
    env_path = os.environ.get("PPO_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            print(f"\n[PPO Model] Using environment variable path: {path}")
            return path
        else:
            print(f"\n[PPO Model] Warning: Environment variable path not found: {path}")

    # Check GUI selection file
    import json
    gui_config_file = Path(".test_model_selection.json")
    if gui_config_file.exists():
        try:
            with open(gui_config_file, 'r') as f:
                config = json.load(f)
                if 'ppo_model' in config:
                    path = Path(config['ppo_model'])
                    if path.exists():
                        print(f"\n[PPO Model] Using GUI selection: {path}")
                        return path
        except Exception as e:
            print(f"\n[PPO Model] Warning: Error reading GUI config: {e}")

    # Check experiment selection
    experiment_name = request.config.getoption("--experiment")
    if not experiment_name and request.config.getoption("--use-latest-experiment"):
        experiment_name = find_latest_experiment()
        if experiment_name:
            print(f"\n[PPO Model] Using latest experiment: {experiment_name}")

    if experiment_name:
        path = find_experiment_models(experiment_name, 'ppo')
        if path:
            print(f"\n[PPO Model] Using experiment model: {path}")
            return path
        else:
            print(f"\n[PPO Model] Warning: No PPO model found in experiment {experiment_name}")

    # Check default path
    default_path = Path("models/ppo/soccer_rl_ppo_final.zip")
    if default_path.exists():
        print(f"\n[PPO Model] Using default path: {default_path}")
        return default_path

    print("\n[PPO Model] Warning: No valid PPO model path found. Tests requiring pre-trained PPO will be skipped or will train a new model.")
    return None


@pytest.fixture(scope="session")
def ddpg_model_path(request):
    """
    Get DDPG model path from (in priority order):
    1. Command-line argument (--ddpg-model)
    2. Environment variable (DDPG_MODEL_PATH)
    3. GUI selection file (.test_model_selection.json)
    4. Experiment selection (--experiment or --use-latest-experiment)
    5. Default path (models/ddpg/soccer_rl_ddpg_final.zip)

    Returns Path object or None if no valid path found.
    """
    # Check command-line argument
    cli_path = request.config.getoption("--ddpg-model")
    if cli_path:
        path = Path(cli_path)
        if path.exists():
            print(f"\n[DDPG Model] Using command-line path: {path}")
            return path
        else:
            print(f"\n[DDPG Model] Warning: Command-line path not found: {path}")

    # Check environment variable
    env_path = os.environ.get("DDPG_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            print(f"\n[DDPG Model] Using environment variable path: {path}")
            return path
        else:
            print(f"\n[DDPG Model] Warning: Environment variable path not found: {path}")

    # Check GUI selection file
    import json
    gui_config_file = Path(".test_model_selection.json")
    if gui_config_file.exists():
        try:
            with open(gui_config_file, 'r') as f:
                config = json.load(f)
                if 'ddpg_model' in config:
                    path = Path(config['ddpg_model'])
                    if path.exists():
                        print(f"\n[DDPG Model] Using GUI selection: {path}")
                        return path
        except Exception as e:
            print(f"\n[DDPG Model] Warning: Error reading GUI config: {e}")

    # Check experiment selection
    experiment_name = request.config.getoption("--experiment")
    if not experiment_name and request.config.getoption("--use-latest-experiment"):
        experiment_name = find_latest_experiment()

    if experiment_name:
        path = find_experiment_models(experiment_name, 'ddpg')
        if path:
            print(f"\n[DDPG Model] Using experiment model: {path}")
            return path
        else:
            print(f"\n[DDPG Model] Warning: No DDPG model found in experiment {experiment_name}")

    # Check default path
    default_path = Path("models/ddpg/soccer_rl_ddpg_final.zip")
    if default_path.exists():
        print(f"\n[DDPG Model] Using default path: {default_path}")
        return default_path

    print("\n[DDPG Model] Warning: No valid DDPG model path found. Tests requiring pre-trained DDPG will be skipped or will train a new model.")
    return None


@pytest.fixture(scope="session")
def pretrained_ppo_model(ppo_model_path):
    """
    Load pre-trained PPO model if available.

    Returns:
        PPO model instance or None if no model available.
    """
    if ppo_model_path is None:
        return None

    try:
        model = PPO.load(str(ppo_model_path), device="cpu")
        print(f"[PPO Model] Successfully loaded from {ppo_model_path}")
        return model
    except Exception as e:
        print(f"[PPO Model] Error loading model: {e}")
        return None


@pytest.fixture(scope="session")
def pretrained_ddpg_model(ddpg_model_path):
    """
    Load pre-trained DDPG model if available.

    Returns:
        DDPG model instance or None if no model available.
    """
    if ddpg_model_path is None:
        return None

    try:
        model = DDPG.load(str(ddpg_model_path), device="cpu")
        print(f"[DDPG Model] Successfully loaded from {ddpg_model_path}")
        return model
    except Exception as e:
        print(f"[DDPG Model] Error loading model: {e}")
        return None


@pytest.fixture
def basic_env():
    """
    Create a basic SoccerEnv instance for testing.

    Returns:
        SoccerEnv instance.
    """
    env = SoccerEnv(difficulty='medium', render_mode=None)
    yield env
    env.close()


@pytest.fixture
def test_config():
    """
    Return test configuration dictionary.

    Returns:
        dict: Test configuration with standard parameters.
    """
    return {
        'difficulty': 'medium',
        'render_mode': None,
        'max_steps': 800,
        'seed': 42
    }
