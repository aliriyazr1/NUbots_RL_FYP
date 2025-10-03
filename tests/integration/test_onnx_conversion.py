# File: tests/integration/test_onnx_conversion.py
"""
ONNX Conversion and Inference Tests for Soccer RL FYP

This module validates ONNX model export and inference performance per
integration requirements FR-10. Tests ensure deployment-ready models.

Student: Ali Riyaz (C3412624)
References:
- Testing plan document (Section 5.2: Integration Testing)
- ONNX specification: https://onnx.ai/
- FR-10: Model export and inference requirements
"""

import pytest
import numpy as np
import numpy.testing as npt
import sys
import os
import time
from pathlib import Path
import tempfile

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Change to project root
os.chdir(project_root)

# Import after path setup
DEPENDENCIES_AVAILABLE = True
IMPORT_ERROR = None

try:
    import torch
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"PyTorch not available: {str(e)}"

try:
    import onnx
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"ONNX not available: {str(e)}"

try:
    import onnxruntime as ort
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = f"ONNX Runtime not available: {str(e)}"

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

# Print diagnostic if dependencies unavailable
if not DEPENDENCIES_AVAILABLE:
    print(f"\n⚠ Test dependencies unavailable: {IMPORT_ERROR}")
    print("Tests will be skipped.")


class DeterministicPPOWrapper(torch.nn.Module):
    """
    Wrapper to extract deterministic actor network for ONNX export.

    Stable Baselines3 PPO is stochastic - it samples actions from a distribution.
    For deployment, we need deterministic behaviour (same input → same output).
    This wrapper extracts only the mean action from the policy network.

    Source: Adapted from src/utils/onnx_converter.py
    """
    def __init__(self, policy, action_space):
        super().__init__()
        self.policy = policy
        self.action_space = action_space

    def forward(self, obs):
        """
        Forward pass for deterministic action prediction.

        Implementation follows PPO policy architecture:
        1. Extract features from observation
        2. Pass through policy MLP
        3. Get mean action (skip stochastic sampling)
        4. Apply action space bounds

        Args:
            obs: Observation tensor [batch_size, 12]

        Returns:
            action: Deterministic action tensor [batch_size, 3]
        """
        # Extract features from observation
        features = self.policy.extract_features(obs)
        # Get policy network output
        policy_features = self.policy.mlp_extractor.policy_net(features)
        # Get mean action (deterministic, no sampling)
        mean_actions = self.policy.action_net(policy_features)

        # Apply action space clipping
        low = torch.tensor(self.action_space.low, dtype=mean_actions.dtype, device=mean_actions.device)
        high = torch.tensor(self.action_space.high, dtype=mean_actions.dtype, device=mean_actions.device)
        mean_actions = torch.clamp(mean_actions, low, high)

        return mean_actions


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="ONNX/PyTorch dependencies not available")
class TestONNXConversion:
    """
    Test ONNX model conversion from PyTorch.

    Requirement: FR-10 - Model must export to ONNX format
    """

    @classmethod
    def setup_class(cls):
        """Setup - load trained model once for all tests"""
        cls.model_path = project_root / "models" / "ppo" / "soccer_rl_ppo_final.zip"

        if not cls.model_path.exists():
            pytest.skip(f"Model not found: {cls.model_path}")

        print(f"\n✓ Loading model from: {cls.model_path}")
        cls.model = PPO.load(cls.model_path, device="cpu")
        cls.policy = cls.model.policy
        cls.policy.to("cpu")  # Ensure policy is on CPU for ONNX export

        # Create environment to get action space for wrapper
        cls.env = SoccerEnv(render_mode=None, difficulty="easy")

        # Create deterministic wrapper for ONNX export
        cls.policy_wrapper = DeterministicPPOWrapper(cls.policy, cls.env.action_space)
        cls.policy_wrapper.eval()
        cls.policy_wrapper.to("cpu")  # Ensure wrapper is on CPU

        # Get model architecture info
        cls.input_dim = 12  # Observation space
        cls.output_dim = 3  # Action space

    @classmethod
    def teardown_class(cls):
        """Cleanup - close environment"""
        if hasattr(cls, 'env'):
            cls.env.close()

    def test_model_loads_successfully(self):
        """Test that trained model loads without errors"""
        print(f"\n✓ Model load test:")
        print(f"  Model type: {type(self.model)}")
        print(f"  Policy type: {type(self.policy)}")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Output dimension: {self.output_dim}")

        assert self.model is not None, "Model should load successfully"
        assert self.policy is not None, "Policy should be accessible"

    def test_onnx_export(self):
        """
        Test PyTorch model exports to ONNX format.

        Creates temporary ONNX file and validates structure.
        """
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            onnx_path = tmp_file.name

        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)

            # Export to ONNX (using deterministic wrapper)
            torch.onnx.export(
                self.policy_wrapper,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['observation'],
                output_names=['action'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                }
            )
            #TODO: Claude code the below stuff
            #TODO: Check why the tests are being skipped got some reason

            print(f"\n✓ ONNX export test:")
            print(f"  Export path: {onnx_path}")
            print(f"  File exists: {os.path.exists(onnx_path)}")
            print(f"  File size: {os.path.getsize(onnx_path) / 1024:.2f} KB")

            assert os.path.exists(onnx_path), "ONNX file should be created"
            assert os.path.getsize(onnx_path) > 0, "ONNX file should not be empty"

            # Validate ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            print(f"  ONNX validation: Passed")

        finally:
            # Cleanup
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_onnx_tensor_shapes(self):
        """
        Test ONNX model has correct input/output tensor shapes.

        Requirement: Input [1, 12], Output [1, 3]
        """
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            onnx_path = tmp_file.name

        try:
            # Export model (using deterministic wrapper)
            dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)
            torch.onnx.export(
                self.policy_wrapper,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                input_names=['observation'],
                output_names=['action']
            )

            # Load ONNX model and check shapes
            ort_session = ort.InferenceSession(onnx_path)

            # Check input shape
            input_info = ort_session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape

            # Check output shape
            output_info = ort_session.get_outputs()[0]
            output_name = output_info.name
            output_shape = output_info.shape

            print(f"\n✓ ONNX tensor shape test:")
            print(f"  Input name: {input_name}")
            print(f"  Input shape: {input_shape} (expected: [batch, 12])")
            print(f"  Output name: {output_name}")
            print(f"  Output shape: {output_shape} (expected: [batch, 3])")

            # Validate shapes (allowing dynamic batch size)
            assert input_shape[1] == self.input_dim, \
                f"Input dimension should be {self.input_dim}, got {input_shape[1]}"
            assert output_shape[1] == self.output_dim, \
                f"Output dimension should be {self.output_dim}, got {output_shape[1]}"

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="ONNX/PyTorch dependencies not available")
class TestONNXInferenceAccuracy:
    """
    Test ONNX inference produces same results as PyTorch.

    Requirement: FR-10 - Output within 10^-4 tolerance
    """

    @classmethod
    def setup_class(cls):
        """Setup - load model and export to ONNX"""
        cls.model_path = project_root / "models" / "ppo" / "soccer_rl_ppo_final.zip"

        if not cls.model_path.exists():
            pytest.skip(f"Model not found: {cls.model_path}")

        cls.model = PPO.load(cls.model_path, device="cpu")
        cls.policy = cls.model.policy
        cls.policy.eval()  # Set to evaluation mode
        cls.policy.to("cpu")  # Ensure policy is on CPU

        # Create environment to get action space
        cls.env = SoccerEnv(render_mode=None, difficulty="easy")

        # Create deterministic wrapper
        cls.policy_wrapper = DeterministicPPOWrapper(cls.policy, cls.env.action_space)
        cls.policy_wrapper.eval()
        cls.policy_wrapper.to("cpu")  # Ensure wrapper is on CPU

        cls.input_dim = 12
        cls.output_dim = 3

        # Export to ONNX (persistent for class)
        cls.onnx_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        cls.onnx_path = cls.onnx_file.name

        dummy_input = torch.randn(1, cls.input_dim, dtype=torch.float32)
        torch.onnx.export(
            cls.policy_wrapper,
            dummy_input,
            cls.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action']
        )

        # Create ONNX runtime session
        cls.ort_session = ort.InferenceSession(cls.onnx_path)

    @classmethod
    def teardown_class(cls):
        """Cleanup ONNX file and environment"""
        if hasattr(cls, 'onnx_path') and os.path.exists(cls.onnx_path):
            os.unlink(cls.onnx_path)
        if hasattr(cls, 'env'):
            cls.env.close()

    def test_pytorch_vs_onnx_single_input(self):
        """
        Test single input produces same output in PyTorch and ONNX.

        Tolerance: max difference < 10^-4
        """
        # Create test input
        test_obs = np.random.randn(1, self.input_dim).astype(np.float32)

        # PyTorch inference (using deterministic wrapper)
        with torch.no_grad():
            torch_input = torch.from_numpy(test_obs)
            torch_output = self.policy_wrapper(torch_input)
            pytorch_result = torch_output.numpy()

        # ONNX inference
        onnx_result = self.ort_session.run(
            None,
            {'observation': test_obs}
        )[0]

        # Calculate difference
        abs_diff = np.abs(pytorch_result - onnx_result)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        print(f"\n✓ Single input inference test:")
        print(f"  PyTorch output: {pytorch_result[0]}")
        print(f"  ONNX output:    {onnx_result[0]}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Tolerance: 1e-4")

        # Requirement: max difference < 10^-4
        assert max_diff < 1e-4, \
            f"Max difference {max_diff:.2e} exceeds tolerance 1e-4"

    def test_pytorch_vs_onnx_multiple_inputs(self):
        """
        Test 100 random observations for PyTorch vs ONNX accuracy.

        Requirement: FR-10 - Test with 100 random observations
        """
        num_tests = 100
        max_differences = []
        mean_differences = []

        for i in range(num_tests):
            # Random observation
            test_obs = np.random.randn(1, self.input_dim).astype(np.float32)

            # PyTorch inference (using deterministic wrapper)
            with torch.no_grad():
                torch_input = torch.from_numpy(test_obs)
                torch_output = self.policy_wrapper(torch_input)
                pytorch_result = torch_output.numpy()

            # ONNX inference
            onnx_result = self.ort_session.run(
                None,
                {'observation': test_obs}
            )[0]

            # Calculate differences
            abs_diff = np.abs(pytorch_result - onnx_result)
            max_differences.append(np.max(abs_diff))
            mean_differences.append(np.mean(abs_diff))

        # Statistics
        overall_max = np.max(max_differences)
        overall_mean = np.mean(mean_differences)
        percentile_95 = np.percentile(max_differences, 95)
        percentile_99 = np.percentile(max_differences, 99)

        print(f"\n✓ Multiple inputs test (n={num_tests}):")
        print(f"  Overall max difference: {overall_max:.2e}")
        print(f"  Overall mean difference: {overall_mean:.2e}")
        print(f"  95th percentile: {percentile_95:.2e}")
        print(f"  99th percentile: {percentile_99:.2e}")
        print(f"  Tolerance: 1e-4")

        # All tests should pass tolerance
        failures = sum(1 for d in max_differences if d >= 1e-4)
        print(f"  Tests within tolerance: {num_tests - failures}/{num_tests}")

        assert overall_max < 1e-4, \
            f"Max difference {overall_max:.2e} exceeds tolerance 1e-4"


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="ONNX/PyTorch dependencies not available")
class TestONNXDeterminism:
    """
    Test ONNX inference is deterministic.

    Requirement: Same input → identical output (bit-exact)
    """

    @classmethod
    def setup_class(cls):
        """Setup - load model and export to ONNX"""
        cls.model_path = project_root / "models" / "ppo" / "soccer_rl_ppo_final.zip"

        if not cls.model_path.exists():
            pytest.skip(f"Model not found: {cls.model_path}")

        cls.model = PPO.load(cls.model_path, device="cpu")
        cls.policy = cls.model.policy
        cls.policy.eval()
        cls.policy.to("cpu")  # Ensure policy is on CPU

        # Create environment to get action space
        cls.env = SoccerEnv(render_mode=None, difficulty="easy")

        # Create deterministic wrapper
        cls.policy_wrapper = DeterministicPPOWrapper(cls.policy, cls.env.action_space)
        cls.policy_wrapper.eval()
        cls.policy_wrapper.to("cpu")  # Ensure wrapper is on CPU

        cls.input_dim = 12

        # Export to ONNX
        cls.onnx_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        cls.onnx_path = cls.onnx_file.name

        dummy_input = torch.randn(1, cls.input_dim, dtype=torch.float32)
        torch.onnx.export(
            cls.policy_wrapper,
            dummy_input,
            cls.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action']
        )

        cls.ort_session = ort.InferenceSession(cls.onnx_path)

    @classmethod
    def teardown_class(cls):
        """Cleanup"""
        if hasattr(cls, 'onnx_path') and os.path.exists(cls.onnx_path):
            os.unlink(cls.onnx_path)
        if hasattr(cls, 'env'):
            cls.env.close()

    def test_deterministic_inference(self):
        """
        Test same input produces identical output across multiple inferences.

        Critical for deployment reproducibility.
        """
        # Fixed input
        test_obs = np.random.randn(1, self.input_dim).astype(np.float32)

        # Run inference 10 times
        results = []
        for i in range(10):
            result = self.ort_session.run(
                None,
                {'observation': test_obs}
            )[0]
            results.append(result.copy())

        # Check all results are identical
        max_diff_across_runs = 0.0
        for i in range(1, len(results)):
            diff = np.max(np.abs(results[0] - results[i]))
            max_diff_across_runs = max(max_diff_across_runs, diff)

        print(f"\n✓ Determinism test:")
        print(f"  Number of runs: {len(results)}")
        print(f"  Max difference across runs: {max_diff_across_runs:.2e}")
        print(f"  First output: {results[0][0]}")
        print(f"  Last output:  {results[-1][0]}")

        # Should be bit-exact (0.0 difference)
        assert max_diff_across_runs == 0.0, \
            f"ONNX inference should be deterministic, got difference {max_diff_across_runs}"


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="ONNX/PyTorch dependencies not available")
class TestONNXInferenceLatency:
    """
    Test ONNX inference latency meets performance requirements.

    Requirement: FR-10 - Inference <20ms (95th percentile over 1000 runs)
    """

    @classmethod
    def setup_class(cls):
        """Setup - load model and export to ONNX"""
        cls.model_path = project_root / "models" / "ppo" / "soccer_rl_ppo_final.zip"

        if not cls.model_path.exists():
            pytest.skip(f"Model not found: {cls.model_path}")

        cls.model = PPO.load(cls.model_path, device="cpu")
        cls.policy = cls.model.policy
        cls.policy.eval()
        cls.policy.to("cpu")  # Ensure policy is on CPU

        # Create environment to get action space
        cls.env = SoccerEnv(render_mode=None, difficulty="easy")

        # Create deterministic wrapper
        cls.policy_wrapper = DeterministicPPOWrapper(cls.policy, cls.env.action_space)
        cls.policy_wrapper.eval()
        cls.policy_wrapper.to("cpu")  # Ensure wrapper is on CPU

        cls.input_dim = 12

        # Export to ONNX
        cls.onnx_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        cls.onnx_path = cls.onnx_file.name

        dummy_input = torch.randn(1, cls.input_dim, dtype=torch.float32)
        torch.onnx.export(
            cls.policy_wrapper,
            dummy_input,
            cls.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action']
        )

        cls.ort_session = ort.InferenceSession(cls.onnx_path)

    @classmethod
    def teardown_class(cls):
        """Cleanup"""
        if hasattr(cls, 'onnx_path') and os.path.exists(cls.onnx_path):
            os.unlink(cls.onnx_path)
        if hasattr(cls, 'env'):
            cls.env.close()

    def test_inference_latency(self):
        """
        Test ONNX inference latency over 1000 runs.

        Requirement: 95th percentile < 20ms
        """
        num_runs = 1000
        latencies = []

        # Generate test observations
        test_observations = [
            np.random.randn(1, self.input_dim).astype(np.float32)
            for _ in range(num_runs)
        ]

        # Warmup runs (JIT compilation, caching)
        for _ in range(10):
            self.ort_session.run(None, {'observation': test_observations[0]})

        # Measure latency
        for obs in test_observations:
            start_time = time.perf_counter()
            self.ort_session.run(None, {'observation': obs})
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency_ms)

        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)

        print(f"\n✓ Inference latency test (n={num_runs}):")
        print(f"  Mean:         {mean_latency:.3f} ms")
        print(f"  Median:       {median_latency:.3f} ms")
        print(f"  Min:          {min_latency:.3f} ms")
        print(f"  Max:          {max_latency:.3f} ms")
        print(f"  95th percentile: {p95_latency:.3f} ms")
        print(f"  99th percentile: {p99_latency:.3f} ms")
        print(f"  Requirement:  < 20.0 ms (95th percentile)")

        # Requirement: 95th percentile < 20ms
        assert p95_latency < 20.0, \
            f"95th percentile latency {p95_latency:.3f}ms exceeds requirement 20ms"

    def test_pytorch_vs_onnx_latency(self):
        """
        Compare PyTorch vs ONNX inference speed.

        ONNX should be faster or comparable for deployment.
        """
        num_runs = 100
        test_observations = [
            np.random.randn(1, self.input_dim).astype(np.float32)
            for _ in range(num_runs)
        ]

        # PyTorch latency (using deterministic wrapper)
        pytorch_latencies = []
        for obs in test_observations:
            torch_obs = torch.from_numpy(obs)
            start_time = time.perf_counter()
            with torch.no_grad():
                self.policy_wrapper(torch_obs)
            end_time = time.perf_counter()
            pytorch_latencies.append((end_time - start_time) * 1000)

        # ONNX latency
        onnx_latencies = []
        for obs in test_observations:
            start_time = time.perf_counter()
            self.ort_session.run(None, {'observation': obs})
            end_time = time.perf_counter()
            onnx_latencies.append((end_time - start_time) * 1000)

        pytorch_mean = np.mean(pytorch_latencies)
        onnx_mean = np.mean(onnx_latencies)
        speedup = pytorch_mean / onnx_mean

        print(f"\n✓ PyTorch vs ONNX latency comparison (n={num_runs}):")
        print(f"  PyTorch mean: {pytorch_mean:.3f} ms")
        print(f"  ONNX mean:    {onnx_mean:.3f} ms")
        print(f"  Speedup:      {speedup:.2f}x")

        # ONNX should be reasonably fast (not slower than PyTorch by >2x)
        # Note: On CPU, ONNX and PyTorch may be comparable
        assert onnx_mean < pytorch_mean * 2.0, \
            f"ONNX significantly slower than PyTorch: {onnx_mean:.2f}ms vs {pytorch_mean:.2f}ms"


if __name__ == "__main__":
    """Run tests with verbose output"""
    pytest.main([__file__, "-v", "-s"])