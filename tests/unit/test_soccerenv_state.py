# File: tests/unit/test_soccerenv_state.py
"""
State and Observation Space Tests for Soccer RL FYP

This module validates the observation vector construction, bounds checking,
and coordinate transformations. Tests ensure the RL agent receives correct
state information for policy learning.

Student: Ali Riyaz (C3412624)
References:
- Testing plan document (Section 4.3: State Space Validation)
- OpenAI Gym specification for observation spaces
- Coordinate transformation mathematics (rotation matrices)
"""

import pytest
import numpy as np
import numpy.testing as npt
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Change to project root so config file can be found
os.chdir(project_root)

from environments.soccerenv import SoccerEnv


class TestObservationSpace:
    """Test observation space structure and bounds"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))
        self.env.reset()

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def test_observation_vector_length(self):
        """
        Test observation vector has exactly 12 elements.

        Observation structure (from soccerenv.py:355-362):
        [0] norm_robot_x
        [1] norm_robot_y
        [2] norm_angle
        [3] norm_ball_x
        [4] norm_ball_y
        [5] norm_opponent_x
        [6] norm_opponent_y
        [7] norm_robot_vx
        [8] norm_robot_vy
        [9] ball_distance
        [10] goal_distance
        [11] has_ball
        """
        obs, _ = self.env.reset()

        print(f"\n✓ Observation vector structure test:")
        print(f"  Expected length: 12")
        print(f"  Actual length: {len(obs)}")
        print(f"  Data type: {obs.dtype}")
        print(f"  Observation space shape: {self.env.observation_space.shape}")

        assert len(obs) == 12, f"Observation should have 12 elements, got {len(obs)}"
        assert obs.dtype == np.float32, f"Observation dtype should be float32, got {obs.dtype}"
        assert self.env.observation_space.shape == (12,), \
            f"Observation space shape should be (12,), got {self.env.observation_space.shape}"

    def test_observation_element_semantics(self):
        """
        Test each observation element has correct semantics and valid values.
        """
        # Set known state
        self.env.robot_pos = np.array([600.0, 300.0])  # Right side of field (field_width=900)
        self.env.robot_angle = np.pi / 4  # 45 degrees
        self.env.ball_pos = np.array([650.0, 300.0])  # Slightly ahead
        self.env.opponent_pos = np.array([200.0, 300.0])  # Far left
        self.env.robot_vel = np.array([2.0, 1.0])
        self.env.has_ball = True

        obs = self.env._get_obs()

        print(f"\n✓ Observation element semantics test:")
        print(f"  Field size: {self.env.field_width}x{self.env.field_height}")
        print(f"  [0] robot_x: {obs[0]:.3f} (robot at {self.env.robot_pos[0]:.0f}px)")
        print(f"  [1] robot_y: {obs[1]:.3f} (robot at {self.env.robot_pos[1]:.0f}px)")
        print(f"  [2] robot_angle: {obs[2]:.3f} (angle {np.degrees(self.env.robot_angle):.0f}°)")
        print(f"  [3] ball_x: {obs[3]:.3f} (ball at {self.env.ball_pos[0]:.0f}px)")
        print(f"  [4] ball_y: {obs[4]:.3f} (ball at {self.env.ball_pos[1]:.0f}px)")
        print(f"  [5] opponent_x: {obs[5]:.3f} (expected: 0.0 - disabled)")
        print(f"  [6] opponent_y: {obs[6]:.3f} (expected: 0.0 - disabled)")
        print(f"  [7] robot_vx: {obs[7]:.3f} (velocity: {self.env.robot_vel[0]:.1f})")
        print(f"  [8] robot_vy: {obs[8]:.3f} (velocity: {self.env.robot_vel[1]:.1f})")
        print(f"  [9] ball_distance: {obs[9]:.3f}")
        print(f"  [10] goal_distance: {obs[10]:.3f}")
        print(f"  [11] has_ball: {obs[11]:.3f} (expected: 1.0)")

        # Check robot position is in right side of field (positive normalized x)
        assert obs[0] > 0, f"Robot at {self.env.robot_pos[0]}px should be in right half (positive x), got {obs[0]}"

        # Check robot angle is in valid range
        assert -1 <= obs[2] <= 1, "Robot angle should be normalized to [-1,1]"

        # Check ball is ahead of robot (higher normalized x)
        assert obs[3] >= obs[0], "Ball should be ahead or at robot position"

        # Check opponent is disabled (set to 0)
        assert obs[5] == 0.0, "Opponent x should be 0.0 (disabled)"
        assert obs[6] == 0.0, "Opponent y should be 0.0 (disabled)"

        # Check distances are in [0,1]
        assert 0 <= obs[9] <= 1, "Ball distance should be in [0,1]"
        assert 0 <= obs[10] <= 1, "Goal distance should be in [0,1]"

        # Check has_ball is binary
        assert obs[11] in [0.0, 1.0], "has_ball should be 0.0 or 1.0"
        assert obs[11] == 1.0, "has_ball should be 1.0 (was set to True)"

    def test_observation_bounds_over_time(self):
        """
        Test observation stays within bounds over many random steps.

        Requirement: All elements should respect their defined bounds:
        - Positions/angles/velocities: [-1, 1]
        - Distances: [0, 1]
        - has_ball: {0, 1}
        """
        self.env.reset()

        observations = []
        num_steps = 200

        for _ in range(num_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            observations.append(obs.copy())

            if terminated or truncated:
                self.env.reset()

        observations = np.array(observations)

        # Check bounds for each element
        print(f"\n✓ Observation bounds test over {num_steps} steps:")

        # Elements 0-8: Should be in [-1, 1]
        for i in range(9):
            min_val = np.min(observations[:, i])
            max_val = np.max(observations[:, i])
            print(f"  Element {i}: min={min_val:.3f}, max={max_val:.3f} (expected: [-1,1])")

            assert min_val >= -1.0, f"Element {i} below lower bound: {min_val}"
            assert max_val <= 1.0, f"Element {i} above upper bound: {max_val}"

        # Elements 9-10: Distances should be in [0, 1]
        for i in [9, 10]:
            min_val = np.min(observations[:, i])
            max_val = np.max(observations[:, i])
            print(f"  Element {i}: min={min_val:.3f}, max={max_val:.3f} (expected: [0,1])")

            assert min_val >= 0.0, f"Element {i} below 0: {min_val}"
            assert max_val <= 1.0, f"Element {i} above 1: {max_val}"

        # Element 11: has_ball should be binary
        unique_values = np.unique(observations[:, 11])
        print(f"  Element 11 (has_ball): unique values = {unique_values} (expected: {{0, 1}})")
        assert np.all(np.isin(unique_values, [0.0, 1.0])), \
            f"has_ball should only be 0 or 1, got {unique_values}"

    def test_observation_no_nan_or_inf(self):
        """
        Test observation never contains NaN or infinity values.

        Critical for RL training stability.
        """
        self.env.reset()

        observations = []
        num_steps = 100

        for _ in range(num_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Check for NaN/inf
            assert not np.any(np.isnan(obs)), f"Observation contains NaN at step {_}"
            assert not np.any(np.isinf(obs)), f"Observation contains infinity at step {_}"

            observations.append(obs)

            if terminated or truncated:
                self.env.reset()

        print(f"\n✓ NaN/Inf test: {num_steps} steps, all values finite")


class TestCoordinateTransformation:
    """
    Test coordinate transformations (world frame → robot frame).

    Note: Current implementation uses world coordinates. This test validates
    that coordinates are correctly represented and could support transformation
    to robot-centric frame if needed.
    """

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))
        self.env.reset()

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def _world_to_robot_frame(self, world_pos, robot_pos, robot_angle):
        """
        Helper: Transform world coordinates to robot-centric frame.

        Transformation:
        1. Translate by -robot_pos (move robot to origin)
        2. Rotate by -robot_angle (align robot heading with x-axis)

        Rotation matrix: R(-θ) = [[cos(-θ), -sin(-θ)],
                                   [sin(-θ),  cos(-θ)]]

        Reference: Robotics coordinate transformation (Craig, 2005)
        """
        # Translate to robot origin
        relative_pos = world_pos - robot_pos

        # Rotate to robot frame
        cos_theta = np.cos(-robot_angle)
        sin_theta = np.sin(-robot_angle)

        robot_frame_x = relative_pos[0] * cos_theta - relative_pos[1] * sin_theta
        robot_frame_y = relative_pos[0] * sin_theta + relative_pos[1] * cos_theta

        return np.array([robot_frame_x, robot_frame_y])

    def test_coordinate_frame_consistency(self):
        """
        Test that world coordinates are consistent with robot position.

        Current implementation uses world frame, so robot position in observation
        should match actual world position (when normalized).
        """
        # Set known positions
        self.env.robot_pos = np.array([600.0, 400.0])
        self.env.ball_pos = np.array([700.0, 450.0])

        obs = self.env._get_obs()

        # De-normalize robot position
        denorm_robot_x = (obs[0] + 1) * self.env.field_width / 2
        denorm_robot_y = (obs[1] + 1) * self.env.field_height / 2

        # De-normalize ball position
        denorm_ball_x = (obs[3] + 1) * self.env.field_width / 2
        denorm_ball_y = (obs[4] + 1) * self.env.field_height / 2

        print(f"\n✓ Coordinate frame consistency test:")
        print(f"  Actual robot pos: [{self.env.robot_pos[0]:.1f}, {self.env.robot_pos[1]:.1f}]")
        print(f"  Denorm robot pos: [{denorm_robot_x:.1f}, {denorm_robot_y:.1f}]")
        print(f"  Actual ball pos: [{self.env.ball_pos[0]:.1f}, {self.env.ball_pos[1]:.1f}]")
        print(f"  Denorm ball pos: [{denorm_ball_x:.1f}, {denorm_ball_y:.1f}]")

        # Should match within small tolerance (due to clipping)
        npt.assert_array_almost_equal(
            [denorm_robot_x, denorm_robot_y],
            self.env.robot_pos,
            decimal=1,
            err_msg="De-normalized robot position should match actual position"
        )

        npt.assert_array_almost_equal(
            [denorm_ball_x, denorm_ball_y],
            self.env.ball_pos,
            decimal=1,
            err_msg="De-normalized ball position should match actual position"
        )

    @pytest.mark.parametrize("angle_deg,angle_rad,expected_norm,description", [
        (0, 0.0, 0.0, "facing right (0°)"),
        (90, np.pi/2, 0.5, "facing down (90°)"),
        (180, np.pi, 1.0, "facing left (180°)"),
        (-90, -np.pi/2, -0.5, "facing up (-90°)"),  # Use -90° instead of 270°
        (45, np.pi/4, 0.25, "facing down-right (45°)"),
    ])
    def test_robot_orientation_edge_cases(self, angle_deg, angle_rad, expected_norm, description):
        """
        Test robot angle normalization at key orientations.

        Angle convention (from soccerenv.py:436-440):
        - θ = 0: Robot faces +X (right, toward goal)
        - θ = π/2: Robot faces +Y (down)
        - θ = π: Robot faces -X (left, away from goal)
        - θ = -π/2: Robot faces -Y (up)

        Normalization: norm_angle = angle / π, clipped to [-1, 1]
        So: -π → -1, 0 → 0, π → 1
        """
        self.env.robot_pos = np.array([400.0, 300.0])
        self.env.robot_angle = angle_rad

        obs = self.env._get_obs()

        # De-normalize angle
        denorm_angle = obs[2] * np.pi

        print(f"\n✓ Robot orientation test - {description}:")
        print(f"  Set angle: {angle_deg}° ({angle_rad:.4f} rad)")
        print(f"  Expected normalized: {expected_norm:.4f}")
        print(f"  Observed normalized: {obs[2]:.4f}")
        print(f"  De-normalized: {denorm_angle:.4f} rad ({np.degrees(denorm_angle):.1f}°)")

        # Check normalized value matches expected
        npt.assert_almost_equal(
            obs[2],
            expected_norm,
            decimal=3,
            err_msg=f"Normalized angle mismatch for {description}"
        )

        # Normalized angle should be in [-1, 1]
        assert -1 <= obs[2] <= 1, f"Normalized angle out of bounds for {description}"

    def test_relative_position_calculation(self):
        """
        Test relative position calculations (ball distance, goal distance).

        Validates that distances are correctly computed and normalized.
        """
        # Place robot and ball at known positions
        self.env.robot_pos = np.array([300.0, 300.0])
        self.env.ball_pos = np.array([400.0, 400.0])  # 100√2 ≈ 141.4 pixels away
        self.env.goal_pos = self.env._goal_center  # Goal center

        obs = self.env._get_obs()

        # Calculate expected distances
        expected_ball_dist = np.linalg.norm(self.env.ball_pos - self.env.robot_pos)
        expected_goal_dist = np.linalg.norm(self.env.goal_pos - self.env.robot_pos)

        max_dist = self.env._max_distance

        expected_ball_norm = expected_ball_dist / max_dist
        expected_goal_norm = expected_goal_dist / max_dist

        observed_ball_norm = obs[9]
        observed_goal_norm = obs[10]

        print(f"\n✓ Relative position calculation test:")
        print(f"  Max distance: {max_dist:.1f} pixels")
        print(f"  Ball distance: {expected_ball_dist:.1f} pixels → normalized: {expected_ball_norm:.3f}")
        print(f"  Observed ball distance: {observed_ball_norm:.3f}")
        print(f"  Goal distance: {expected_goal_dist:.1f} pixels → normalized: {expected_goal_norm:.3f}")
        print(f"  Observed goal distance: {observed_goal_norm:.3f}")

        # Should match within tolerance (clipping may affect exact values)
        npt.assert_almost_equal(
            observed_ball_norm,
            min(expected_ball_norm, 1.0),  # Clipped to [0,1]
            decimal=2,
            err_msg="Ball distance calculation mismatch"
        )

        npt.assert_almost_equal(
            observed_goal_norm,
            min(expected_goal_norm, 1.0),  # Clipped to [0,1]
            decimal=2,
            err_msg="Goal distance calculation mismatch"
        )

    def test_velocity_normalization(self):
        """
        Test velocity normalization is correct.

        From soccerenv.py:340: max_vel = self.robot_speed * 2.0
        Velocities normalized by max_vel and clipped to [-1, 1]
        """
        self.env.robot_pos = np.array([400.0, 300.0])
        self.env.robot_vel = np.array([3.0, 2.0])  # Known velocity

        obs = self.env._get_obs()

        max_vel = self.env.robot_speed * 2.0
        expected_vx = np.clip(self.env.robot_vel[0] / max_vel, -1, 1)
        expected_vy = np.clip(self.env.robot_vel[1] / max_vel, -1, 1)

        observed_vx = obs[7]
        observed_vy = obs[8]

        print(f"\n✓ Velocity normalization test:")
        print(f"  Robot speed: {self.env.robot_speed:.2f} pixels/frame")
        print(f"  Max velocity: {max_vel:.2f} pixels/frame")
        print(f"  Actual velocity: [{self.env.robot_vel[0]:.2f}, {self.env.robot_vel[1]:.2f}]")
        print(f"  Expected normalized: [{expected_vx:.3f}, {expected_vy:.3f}]")
        print(f"  Observed normalized: [{observed_vx:.3f}, {observed_vy:.3f}]")

        npt.assert_almost_equal(
            observed_vx,
            expected_vx,
            decimal=3,
            err_msg="Velocity X normalization mismatch"
        )

        npt.assert_almost_equal(
            observed_vy,
            expected_vy,
            decimal=3,
            err_msg="Velocity Y normalization mismatch"
        )


class TestStateTransitionsConsistency:
    """Test state transitions are consistent and deterministic"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def test_state_determinism(self):
        """
        Test same initial state and actions produce same observations.

        Critical for reproducible RL experiments.
        """
        # First run
        np.random.seed(42)
        self.env.reset(seed=42)

        initial_state = {
            'robot_pos': np.array([400.0, 300.0]),
            'ball_pos': np.array([450.0, 300.0]),
            'robot_angle': 0.0,
            'robot_vel': np.array([0.0, 0.0]),
        }

        self.env.robot_pos = initial_state['robot_pos'].copy()
        self.env.ball_pos = initial_state['ball_pos'].copy()
        self.env.robot_angle = initial_state['robot_angle']
        self.env.robot_vel = initial_state['robot_vel'].copy()

        observations1 = []
        actions = [[0.5, 0.2, 0.0] for _ in range(10)]

        for action in actions:
            obs, reward, terminated, truncated, info = self.env.step(action)
            observations1.append(obs.copy())
            if terminated or truncated:
                break

        # Second run with same conditions
        np.random.seed(42)
        self.env.reset(seed=42)

        self.env.robot_pos = initial_state['robot_pos'].copy()
        self.env.ball_pos = initial_state['ball_pos'].copy()
        self.env.robot_angle = initial_state['robot_angle']
        self.env.robot_vel = initial_state['robot_vel'].copy()

        observations2 = []

        for action in actions:
            obs, reward, terminated, truncated, info = self.env.step(action)
            observations2.append(obs.copy())
            if terminated or truncated:
                break

        # Compare observations
        assert len(observations1) == len(observations2), \
            "Different number of observations in repeated runs"

        max_diff = 0.0
        for i, (obs1, obs2) in enumerate(zip(observations1, observations2)):
            diff = np.max(np.abs(obs1 - obs2))
            max_diff = max(max_diff, diff)

            # Should be identical or very close
            npt.assert_array_almost_equal(
                obs1,
                obs2,
                decimal=4,
                err_msg=f"Observations differ at step {i}"
            )

        print(f"\n✓ State determinism test:")
        print(f"  Steps tested: {len(observations1)}")
        print(f"  Maximum observation difference: {max_diff:.8f}")
        print(f"  Deterministic: {'Yes' if max_diff < 0.01 else 'No (check random perturbations)'}")

    def test_observation_changes_with_action(self):
        """
        Test that taking actions changes the observation.

        Ensures environment is responsive to actions.
        """
        obs1, _ = self.env.reset()

        # Take action
        action = [1.0, 0.0, 0.0]  # Forward
        obs2, _, _, _, _ = self.env.step(action)

        # Observations should differ (robot moved)
        obs_diff = np.linalg.norm(obs1 - obs2)

        print(f"\n✓ Observation responsiveness test:")
        print(f"  Initial observation: {obs1[:3]}")
        print(f"  After action: {obs2[:3]}")
        print(f"  Observation difference (L2 norm): {obs_diff:.4f}")

        assert obs_diff > 0.001, \
            "Observation should change after taking action"


class TestObservationSpaceCompatibility:
    """Test observation space compatibility with RL libraries"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def test_observation_space_contains_sample(self):
        """
        Test that generated observations are within declared observation space.

        Required for Gym compatibility.
        """
        obs, _ = self.env.reset()

        # Check if observation is in observation space
        in_space = self.env.observation_space.contains(obs)

        print(f"\n✓ Observation space compatibility test:")
        print(f"  Observation space: Box({self.env.observation_space.low[0]}, " +
              f"{self.env.observation_space.high[0]}, shape={self.env.observation_space.shape})")
        print(f"  Sample observation shape: {obs.shape}")
        print(f"  Sample observation dtype: {obs.dtype}")
        print(f"  In observation space: {in_space}")

        assert in_space, "Generated observation should be within declared observation space"

    def test_observation_space_sample(self):
        """
        Test that observation space can generate samples.

        Though not used directly, this validates the space definition.
        """
        sample = self.env.observation_space.sample()

        print(f"\n✓ Observation space sampling test:")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Sample dtype: {sample.dtype}")
        print(f"  Sample range: [{np.min(sample):.2f}, {np.max(sample):.2f}]")

        assert sample.shape == (12,), "Sample should have 12 elements"
        assert sample.dtype == np.float32, "Sample should be float32"


if __name__ == "__main__":
    """Run tests with verbose output"""
    pytest.main([__file__, "-v", "-s"])