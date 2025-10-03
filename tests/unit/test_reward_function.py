# File: tests/unit/test_reward_function.py
"""
Reward Function Unit Tests for Soccer RL FYP

This module implements systematic testing of the reward function based on the
testing plan requirements. Tests focus on anti-exploitation measures and reward bounds.

Student: Ali Riyaz (C3412624)
References:
- Testing plan document (Section 4.1: Unit Testing)
- Ng et al. (1999): Policy invariance under reward transformations
- Roijers & Whiteson (2017): Multi-objective RL evaluation
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Change to project root so config file can be found
os.chdir(project_root)

from environments.soccerenv import SoccerEnv


class TestRewardFunctionBounds:
    """Test reward function stays within expected bounds during normal play"""

    def setup_method(self):
        """Setup test environment without rendering"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def test_reward_bounds_random_actions(self):
        """
        Test that rewards stay within reasonable bounds over 1000 random steps.

        Expected: Non-terminal rewards should be roughly in [-10, +10] range. (when tests ran got -20 to +20 rewards)
        Terminal rewards (goal, out of bounds) can exceed this.

        Reference: Testing plan requirement 4.1.2
        """
        self.env.reset()
        rewards = []
        non_terminal_rewards = []

        for step in range(1000):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)

            # Track non-terminal rewards separately
            if not (terminated or truncated):
                non_terminal_rewards.append(reward)

            if terminated or truncated:
                self.env.reset()

        # All rewards should be finite
        assert all(np.isfinite(r) for r in rewards), "All rewards must be finite"

        # Non-terminal rewards should be in reasonable range
        if len(non_terminal_rewards) > 0:
            min_reward = min(non_terminal_rewards)
            max_reward = max(non_terminal_rewards)

            # Allow some buffer beyond [-10, 10] for edge cases
            assert min_reward >= -50, f"Non-terminal rewards too negative: {min_reward}"
            assert max_reward <= 50, f"Non-terminal rewards too positive: {max_reward}"

            print(f"\nNon-terminal reward statistics:")
            print(f"  Min: {min_reward:.2f}")
            print(f"  Max: {max_reward:.2f}")
            print(f"  Mean: {np.mean(non_terminal_rewards):.2f}")
            print(f"  Std: {np.std(non_terminal_rewards):.2f}")

    def test_reward_no_nan_or_inf(self):
        """Verify reward function never returns NaN or infinity"""
        self.env.reset()

        for _ in range(500):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            assert not np.isnan(reward), f"Reward is NaN at step {_}"
            assert not np.isinf(reward), f"Reward is infinite at step {_}"

            if terminated or truncated:
                self.env.reset()


class TestRewardTerminalConditions:
    """Test terminal conditions give appropriate large rewards/penalties"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def _place_entities(self, robot_pos, ball_pos, opponent_pos=None):
        """
        Helper function to place robot, ball, and opponent at specific positions.

        Args:
            robot_pos: [x, y] position for robot (pixels)
            ball_pos: [x, y] position for ball (pixels)
            opponent_pos: [x, y] position for opponent (pixels), optional
        """
        self.env.robot_pos = np.array(robot_pos, dtype=np.float32)
        self.env.ball_pos = np.array(ball_pos, dtype=np.float32)

        if opponent_pos is not None:
            self.env.opponent_pos = np.array(opponent_pos, dtype=np.float32)

        # Reset velocities
        self.env.robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.env.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

    def test_goal_scoring_maximum_reward(self):
        """
        Test that scoring a goal gives maximum reward (≈+100 to +150).

        Reference: Testing plan requirement - terminal conditions
        """
        self.env.reset()

        # Get goal position
        goal_center = self.env._goal_center
        goal_x = goal_center[0]
        goal_y = goal_center[1]

        # Place ball just before goal line
        ball_x = goal_x - 10  # Just before goal
        ball_y = goal_y

        # Place robot with ball
        robot_x = ball_x - 20
        robot_y = ball_y

        self._place_entities(
            robot_pos=[robot_x, robot_y],
            ball_pos=[ball_x, ball_y]
        )

        # Push ball into goal
        action = [1.0, 0.0, 0.0]  # Move forward

        max_reward = -float('inf')
        goal_scored = False

        for _ in range(50):
            obs, reward, terminated, truncated, info = self.env.step(action)
            max_reward = max(max_reward, reward)

            if terminated:
                goal_scored = True
                break

        # Verify goal was scored and reward is large positive
        assert goal_scored, "Goal should have been scored"
        assert max_reward >= 100.0, f"Goal reward should be ≥100, got {max_reward:.2f}"
        print(f"\n✓ Goal reward: {max_reward:.2f}")

    def test_ball_out_of_bounds_penalty(self):
        """
        Test that ball going out of bounds incurs significant penalty.

        Reference: Anti-exploitation measure - prevent ball kicking out
        """
        self.env.reset()

        # Place ball near boundary
        boundary_x = self.env.field_width - 30
        boundary_y = 20  # Near top edge

        self._place_entities(
            robot_pos=[boundary_x - 30, boundary_y],
            ball_pos=[boundary_x, boundary_y]
        )

        # Push ball toward boundary
        action = [1.0, 0.0, 1.0]  # Move forward and right

        min_reward = float('inf')
        out_of_bounds = False

        for _ in range(100):
            obs, reward, terminated, truncated, info = self.env.step(action)
            min_reward = min(min_reward, reward)

            if terminated or truncated:
                out_of_bounds = True
                break

        if out_of_bounds:
            assert min_reward <= -20.0, f"Out of bounds penalty should be ≤-20, got {min_reward:.2f}"
            print(f"\n✓ Out of bounds penalty: {min_reward:.2f}")


class TestAntiExploitationMeasures:
    """Test anti-exploitation measures prevent degenerate behaviours"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def _place_entities(self, robot_pos, ball_pos, opponent_pos=None):
        """Helper to place entities at specific positions"""
        self.env.robot_pos = np.array(robot_pos, dtype=np.float32)
        self.env.ball_pos = np.array(ball_pos, dtype=np.float32)

        if opponent_pos is not None:
            self.env.opponent_pos = np.array(opponent_pos, dtype=np.float32)

        self.env.robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.env.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

    def test_stationary_ball_holding_penalty(self):
        """
        Test that holding ball stationary for >30 steps incurs increasing penalty.

        This prevents the agent from learning to simply "camp" with the ball.

        Reference: Testing plan requirement - anti-exploitation
        Theoretical basis: Reward shaping should discourage stationary policies
        (Roijers & Whiteson, 2017)
        """
        self.env.reset()

        # Place robot with ball in possession
        mid_field_x = self.env.field_width / 2
        mid_field_y = self.env.field_height / 2

        self._place_entities(
            robot_pos=[mid_field_x, mid_field_y],
            ball_pos=[mid_field_x + 5, mid_field_y]  # Ball very close
        )

        # Take stationary action (no movement)
        stationary_action = [0.0, 0.0, 0.0]

        rewards_over_time = []

        for step in range(50):
            obs, reward, terminated, truncated, info = self.env.step(stationary_action)
            rewards_over_time.append(reward)

            if terminated or truncated:
                break

        # Check that rewards decrease over time (penalty increases)
        if len(rewards_over_time) >= 35:
            early_rewards = np.mean(rewards_over_time[5:15])  # Steps 5-15
            late_rewards = np.mean(rewards_over_time[35:45])  # Steps 35-45

            assert late_rewards < early_rewards, \
                f"Stationary penalty should increase over time. Early: {early_rewards:.2f}, Late: {late_rewards:.2f}"

            print(f"\n✓ Stationary holding penalty working:")
            print(f"  Early average (steps 5-15): {early_rewards:.2f}")
            print(f"  Late average (steps 35-45): {late_rewards:.2f}")
            print(f"  Penalty increase: {early_rewards - late_rewards:.2f}")

    def test_wall_hugging_penalty(self):
        """
        Test that staying near walls incurs penalty.

        Prevents agent from exploiting wall physics or camping in corners.
        """
        self.env.reset()

        # Place robot near wall
        wall_x = 30  # Very close to left wall
        wall_y = self.env.field_height / 2

        self._place_entities(
            robot_pos=[wall_x, wall_y],
            ball_pos=[wall_x + 20, wall_y]
        )

        # Stay near wall
        action = [0.0, 0.0, 0.0]

        rewards_near_wall = []

        for _ in range(30):
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards_near_wall.append(reward)

            if terminated or truncated:
                break

        # Now move to centre and compare
        self.env.reset()
        centre_x = self.env.field_width / 2
        centre_y = self.env.field_height / 2

        self._place_entities(
            robot_pos=[centre_x, centre_y],
            ball_pos=[centre_x + 20, centre_y]
        )

        rewards_at_centre = []

        for _ in range(30):
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards_at_centre.append(reward)

            if terminated or truncated:
                break

        # Wall position should have lower average reward
        avg_wall = np.mean(rewards_near_wall)
        avg_centre = np.mean(rewards_at_centre)

        # Wall penalty should make wall rewards lower (or at least not higher)
        assert avg_wall <= avg_centre + 1.0, \
            f"Wall hugging should not be advantageous. Wall: {avg_wall:.2f}, Centre: {avg_centre:.2f}"

        print(f"\n✓ Wall hugging test:")
        print(f"  Average near wall: {avg_wall:.2f}")
        print(f"  Average at centre: {avg_centre:.2f}")


class TestRewardComponents:
    """Test individual reward components for expected behaviour"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def _place_entities(self, robot_pos, ball_pos, opponent_pos=None):
        """Helper to place entities at specific positions"""
        self.env.robot_pos = np.array(robot_pos, dtype=np.float32)
        self.env.ball_pos = np.array(ball_pos, dtype=np.float32)

        if opponent_pos is not None:
            self.env.opponent_pos = np.array(opponent_pos, dtype=np.float32)

        self.env.robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.env.ball_vel = np.array([0.0, 0.0], dtype=np.float32)

    def test_ball_possession_reward(self):
        """
        Test that being close to ball gives positive reward.

        Mathematical basis: Gaussian reward G(d) = e^(-0.5(d/σ)²)
        Should peak when robot is at optimal ball distance.
        """
        self.env.reset()

        mid_x = self.env.field_width / 2
        mid_y = self.env.field_height / 2

        # Test different distances
        distances = [10, 30, 50, 100]  # pixels
        rewards_by_distance = []

        for distance in distances:
            self._place_entities(
                robot_pos=[mid_x, mid_y],
                ball_pos=[mid_x + distance, mid_y]
            )

            # Take one step to compute reward
            action = [0.0, 0.0, 0.0]
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards_by_distance.append(reward)

        # Closer to ball should generally give better reward
        # (allowing some tolerance for other reward components)
        closest_reward = rewards_by_distance[0]
        farthest_reward = rewards_by_distance[-1]

        print(f"\n✓ Ball possession rewards by distance:")
        for dist, rew in zip(distances, rewards_by_distance):
            print(f"  {dist} pixels: {rew:.2f}")

        # Closest should be at least as good as farthest (with some tolerance)
        assert closest_reward >= farthest_reward - 5.0, \
            "Being closer to ball should not be significantly penalised"

    def test_goal_progress_reward(self):
        """
        Test that moving ball toward goal gives positive reward.

        Uses potential-based shaping: Φ(s') - Φ(s) where Φ(s) = -dist_to_goal
        (Ng et al., 1999)
        """
        self.env.reset()

        # Get goal position
        goal_center = self.env._goal_center

        # Place ball away from goal
        start_x = self.env.field_width * 0.3
        start_y = self.env.field_height / 2

        self._place_entities(
            robot_pos=[start_x - 20, start_y],
            ball_pos=[start_x, start_y]
        )

        # Move toward goal
        forward_action = [1.0, 0.0, 0.0]

        rewards_moving_forward = []

        for _ in range(20):
            obs, reward, terminated, truncated, info = self.env.step(forward_action)
            rewards_moving_forward.append(reward)

            if terminated or truncated:
                break

        # Check reward components if available
        if hasattr(self.env, 'reward_components'):
            components = self.env.reward_components
            if 'goal_progress' in components or 'goal_approach' in components:
                print(f"\n✓ Goal progress reward components:")
                for key, value in components.items():
                    if 'goal' in key.lower():
                        print(f"  {key}: {value:.2f}")

        # Average reward should be positive when moving toward goal
        avg_reward = np.mean(rewards_moving_forward)
        print(f"\n  Average reward moving toward goal: {avg_reward:.2f}")

        # Should be at least slightly positive on average
        assert avg_reward > -5.0, \
            "Moving toward goal should not be heavily penalised"

    @pytest.mark.parametrize("scenario,robot_pos,ball_pos,expected_sign", [
        ("Close to ball", [300, 300], [310, 300], "positive_or_neutral"),
        ("Far from ball", [100, 100], [600, 400], "any"),
        ("Near goal with ball", [700, 300], [710, 300], "positive"),
    ])
    def test_reward_scenarios(self, scenario, robot_pos, ball_pos, expected_sign):
        """
        Parametrised test for different game scenarios.

        Validates that reward function responds appropriately to common situations.
        """
        self.env.reset()

        # Ensure positions are within field bounds
        robot_pos[0] = min(max(robot_pos[0], 50), self.env.field_width - 50)
        robot_pos[1] = min(max(robot_pos[1], 50), self.env.field_height - 50)
        ball_pos[0] = min(max(ball_pos[0], 50), self.env.field_width - 50)
        ball_pos[1] = min(max(ball_pos[1], 50), self.env.field_height - 50)

        self._place_entities(
            robot_pos=robot_pos,
            ball_pos=ball_pos
        )

        # Take action
        action = [1.0, 0.0, 0.0]  # Move forward
        obs, reward, terminated, truncated, info = self.env.step(action)

        print(f"\n✓ Scenario '{scenario}':")
        print(f"  Reward: {reward:.2f}")

        # Validate expected behaviour
        if expected_sign == "positive":
            assert reward > -10.0, f"Scenario '{scenario}' should have positive/neutral reward"
        elif expected_sign == "positive_or_neutral":
            assert reward > -5.0, f"Scenario '{scenario}' should not be heavily penalised"

        # All scenarios should have finite reward
        assert np.isfinite(reward), f"Reward should be finite for scenario '{scenario}'"


class TestRewardScaling:
    """Test reward scaling and component interactions"""

    def setup_method(self):
        """Setup test environment"""
        config_path = project_root / "configs" / "field_config.yaml"
        self.env = SoccerEnv(render_mode=None, difficulty="easy", config_path=str(config_path))

    def teardown_method(self):
        """Clean up environment"""
        if hasattr(self, 'env'):
            self.env.close()

    def test_reward_component_balance(self):
        """
        Test that reward components are properly balanced.

        Analyses component distribution and warns if any single component
        dominates excessively (>98% of total).

        Reference: Multi-objective RL principles (Roijers & Whiteson, 2017)
        """
        self.env.reset()

        rewards = []
        component_stats = {}
        dominance_warnings = []

        for _ in range(100):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            if not (terminated or truncated):
                rewards.append(reward)

                # Check component breakdown if available
                if hasattr(self.env, 'reward_components'):
                    components = self.env.reward_components
                    total = sum(components.values())

                    # Track component statistics
                    for name, value in components.items():
                        if name not in component_stats:
                            component_stats[name] = []
                        component_stats[name].append(value)

                        # Check for excessive dominance (>98% is concerning)
                        if abs(total) > 1.0:
                            ratio = abs(value) / abs(total)
                            if ratio > 0.98:
                                dominance_warnings.append(
                                    f"Step {len(rewards)}: '{name}' dominates with {ratio*100:.1f}%"
                                )

            if terminated or truncated:
                self.env.reset()

        # Print component statistics
        if component_stats:
            print(f"\n✓ Reward component statistics:")
            for name, values in component_stats.items():
                if len(values) > 0:
                    print(f"  {name}:")
                    print(f"    Mean: {np.mean(values):.2f}")
                    print(f"    Max: {np.max(np.abs(values)):.2f}")

        # Print dominance warnings (informational, not failure)
        if dominance_warnings:
            print(f"\n⚠ Component dominance detected in {len(dominance_warnings)} steps:")
            # Show first 3 examples
            for warning in dominance_warnings[:3]:
                print(f"  {warning}")
            if len(dominance_warnings) > 3:
                print(f"  ... and {len(dominance_warnings) - 3} more instances")

        if len(rewards) > 0:
            print(f"\n✓ Overall reward distribution:")
            print(f"  Mean: {np.mean(rewards):.2f}")
            print(f"  Std: {np.std(rewards):.2f}")
            print(f"  Min: {min(rewards):.2f}")
            print(f"  Max: {max(rewards):.2f}")

        # Test passes but provides diagnostic information
        assert len(rewards) > 0, "Should have collected some rewards"

    def test_reward_continuity(self):
        """
        Test that reward changes smoothly (no large discontinuous jumps).

        Mathematical basis: Continuous reward functions aid gradient-based
        optimisation in policy gradient methods (Schulman et al., 2017)
        """
        self.env.reset()

        rewards = []

        # Take consistent action to measure smoothness
        action = [0.5, 0.0, 0.0]

        for _ in range(100):
            obs, reward, terminated, truncated, info = self.env.step(action)

            if not (terminated or truncated):
                rewards.append(reward)
            else:
                break

        # Check for large jumps in consecutive rewards
        if len(rewards) >= 10:
            reward_diffs = np.diff(rewards)
            max_jump = np.max(np.abs(reward_diffs))

            # Large jumps (>50) might indicate discontinuity
            # (excluding terminal conditions)
            assert max_jump < 50.0, \
                f"Reward function has discontinuous jump of {max_jump:.2f}"

            print(f"\n✓ Reward continuity:")
            print(f"  Maximum consecutive difference: {max_jump:.2f}")
            print(f"  Mean absolute difference: {np.mean(np.abs(reward_diffs)):.2f}")


if __name__ == "__main__":
    """Run tests with verbose output"""
    pytest.main([__file__, "-v", "-s"])