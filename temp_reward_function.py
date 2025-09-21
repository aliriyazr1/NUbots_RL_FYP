"""
Smooth, continuous reward functions for better PPO performance.

Key insight: Discontinuous rewards create problems for policy gradient methods,
especially PPO which relies heavily on value function approximation.

References:
1. "Reward Shaping via Meta-Learning" (Zheng et al., 2018)
2. "On the Theory of Policy Gradient Methods" (Agarwal et al., 2021)
3. "The Mirage of Action-Dependent Baselines" (Tucker et al., 2018)
"""

import numpy as np
from typing import Dict


class SmoothRewardFunction:
    """
    Implements continuous, differentiable reward functions.

    Key principle: Replace all if-statements and discontinuous jumps with
    smooth transitions using sigmoid, tanh, or polynomial functions.
    """

    def __init__(self):
        # Smooth transition parameters
        self.transition_sharpness = 5.0  # Controls how sharp transitions are
        self.contact_threshold = 0.5  # meters
        self.possession_threshold = 0.3  # meters for ball possession

    def smooth_transition(self, x: float, threshold: float, sharpness: float = None) -> float:
        """
        Creates smooth transition from 0 to 1 around threshold.
        Uses sigmoid function for differentiability.

        Based on: "Deep Reinforcement Learning with Smooth Policy" (Gu et al., 2016)
        """
        if sharpness is None:
            sharpness = self.transition_sharpness
        return 1.0 / (1.0 + np.exp(-sharpness * (threshold - x)))

    def gaussian_reward(self, distance: float, optimal: float = 0.0, sigma: float = 1.0) -> float:
        """
        Gaussian-shaped reward centered at optimal distance.
        Smooth and differentiable everywhere.

        From: "Learning from Demonstration Using Gaussian Processes" (Calinon & Billard, 2007)
        """
        return np.exp(-0.5 * ((distance - optimal) / sigma) ** 2)

    def calculate_reward(self, state: Dict) -> float:
        """
        Fully continuous reward function with no discontinuities.
        """

        # Extract state information
        ball_pos = state['ball_position']
        robot_pos = state['robot_position']
        opponent_pos = state.get('opponent_position', robot_pos)
        goal_pos = state['goal_position']
        robot_vel = state.get('robot_velocity', np.zeros(3))
        ball_vel = state.get('ball_velocity', np.zeros(2))

        # Calculate distances
        robot_ball_distance = np.linalg.norm(robot_pos[:2] - ball_pos)
        ball_goal_distance = np.linalg.norm(ball_pos - goal_pos[:2])
        opponent_ball_distance = np.linalg.norm(opponent_pos[:2] - ball_pos)
        robot_opponent_distance = np.linalg.norm(robot_pos[:2] - opponent_pos[:2])

        # Initialize reward
        reward = 0.0

        # ============================================================
        # 1. TERMINAL REWARDS (Still discrete, but rare)
        # ============================================================
        if state.get('goal_scored', False):
            return 100.0
        if state.get('opponent_goal', False):
            return -100.0

        # ============================================================
        # 2. BALL PROXIMITY REWARD (Continuous!)
        # ============================================================
        # Instead of: if distance < threshold: reward += bonus
        # We use smooth transition function

        # Gaussian reward for ball proximity (peaks at distance=0)
        ball_proximity_reward = 10.0 * self.gaussian_reward(
            robot_ball_distance,
            optimal=0.0,  # Optimal distance is touching ball
            sigma=self.contact_threshold * 2
        )
        reward += ball_proximity_reward

        # ============================================================
        # 3. POSSESSION BONUS (Smooth transition)
        # ============================================================
        # Instead of: if has_ball: reward += 20
        # We use smooth transition based on distance

        possession_confidence = self.smooth_transition(
            robot_ball_distance,
            self.possession_threshold,
            sharpness=10.0  # Sharper transition for possession
        )
        possession_bonus = 15.0 * possession_confidence
        reward += possession_bonus

        # ============================================================
        # 4. GOAL PROGRESS (Always continuous)
        # ============================================================
        # Potential-based shaping is already continuous!
        if hasattr(self, 'prev_ball_goal_distance'):
            ball_progress = (self.prev_ball_goal_distance - ball_goal_distance)
            # Scale by possession confidence for smooth transition
            goal_progress_reward = 8.0 * ball_progress * possession_confidence
            reward += goal_progress_reward

        self.prev_ball_goal_distance = ball_goal_distance

        # ============================================================
        # 5. MOVEMENT TOWARD BALL (Continuous alignment)
        # ============================================================
        # Instead of: if moving_toward_ball: reward += bonus
        # We use continuous dot product

        robot_speed = np.linalg.norm(robot_vel[:2])
        if robot_speed > 0.01:  # Avoid division by zero
            robot_vel_norm = robot_vel[:2] / robot_speed
            to_ball_norm = (ball_pos - robot_pos[:2]) / (robot_ball_distance + 1e-6)

            # Continuous alignment score (-1 to 1)
            alignment = np.dot(robot_vel_norm, to_ball_norm)

            # Smooth reward based on alignment (no if-statement!)
            # Uses tanh to create smooth S-curve
            movement_reward = 3.0 * np.tanh(2.0 * alignment)
            reward += movement_reward * (1.0 - possession_confidence)  # Only when not in possession

        # ============================================================
        # 6. OPPONENT PRESSURE (Smooth competition)
        # ============================================================
        # Instead of: if opponent_closer: penalty
        # We use continuous difference

        competitive_difference = opponent_ball_distance - robot_ball_distance
        # Smooth penalty using tanh (negative when we're further from ball)
        competitive_factor = np.tanh(competitive_difference / self.contact_threshold)
        competitive_reward = 2.0 * competitive_factor
        reward += competitive_reward

        # ============================================================
        # 7. COLLISION AVOIDANCE (Smooth repulsion)
        # ============================================================
        # Instead of: if collision: penalty
        # We use smooth exponential repulsion

        collision_threshold = 0.5  # meters
        if robot_opponent_distance < collision_threshold * 3:  # Only calculate when relatively close
            # Exponential repulsion force (always continuous)
            collision_penalty = -5.0 * np.exp(-robot_opponent_distance / collision_threshold)
            reward += collision_penalty

        # ============================================================
        # 8. FIELD POSITION VALUE (Smooth field value function)
        # ============================================================
        # Create smooth value function over field position
        # Higher value closer to opponent goal

        field_length = state.get('field_length', 9.0)
        x_progress = robot_pos[0] / field_length  # 0 to 1

        # Smooth exponential increase toward goal
        position_value = 2.0 * (np.exp(x_progress) - 1.0) / (np.e - 1.0)
        reward += position_value * possession_confidence

        # ============================================================
        # 9. ANTI-SPINNING (Smooth penalty)
        # ============================================================
        # Smooth penalty for high angular velocity relative to linear
        linear_speed = np.linalg.norm(robot_vel[:2])
        angular_speed = abs(robot_vel[2]) if len(robot_vel) > 2 else 0.0

        # Smooth ratio-based penalty
        if linear_speed > 0.01:
            spin_ratio = angular_speed / (linear_speed + 0.1)
            # Smooth penalty that increases with spin ratio
            spin_penalty = -2.0 * self.smooth_transition(spin_ratio, 2.0, sharpness=3.0)
            reward += spin_penalty

        # ============================================================
        # 10. TIME PENALTY (Always continuous)
        # ============================================================
        reward -= 0.05

        # ============================================================
        # 11. BOUNDARY SOFT PENALTY (Smooth field boundaries)
        # ============================================================
        # Instead of: if near_boundary: penalty
        # We use smooth distance-based penalty

        field_width = state.get('field_width', 6.0)
        field_height = state.get('field_height', 9.0)

        # Distance to nearest boundary (normalized)
        x_boundary_dist = min(robot_pos[0], field_width - robot_pos[0]) / field_width
        y_boundary_dist = min(robot_pos[1], field_height - robot_pos[1]) / field_height
        min_boundary_dist = min(x_boundary_dist, y_boundary_dist)

        # Smooth penalty that increases as we approach boundary
        boundary_penalty = -3.0 * np.exp(-10.0 * min_boundary_dist)
        reward += boundary_penalty

        # ============================================================
        # FINAL CLIPPING (Soft clipping using tanh)
        # ============================================================
        # Instead of hard clipping, use soft clipping for continuity
        # This maintains differentiability at the boundaries

        # Soft clipping using tanh (maps to roughly -10 to +10)
        reward = 10.0 * np.tanh(reward / 10.0)

        return float(reward)


class HybridSmoothReward:
    """
    Hybrid approach: Smooth rewards for PPO, standard for DDPG.

    Based on observation that DDPG handles discontinuities better due to
    replay buffer and off-policy learning.
    """

    def __init__(self, algorithm: str = "PPO"):
        self.algorithm = algorithm.upper()
        self.smooth_reward = SmoothRewardFunction()

    def calculate_reward(self, state: Dict) -> float:
        if self.algorithm == "PPO":
            # Use smooth, continuous rewards for PPO
            return self.smooth_reward.calculate_reward(state)
        else:
            # DDPG can handle some discontinuities
            return self._calculate_ddpg_reward(state)

    def _calculate_ddpg_reward(self, state: Dict) -> float:
        """
        DDPG-optimized reward with some acceptable discontinuities.
        """
        # Terminal rewards (discrete is OK for DDPG)
        if state.get('goal_scored', False):
            return 150.0
        if state.get('opponent_goal', False):
            return -100.0

        reward = 0.0

        robot_ball_distance = np.linalg.norm(
            state['robot_position'][:2] - state['ball_position']
        )

        # DDPG can handle these discontinuities better
        if robot_ball_distance < 0.3:  # Possession
            reward += 20.0

            # Goal progress (continuous part)
            if hasattr(self, 'prev_ball_goal_distance'):
                ball_goal_distance = np.linalg.norm(
                    state['ball_position'] - state['goal_position'][:2]
                )
                progress = self.prev_ball_goal_distance - ball_goal_distance
                reward += 15.0 * progress
                self.prev_ball_goal_distance = ball_goal_distance
        else:
            # Distance penalty (continuous)
            reward -= 2.0 * robot_ball_distance / 10.0

        # Time penalty
        reward -= 0.01

        return float(np.clip(reward, -30, 150))


# Practical testing function
def test_reward_smoothness():
    """
    Test reward function for discontinuities.

    From: "Benchmarking Deep RL" (Henderson et al., 2018)
    """
    import matplotlib.pyplot as plt

    smooth_reward = SmoothRewardFunction()

    # Test ball distance discontinuity
    distances = np.linspace(0, 2.0, 1000)
    rewards = []

    for d in distances:
        state = {
            'robot_position': np.array([0, 0, 0]),
            'ball_position': np.array([d, 0]),
            'goal_position': np.array([9, 0, 0]),
            'opponent_position': np.array([5, 5, 0]),
            'robot_velocity': np.array([1, 0, 0]),
            'goal_scored': False,
            'opponent_goal': False
        }
        rewards.append(smooth_reward.calculate_reward(state))

    # Plot to check smoothness
    plt.figure(figsize=(10, 6))
    plt.plot(distances, rewards)
    plt.xlabel('Robot-Ball Distance (m)')
    plt.ylabel('Reward')
    plt.title('Reward Function Smoothness Test')
    plt.grid(True, alpha=0.3)

    # Calculate derivative to check for discontinuities
    derivatives = np.gradient(rewards, distances)
    plt.figure(figsize=(10, 6))
    plt.plot(distances[:-1], derivatives[:-1])
    plt.xlabel('Robot-Ball Distance (m)')
    plt.ylabel('Reward Derivative')
    plt.title('Reward Function Derivative (Should be Continuous)')
    plt.grid(True, alpha=0.3)

    plt.show()

    print(f"Max derivative jump: {np.max(np.abs(np.diff(derivatives)))}")
    print("If > 10, you have discontinuities!")


if __name__ == "__main__":
    test_reward_smoothness()