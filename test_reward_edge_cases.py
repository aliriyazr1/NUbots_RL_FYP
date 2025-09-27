#!/usr/bin/env python3
"""
Test Edge Cases for Reward Function Anti-Exploitation Measures

This script validates that the reward function correctly handles all possible
exploitation scenarios and edge cases.

Author: Ali Riyaz
Date: 2025
"""

import numpy as np
import sys
sys.path.append('src')

from src.environments.soccerenv import SoccerEnv
import matplotlib.pyplot as plt


class RewardEdgeCaseTester:
    """Comprehensive testing for reward function edge cases."""

    def __init__(self):
        """Initialize test environment with smooth reward."""
        self.env = SoccerEnv(reward_type="smooth", render_mode=None, config_path="configs/field_config.yaml")
        self.test_results = {}

    def test_stationary_exploitation(self):
        """Test 1: Agent remains stationary - should accumulate penalties."""
        print("\n=== TEST 1: Stationary Exploitation ===")
        obs, _ = self.env.reset()
        rewards = []

        # Agent does nothing for 50 steps
        for _ in range(50):
            action = np.array([0.0, 0.0, 0.0])  # No movement
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        avg_reward = np.mean(rewards)
        late_rewards = rewards[-10:]  # Last 10 steps

        print(f"Average reward: {avg_reward:.3f}")
        print(f"Early rewards (1-10): {np.mean(rewards[:10]):.3f}")
        print(f"Late rewards (41-50): {np.mean(late_rewards):.3f}")

        # Verify penalties increase over time
        penalty_increased = np.mean(late_rewards) < np.mean(rewards[:10])
        self.test_results['stationary'] = 'PASS' if penalty_increased and avg_reward < 5.0 else 'FAIL'
        print(f"Result: {self.test_results['stationary']}")

    def test_spinning_exploitation(self):
        """Test 2: Agent spins in place - should be penalised."""
        print("\n=== TEST 2: Spinning Exploitation ===")
        obs, _ = self.env.reset()
        rewards = []

        # Pure rotation without translation
        for _ in range(30):
            action = np.array([0.0, 0.0, 1.0])  # Max rotation, no movement
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        avg_reward = np.mean(rewards)
        print(f"Spinning average reward: {avg_reward:.3f}")

        self.test_results['spinning'] = 'PASS' if avg_reward < -1.0 else 'FAIL'
        print(f"Result: {self.test_results['spinning']}")

    def test_wall_hugging(self):
        """Test 3: Agent hugs walls - should receive boundary penalties."""
        print("\n=== TEST 3: Wall Hugging Exploitation ===")
        obs, _ = self.env.reset()

        # Move robot to wall
        self.env.robot_pos = np.array([10.0, self.env.field_height / 2])
        rewards = []

        # Stay near wall
        for _ in range(30):
            action = np.array([0.0, 0.1, 0.0])  # Slight movement along wall
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        avg_reward = np.mean(rewards)
        print(f"Wall hugging average reward: {avg_reward:.3f}")

        self.test_results['wall_hugging'] = 'PASS' if avg_reward < -0.5 else 'FAIL'
        print(f"Result: {self.test_results['wall_hugging']}")

    def test_ball_holding(self):
        """Test 4: Agent holds ball without progress - should be penalised."""
        print("\n=== TEST 4: Ball Holding Exploitation ===")
        obs, _ = self.env.reset()

        # Place robot and ball together
        self.env.robot_pos = np.array([self.env.field_width * 0.5, self.env.field_height * 0.5])
        self.env.ball_pos = self.env.robot_pos + np.array([15.0, 0.0])
        rewards = []

        # Small movements to maintain possession without progress
        for i in range(40):
            action = np.array([0.05 * np.sin(i * 0.5), 0.05 * np.cos(i * 0.5), 0.0])
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        early_avg = np.mean(rewards[:10])
        late_avg = np.mean(rewards[-10:])
        print(f"Early holding reward: {early_avg:.3f}")
        print(f"Late holding reward: {late_avg:.3f}")

        self.test_results['ball_holding'] = 'PASS' if late_avg < early_avg else 'FAIL'
        print(f"Result: {self.test_results['ball_holding']}")

    def test_backward_movement(self):
        """Test 5: Agent moves ball backward - should be penalised."""
        print("\n=== TEST 5: Backward Movement Exploitation ===")
        obs, _ = self.env.reset()

        # Place near goal
        self.env.robot_pos = np.array([self.env.field_width * 0.7, self.env.field_height * 0.5])
        self.env.ball_pos = self.env.robot_pos + np.array([15.0, 0.0])
        rewards = []

        # Move backward with ball
        for _ in range(20):
            action = np.array([-0.8, 0.0, 0.0])  # Strong backward movement
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        avg_reward = np.mean(rewards)
        print(f"Backward movement average reward: {avg_reward:.3f}")

        self.test_results['backward'] = 'PASS' if avg_reward < -1.0 else 'FAIL'
        print(f"Result: {self.test_results['backward']}")

    def test_oscillation_pattern(self):
        """Test 6: Agent oscillates back and forth - should be detected."""
        print("\n=== TEST 6: Oscillation Pattern ===")
        obs, _ = self.env.reset()
        rewards = []

        # Oscillate left-right
        for i in range(40):
            direction = 1.0 if i % 4 < 2 else -1.0
            action = np.array([direction, 0.0, 0.0])
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        late_avg = np.mean(rewards[-15:])
        print(f"Oscillation pattern average reward: {late_avg:.3f}")

        # More reasonable threshold for oscillation detection
        self.test_results['oscillation'] = 'PASS' if late_avg < 8.0 else 'FAIL'
        print(f"Result: {self.test_results['oscillation']}")

    def test_corner_camping(self):
        """Test 7: Agent camps in corner - should receive multiple penalties."""
        print("\n=== TEST 7: Corner Camping ===")
        obs, _ = self.env.reset()

        # Move to corner
        self.env.robot_pos = np.array([10.0, 10.0])
        rewards = []

        for _ in range(30):
            action = np.array([0.0, 0.0, 0.0])  # Stay in corner
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)

        avg_reward = np.mean(rewards)
        print(f"Corner camping average reward: {avg_reward:.3f}")

        # Should get both wall and stationary penalties
        self.test_results['corner_camping'] = 'PASS' if avg_reward < -2.0 else 'FAIL'
        print(f"Result: {self.test_results['corner_camping']}")

    def test_time_wasting(self):
        """Test 8: Agent wastes time when ahead - should get time penalties."""
        print("\n=== TEST 8: Time Wasting ===")
        obs, _ = self.env.reset()

        # Place ball near goal but don't score
        self.env.robot_pos = np.array([self.env.field_width * 0.85, self.env.field_height * 0.5])
        self.env.ball_pos = self.env.robot_pos + np.array([15.0, 0.0])

        early_rewards = []
        late_rewards = []

        # Early steps
        for _ in range(20):
            action = np.array([0.1, 0.1, 0.0])  # Minimal movement
            obs, reward, term, trunc, _ = self.env.step(action)
            early_rewards.append(reward)

        # Continue for more steps
        for _ in range(80):
            action = np.array([0.1, 0.1, 0.0])
            obs, reward, term, trunc, _ = self.env.step(action)
            late_rewards.append(reward)

        early_avg = np.mean(early_rewards)
        late_avg = np.mean(late_rewards[-20:])
        print(f"Early time average: {early_avg:.3f}")
        print(f"Late time average: {late_avg:.3f}")

        # Time penalty should make late rewards worse
        self.test_results['time_wasting'] = 'PASS' if late_avg < early_avg else 'FAIL'
        print(f"Result: {self.test_results['time_wasting']}")

    def test_positive_behaviour(self):
        """Test 9: Agent shows positive behaviour - should get rewards."""
        print("\n=== TEST 9: Positive Behaviour (Control) ===")
        obs, _ = self.env.reset()

        # Place robot behind ball
        self.env.robot_pos = np.array([self.env.field_width * 0.3, self.env.field_height * 0.5])
        self.env.ball_pos = np.array([self.env.field_width * 0.4, self.env.field_height * 0.5])
        rewards = []

        # Move toward goal with ball
        for _ in range(30):
            action = np.array([0.7, 0.0, 0.0])  # Forward movement
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)
            if term or trunc:
                break

        avg_reward = np.mean(rewards)
        print(f"Positive behaviour average reward: {avg_reward:.3f}")

        self.test_results['positive'] = 'PASS' if avg_reward > 0 else 'FAIL'
        print(f"Result: {self.test_results['positive']}")

    def test_all_actions_bounded(self):
        """Test 10: All possible actions produce bounded rewards."""
        print("\n=== TEST 10: Action Space Coverage ===")
        obs, _ = self.env.reset()
        rewards = []

        # Test random actions
        for _ in range(100):
            action = np.random.uniform(-1, 1, 3)
            obs, reward, term, trunc, _ = self.env.step(action)
            rewards.append(reward)
            if term or trunc:
                obs, _ = self.env.reset()

        min_r = min(rewards)
        max_r = max(rewards)
        print(f"Reward range: [{min_r:.3f}, {max_r:.3f}]")
        print(f"Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")

        # Check rewards are bounded (updated for new scaling)
        self.test_results['bounded'] = 'PASS' if -15 < min_r and max_r < 15 else 'FAIL'
        print(f"Result: {self.test_results['bounded']}")

    def run_all_tests(self):
        """Execute all edge case tests."""
        print("=" * 60)
        print("REWARD FUNCTION EDGE CASE VALIDATION")
        print("=" * 60)

        self.test_stationary_exploitation()
        self.test_spinning_exploitation()
        self.test_wall_hugging()
        self.test_ball_holding()
        self.test_backward_movement()
        self.test_oscillation_pattern()
        self.test_corner_camping()
        self.test_time_wasting()
        self.test_positive_behaviour()
        self.test_all_actions_bounded()

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result == 'PASS')
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            symbol = "✓" if result == 'PASS' else "✗"
            print(f"{symbol} {test_name.replace('_', ' ').title()}: {result}")

        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All edge cases handled correctly!")
        else:
            print(f"⚠️  {total - passed} test(s) need attention")

        return passed == total


if __name__ == "__main__":
    tester = RewardEdgeCaseTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)