# File: tests/unit/test_environment.py
import pytest
import numpy as np
from SoccerEnv.soccerenv import SoccerEnv

class TestSoccerEnvironment:
    """Unit tests for SoccerEnv following your test plan requirements"""
    
    def setup_method(self):
        """Setup test environment"""
        self.env = SoccerEnv(render_mode=None, difficulty="easy")
    
    def test_environment_initialization(self):
        """Test environment initializes correctly"""
        assert self.env.field_width > 0
        assert self.env.field_height > 0
        assert self.env.action_space.shape == (3,)
        assert self.env.observation_space.shape == (12,)
    
    def test_coordinate_system_consistency(self):
        """Test coordinate system follows expected conventions"""
        obs, _ = self.env.reset()
        
        # Test forward movement
        action = [1.0, 0.0, 0.0]  # Forward
        initial_pos = self.env.robot_pos.copy()
        
        for _ in range(10):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        
        # Robot should move in positive X direction (toward goal)
        assert self.env.robot_pos[0] > initial_pos[0], "Forward movement should increase X"
    
    def test_action_space_bounds(self):
        """Test action space properly clips values"""
        obs, _ = self.env.reset()
        
        # Test extreme actions
        extreme_action = [10.0, -10.0, 5.0]
        obs, reward, terminated, truncated, info = self.env.step(extreme_action)
        
        # Should not crash and should be bounded
        assert not np.isnan(obs).any(), "Observation should not contain NaN"
        assert np.isfinite(reward), "Reward should be finite"
    
    def test_ball_physics_consistency(self):
        """Test ball physics behave consistently"""
        obs, _ = self.env.reset()
        initial_ball_pos = self.env.ball_pos.copy()
        
        # Move robot toward ball
        for _ in range(50):
            action = [1.0, 0.0, 0.0]  # Move forward
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        
        # Ball should move when robot contacts it
        ball_moved = np.linalg.norm(self.env.ball_pos - initial_ball_pos) > 5.0
        assert ball_moved, "Ball should move when robot contacts it"
    
    def test_reward_function_range(self):
        """Test reward function returns reasonable values"""
        obs, _ = self.env.reset()
        rewards = []
        
        for _ in range(100):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                obs, _ = self.env.reset()
        
        # Rewards should be finite and in reasonable range
        assert all(np.isfinite(r) for r in rewards), "All rewards should be finite"
        assert min(rewards) >= -1000, "Rewards should not be extremely negative"
        assert max(rewards) <= 200, "Rewards should not be extremely positive"