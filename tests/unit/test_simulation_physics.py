# File: tests/environment/test_physics.py
import pytest
import numpy as np
from SoccerEnv.soccerenv import SoccerEnv

class TestPhysicsConsistency:
    """Test physics calculations match expected behavior"""
    
    def test_robot_movement_directions(self):
        """Test robot moves in expected directions"""
        env = SoccerEnv(render_mode=None)
        obs, _ = env.reset()
        
        # Set robot to face right (toward goal)
        env.robot_angle = 0.0
        initial_pos = env.robot_pos.copy()
        
        #TODO: Decide whether the right and left strafe should be flipped
        # # Test each direction
        test_cases = [
            ([1.0, 0.0, 0.0], "forward", lambda p: p[0] > initial_pos[0]),
            ([-1.0, 0.0, 0.0], "backward", lambda p: p[0] < initial_pos[0]),
            ([0.0, 1.0, 0.0], "right_strafe", lambda p: p[1] != initial_pos[1]),
            ([0.0, -1.0, 0.0], "left_strafe", lambda p: p[1] != initial_pos[1])
        ]
        
        for action, direction, check_func in test_cases:
            env.robot_pos = initial_pos.copy()
            
            # Apply action multiple times
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            assert check_func(env.robot_pos), f"{direction} movement failed"
    
    def test_ball_robot_interaction(self):
        """Test ball-robot physics interaction"""
        env = SoccerEnv(render_mode=None)
        obs, _ = env.reset()
        
        # Position robot near ball
        env.robot_pos = env.ball_pos + np.array([-20.0, 0.0])
        initial_ball_pos = env.ball_pos.copy()
        
        # Push ball
        for _ in range(20):
            action = [1.0, 0.0, 0.0]  # Move toward ball
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Ball should move in direction of push
        ball_displacement = env.ball_pos - initial_ball_pos
        assert ball_displacement[0] > 0, "Ball should move forward when pushed"