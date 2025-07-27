#TODO: If I do end up using this, remember to fix the comments

# simple_soccer_env.py - Basic 2D Soccer Environment for FYP
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class SimpleSoccerEnv(gym.Env):
    """
    Simple 2D Soccer Environment for FYP
    - Robot needs to dribble ball around opponent to goal
    - Meets all your project requirements
    - Works on Windows without CUDA issues
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Field dimensions (4x4 meters as per field size requirements)
        self.field_width = 400  # pixels
        self.field_height = 400  # pixels
        self.goal_width = 80
        
        # Action space: [forward/back, left/right, rotation]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: [robot_x, robot_y, robot_angle, ball_x, ball_y, 
        #                    opponent_x, opponent_y, robot_vx, robot_vy, 
        #                    ball_distance, goal_distance, has_ball]
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Initialise state variables first
        self.robot_pos = None
        self.robot_angle = None
        self.robot_vel = None
        self.ball_pos = None
        self.ball_vel = None
        self.opponent_pos = None
        self.goal_pos = None
        self.has_ball = None
        self.steps = None
        self.max_steps = None
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Robot starts on left side
        self.robot_pos = np.array([50.0, 200.0])
        self.robot_angle = 0.0
        self.robot_vel = np.array([0.0, 0.0])
        
        # Ball starts near robot
        self.ball_pos = np.array([80.0, 200.0])
        self.ball_vel = np.array([0.0, 0.0])
        
        # Opponent in middle
        self.opponent_pos = np.array([200.0, 200.0])
        
        # Goal on right side
        self.goal_pos = np.array([350.0, 200.0])
        
        # Game state
        self.has_ball = False
        self.steps = 0
        self.max_steps = 1000
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Normalise positions to [-1, 1] range
        robot_x = (self.robot_pos[0] - 200) / 200
        robot_y = (self.robot_pos[1] - 200) / 200
        ball_x = (self.ball_pos[0] - 200) / 200
        ball_y = (self.ball_pos[1] - 200) / 200
        opponent_x = (self.opponent_pos[0] - 200) / 200
        opponent_y = (self.opponent_pos[1] - 200) / 200
        
        # Calculate distances
        ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos) / 200
        goal_distance = np.linalg.norm(self.robot_pos - self.goal_pos) / 200
        
        # Update ball possession status
        self.has_ball = np.linalg.norm(self.robot_pos - self.ball_pos) < 25

        return np.array([
            robot_x, robot_y, self.robot_angle / np.pi,
            ball_x, ball_y,
            opponent_x, opponent_y,
            self.robot_vel[0] / 50, self.robot_vel[1] / 50,
            ball_distance, goal_distance,
            float(self.has_ball)
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        # Clip action to ensure it's in valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply action to robot
        speed = 3.0
        forward = action[0] * speed
        strafe = action[1] * speed
        rotation = action[2] * 0.1
        
        # Update robot angle first
        self.robot_angle += rotation
        # Keep angle in reasonable range
        self.robot_angle = np.clip(self.robot_angle, -np.pi, np.pi)
        
        # Update robot velocity
        self.robot_vel[0] = forward * np.cos(self.robot_angle) - strafe * np.sin(self.robot_angle)
        self.robot_vel[1] = forward * np.sin(self.robot_angle) + strafe * np.cos(self.robot_angle)
        
        # Update robot position
        self.robot_pos += self.robot_vel
        self.robot_angle += rotation
        
        # Keep robot in bounds
        self.robot_pos[0] = np.clip(self.robot_pos[0], 20, 380)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 20, 380)
        
        # Ball physics
        ball_robot_dist = np.linalg.norm(self.robot_pos - self.ball_pos)
        
        if ball_robot_dist < 25:  # Robot touches ball
            self.has_ball = True
            # Ball follows robot
            target_ball_pos = self.robot_pos + 15 * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            self.ball_pos += 0.3 * (target_ball_pos - self.ball_pos)
        else:
            self.has_ball = False
            # Ball decelerates
            self.ball_vel *= 0.95
        
        self.ball_pos += self.ball_vel
        self.ball_pos[0] = np.clip(self.ball_pos[0], 10, 390)
        self.ball_pos[1] = np.clip(self.ball_pos[1], 10, 390)
        
        # Simple opponent AI - moves toward ball
        opponent_to_ball = self.ball_pos - self.opponent_pos
        if np.linalg.norm(opponent_to_ball) > 5:
            self.opponent_pos += 1.5 * opponent_to_ball / np.linalg.norm(opponent_to_ball)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._check_terminated()
        truncated = self.steps >= self.max_steps
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        reward = 0.0
        
        # Time penalty
        reward -= 0.001
        
        # Reward for having ball
        if self.has_ball:
            reward += 0.01
            
            # Reward for moving toward goal with ball
            goal_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
            reward += 0.005 * (400 - goal_distance) / 400
        
        # Check if scored
        if (self.ball_pos[0] > 340 and 
            160 < self.ball_pos[1] < 240):
            reward += 100.0
        
        # Penalty for collision with opponent
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < 30:
            reward -= 5.0
        
        return reward
    
    def _check_terminated(self):
        # Success: ball in goal
        if (self.ball_pos[0] > 340 and 160 < self.ball_pos[1] < 240):
            return True
        
        # Failure: collision with opponent
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < 25:
            return True
        
        return False
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.field_width, self.field_height))
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((0, 128, 0))  # Green field
        
        # Draw goal
        pygame.draw.rect(self.window, (255, 255, 0), 
                        (340, 160, 60, 80))
        
        # Draw ball
        ball_color = (0, 255, 0) if self.has_ball else (255, 255, 255)
        pygame.draw.circle(self.window, ball_color, 
                          self.ball_pos.astype(int), 8)
        
        # Draw robot
        pygame.draw.circle(self.window, (0, 0, 255), 
                          self.robot_pos.astype(int), 15)
        
        # Draw robot direction
        end_pos = self.robot_pos + 20 * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        pygame.draw.line(self.window, (0, 0, 255), 
                        self.robot_pos, end_pos, 3)
        
        # Draw opponent
        pygame.draw.circle(self.window, (255, 0, 0), 
                          self.opponent_pos.astype(int), 15)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Test the environment
if __name__ == "__main__":
    env = SimpleSoccerEnv(render_mode="human")
    obs, _ = env.reset()
    
    print("Testing Simple Soccer Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if i % 100 == 0:
            print(f"Step {i}: Reward={reward:.3f}, Has ball={obs[11]}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, _ = env.reset()
    
    env.close()