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
    - Positions of the ball and 2 robots randomised at the start of a new episode
    - 
    """
    
    # Constructor to initialise the environment including action, observation spaces along with state variables
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

        # Game parameters
        self.possession_distance = 25.0
        self.collision_distance = 30.0

        # Initialise state variables first
        self.robot_pos = None
        self.robot_angle = None
        self.robot_vel = None
        self.ball_pos = None
        self.ball_vel = None
        self.opponent_pos = None
        self.opponent_has_ball = None
        self.goal_pos = None
        self.has_ball = None
        self.steps = None
        self.max_steps = None

        # Track possession time for better rewards
        self.robot_possession_time = 0
        self.opponent_possession_time = 0

        self.reset()
    
    # Function to setup variables for a new episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise starting positions
        # This is essential for the robot to learn generalisable behavior
        
        # Robot starts somewhere on left half of field
        self.robot_pos = np.array([
            np.random.uniform(30, 180),   # Left half: x between 30-180
            np.random.uniform(50, 350)    # y can be anywhere reasonable
        ])

        # Random initial robot orientation
        self.robot_angle = np.random.uniform(-np.pi/2, np.pi/2)  # Roughly facing right
        self.robot_vel = np.array([0.0, 0.0])
        
        # Ball starts in random position (not too close to goal)
        self.ball_pos = np.array([
            np.random.uniform(60, 320),   # Anywhere except very close to goal
            np.random.uniform(50, 350)
        ])
        self.ball_vel = np.array([0.0, 0.0])
        
        # Opponent starts in random position (prefer middle/right side)
        self.opponent_pos = np.array([
            np.random.uniform(150, 350),  # Middle to right side
            np.random.uniform(50, 350)
        ])
        
        # Ensure minimum distances to avoid starting in collision
        while np.linalg.norm(self.robot_pos - self.opponent_pos) < 50:
            self.opponent_pos = np.array([
                np.random.uniform(150, 350),
                np.random.uniform(50, 350)
            ])
        
        # # Robot starts on left side
        # self.robot_pos = np.array([50.0, 200.0])
        # self.robot_angle = 0.0
        # self.robot_vel = np.array([0.0, 0.0])
        
        # # Ball starts near robot
        # self.ball_pos = np.array([80.0, 200.0])
        # self.ball_vel = np.array([0.0, 0.0])
        
        # # Opponent in middle
        # self.opponent_pos = np.array([200.0, 200.0])
        
        # Goal on right side
        self.goal_pos = np.array([350.0, 200.0])
        
        # Game state
        self.has_ball = False
        
        self.opponent_has_ball = False
        self.steps = 0
        self.max_steps = 500  # Shorter episodes for more variety
        self.robot_possession_time = 0
        self.opponent_possession_time = 0
        
        return self._get_obs(), {}
    
    # Function to 
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
        
        # Check ball possession for both robot and opponent
        robot_ball_dist = np.linalg.norm(self.robot_pos - self.ball_pos)
        opponent_ball_dist = np.linalg.norm(self.opponent_pos - self.ball_pos)

        # # Update ball possession statuses: whoever is closer gets possession (if within range)
        # if robot_ball_dist < 25 and robot_ball_dist <= opponent_ball_dist:
        #     self.has_ball = True
        #     self.opponent_has_ball = False
        # elif opponent_ball_dist < 25 and opponent_ball_dist < robot_ball_dist:
        #     self.has_ball = False
        #     self.opponent_has_ball = True
        # else:
        #     # Neither has possession
        #     self.has_ball = False
        #     self.opponent_has_ball = False

        # Possession logic: closer player gets ball, but with some "stickiness" to make it harder to steal the ball from the robot that has possession
        if robot_ball_dist < self.possession_distance:
            if not self.opponent_has_ball or robot_ball_dist < opponent_ball_dist * 0.8:
                self.has_ball = True
                self.opponent_has_ball = False
                self.robot_possession_time += 1
                self.opponent_possession_time = 0
        elif opponent_ball_dist < self.possession_distance:
            if not self.has_ball or opponent_ball_dist < robot_ball_dist * 0.8:
                self.has_ball = False
                self.opponent_has_ball = True
                self.opponent_possession_time += 1
                self.robot_possession_time = 0
        else:
            # Ball is loose. Neither player has possession
            self.has_ball = False
            self.opponent_has_ball = False
            self.robot_possession_time = 0
            self.opponent_possession_time = 0

        return np.array([
            robot_x, robot_y, self.robot_angle / np.pi,
            ball_x, ball_y,
            opponent_x, opponent_y,
            self.robot_vel[0] / 50, self.robot_vel[1] / 50,
            ball_distance, goal_distance,
            float(self.has_ball)
        ], dtype=np.float32)
    
    # To take an action
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
        
        # Keep robot in bounds
        self.robot_pos[0] = np.clip(self.robot_pos[0], 20, 380)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 20, 380)
        
        # Ball physics
        if self.has_ball:
            # Ball follows robot
            offset_distance = 20
            target_ball_pos = self.robot_pos + offset_distance * np.array([
                np.cos(self.robot_angle), np.sin(self.robot_angle)
            ])
            self.ball_pos += 0.4 * (target_ball_pos - self.ball_pos)
            
        elif self.opponent_has_ball:
            # Ball follows opponent
            offset_distance = 20
            # Opponent tries to move ball away from robot
            to_robot = self.robot_pos - self.opponent_pos
            if np.linalg.norm(to_robot) > 0:
                away_from_robot = -to_robot / np.linalg.norm(to_robot)
                target_ball_pos = self.opponent_pos + offset_distance * away_from_robot
                self.ball_pos += 0.4 * (target_ball_pos - self.ball_pos)
        else:
            # Ball is loose - apply physics
            self.ball_vel *= 0.9  # Friction
            self.ball_pos += self.ball_vel
            
            # Add some random ball movement when loose
            if np.random.random() < 0.1:
                self.ball_vel += np.random.uniform(-0.5, 0.5, 2)
        
        # Keep ball in bounds
        self.ball_pos[0] = np.clip(self.ball_pos[0], 15, 385)
        self.ball_pos[1] = np.clip(self.ball_pos[1], 15, 385)
        
        # IMPROVED: Smarter opponent AI
        self._update_opponent()
        
        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_terminated()
        truncated = self.steps >= self.max_steps
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _update_opponent(self):
        """Improved opponent AI with different behaviours based on game state"""
        opponent_to_ball = self.ball_pos - self.opponent_pos
        opponent_to_robot = self.robot_pos - self.opponent_pos
        
        if self.opponent_has_ball:
            # Opponent has ball - try to keep it away from robot
            if np.linalg.norm(opponent_to_robot) < 60:
                # Robot is close - move away
                escape_dir = -opponent_to_robot / np.linalg.norm(opponent_to_robot)
                self.opponent_pos += 2.5 * escape_dir
            else:
                # Robot is far - move randomly to be unpredictable
                random_move = np.random.uniform(-1, 1, 2)
                self.opponent_pos += 1.0 * random_move
                
        elif self.has_ball:
            # Robot has ball - try to intercept
            if np.linalg.norm(opponent_to_robot) > 30:
                # Move toward robot to pressure
                intercept_dir = opponent_to_robot / np.linalg.norm(opponent_to_robot)
                self.opponent_pos += 2.0 * intercept_dir
            else:
                # Already close - circle around
                angle = np.arctan2(opponent_to_robot[1], opponent_to_robot[0]) + 0.3
                circle_move = np.array([np.cos(angle), np.sin(angle)])
                self.opponent_pos += 1.5 * circle_move
        else:
            # Ball is loose - chase it
            if np.linalg.norm(opponent_to_ball) > 5:
                ball_dir = opponent_to_ball / np.linalg.norm(opponent_to_ball)
                self.opponent_pos += 2.0 * ball_dir
        
        # Keep opponent in bounds
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], 20, 380)
        self.opponent_pos[1] = np.clip(self.opponent_pos[1], 20, 380)

    def _calculate_reward(self):
        reward = 0.0
        
        # Time penalty
        reward -= 0.01
        
        # Reward for having ball
        if self.has_ball:
            reward += 0.1
            
            # Progressive reward for keeping ball longer
            reward += min(self.robot_possession_time * 0.001, 0.05)

            # Reward for moving toward goal with ball
            goal_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
            reward += 0.005 * (400 - goal_distance) / 400

            # Bonus for being close to goal with ball
            if goal_distance < 100:
                reward += 0.2
        
        if self.opponent_has_ball:
            reward -= 0.02  # Penalty for losing possession to opponent

            # Additional penalty if opponent has ball for long time
            reward -= min(self.opponent_possession_time * 0.001, 0.03)

        # Check if scored, then give a huge reward
        if (self.ball_pos[0] > 340 and 
            160 < self.ball_pos[1] < 240):
            reward += 100.0
        
        # Penalty for collision with opponent
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < 30:
            reward -= 5.0
        
        # Small penalty for being too far from action
        ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        if ball_distance > 150:  # If robot is very far from ball
            reward -= 0.01

        return reward
    
    def _check_terminated(self):
        # Success: ball in goal
        if (self.ball_pos[0] > 340 and 160 < self.ball_pos[1] < 240):
            return True
        
        # Failure: collision with opponent
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < 25:
            return True

        # Failure: end if opponent controls ball for too long (30 steps)
        if self.opponent_possession_time > 30:
            return True
        
        return False
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.field_width, self.field_height))
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            
        # Clear screen
        self.window.fill((0, 128, 0))  # Green field
        
        # Draw field boundary
        pygame.draw.rect(self.window, (255, 255, 255), (10, 10, 380, 380), 2)
        
        # Draw goal
        pygame.draw.rect(self.window, (255, 255, 0), (340, 160, 60, 80))
        pygame.draw.rect(self.window, (255, 255, 255), (340, 160, 60, 80), 2)
        
        # Draw ball with different colors based on possession
        if self.has_ball:
            ball_color = (0, 255, 0)  # Green if robot has ball
        elif self.opponent_has_ball:
            ball_color = (255, 0, 0)  # Red if opponent has ball
        else:
            ball_color = (255, 255, 255)  # White if no one has ball

        pygame.draw.circle(self.window, ball_color, 
                          self.ball_pos.astype(int), 8)
        pygame.draw.circle(self.window, (0, 0, 0), 
                          self.ball_pos.astype(int), 8, 2)
        
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
            self.window = None
            self.clock = None


# Test the environment
if __name__ == "__main__":
    env = SimpleSoccerEnv(render_mode="human")
    obs, _ = env.reset()
    
    print("Testing Simple Soccer Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    try:

        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()

            print(f"\n--- Step {i + 1} ---")
            print(f"Robot starts at: ({env.robot_pos[0]:.0f}, {env.robot_pos[1]:.0f})")
            print(f"Ball starts at: ({env.ball_pos[0]:.0f}, {env.ball_pos[1]:.0f})")
            print(f"Opponent starts at: ({env.opponent_pos[0]:.0f}, {env.opponent_pos[1]:.0f})")

            if i % 100 == 0:
                print(f"Step {i}: Reward={reward:.3f}, Has ball={obs[11]}")
            
            if terminated:
                if env.ball_pos[0] > 340 and 160 < env.ball_pos[1] < 240:
                    print(f"Episode ended at step {i}: GOAL SCORED!")
                elif env.opponent_has_ball:
                    print(f"Episode ended at step {i}: OPPONENT GOT BALL!")
                else:
                    print(f"Episode ended at step {i}: COLLISION!")
                obs, _ = env.reset()
            elif truncated:
                print(f"Episode ended at step {i}: TIMEOUT!")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        env.close()
        print("Environment closed properly")