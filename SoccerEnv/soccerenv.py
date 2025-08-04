#TODO: If I do end up using this, remember to fix the comments

# simple_soccer_env.py - Basic 2D Soccer Environment for FYP
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class SoccerEnv(gym.Env):
    """
    Simple 2D Soccer Environment for FYP
    - Robot needs to dribble ball around opponent to goal
    - Positions of the ball and 2 robots randomised at the start of a new episode
    - 
    """
    
    # Constructor to initialise the environment including action, observation spaces along with state variables
    def __init__(self, render_mode=None, difficulty="easy"):
        super().__init__()
        self.difficulty = difficulty
        
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

        # # Game parameters
        # self.possession_distance = 40.0 # Used to be 35.0
        # self.collision_distance = 30.0

        # Game parameters based on difficulty
        if difficulty == "easy":
            self.possession_distance = 50.0  # Easier to get ball
            self.collision_distance = 20.0   # More forgiving collisions
            self.max_steps = 150            # Shorter episodes
        elif difficulty == "medium":
            self.possession_distance = 40.0
            self.collision_distance = 25.0
            self.max_steps = 200
        else:  # hard
            self.possession_distance = 35.0
            self.collision_distance = 30.0
            self.max_steps = 250

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

        # Track possession time for better rewards
        self.robot_possession_time = 0
        self.opponent_possession_time = 0

        self.reset()
    
    # Function to setup variables for a new episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise starting positions
        # This is essential for the robot to learn generalisable behavior
        
        # Starting positions based on difficulty
        if self.difficulty == "easy":
            # Easy: Robot starts close to ball, while the opponent is far
            self.robot_pos = np.array([80.0, 200.0])
            self.ball_pos = np.array([120.0, 200.0])
            self.opponent_pos = np.array([300.0, 300.0])
        elif self.difficulty == "medium":
            # Medium: More challenging
            self.robot_pos = np.array([
                np.random.uniform(50, 150),
                np.random.uniform(100, 300)
            ])
            self.ball_pos = np.array([
                np.random.uniform(80, 200),
                np.random.uniform(100, 300)
            ])
            self.opponent_pos = np.array([
                np.random.uniform(200, 350),
                np.random.uniform(100, 300)
            ])
        else:  # hard
            # Hard: Random positions
            self.robot_pos = np.array([
                np.random.uniform(30, 180),
                np.random.uniform(50, 350)
            ])
            self.ball_pos = np.array([
                np.random.uniform(60, 320),
                np.random.uniform(50, 350)
            ])
            self.opponent_pos = np.array([
                np.random.uniform(150, 350),
                np.random.uniform(50, 350)
            ])

        # # Robot starts somewhere on left half of field
        # self.robot_pos = np.array([
        #     np.random.uniform(30, 180),   # Left half: x between 30-180
        #     np.random.uniform(50, 350)    # y can be anywhere reasonable
        # ])
        # # Ball starts in random position (not too close to goal)
        # self.ball_pos = np.array([
        #     np.random.uniform(60, 320),   # Anywhere except very close to goal
        #     np.random.uniform(50, 350)
        # ])

        # # Opponent starts in random position (prefer middle/right side)
        # self.opponent_pos = np.array([
        #     np.random.uniform(150, 350),  # Middle to right side
        #     np.random.uniform(50, 350)
        # ])

        # Random initial robot orientation
        self.robot_angle = np.random.uniform(-np.pi/4, np.pi/4)  # Roughly facing right (Used to be between -pi/2 and pi/2)
        self.robot_vel = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.goal_pos = np.array([350.0, 200.0]) # Goal center on right side

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
        
        
        # Game state
        self.has_ball = False
        self.opponent_has_ball = False
        self.steps = 0
        self.robot_possession_time = 0
        self.opponent_possession_time = 0
        
        return self._get_obs(), {}
    
    # Function to 
    def _get_obs(self):
        # Normalise positions to [-1, 1] range with clipping
        robot_x = np.clip((self.robot_pos[0] - 200) / 200, -1, 1)
        robot_y = np.clip((self.robot_pos[1] - 200) / 200, -1, 1)
        ball_x = np.clip((self.ball_pos[0] - 200) / 200, -1, 1)
        ball_y = np.clip((self.ball_pos[1] - 200) / 200, -1, 1)
        opponent_x = np.clip((self.opponent_pos[0] - 200) / 200, -1, 1)
        opponent_y = np.clip((self.opponent_pos[1] - 200) / 200, -1, 1)
        
        # Calculate distances with proper normalisation
        ball_distance = np.clip(np.linalg.norm(self.robot_pos - self.ball_pos) / 200, 0, 1)
        goal_distance = np.clip(np.linalg.norm(self.robot_pos - self.goal_pos) / 200, 0, 1)
        
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

        # return np.array([
        #     robot_x, robot_y, self.robot_angle / np.pi,
        #     ball_x, ball_y,
        #     opponent_x, opponent_y,
        #     self.robot_vel[0] / 50, self.robot_vel[1] / 50,
        #     ball_distance, goal_distance,
        #     float(self.has_ball)
        # ], dtype=np.float32)

        # All observations properly bounded
        obs = np.array([
            robot_x, robot_y, 
            np.clip(self.robot_angle / np.pi, -1, 1),
            ball_x, ball_y,
            opponent_x, opponent_y,
            np.clip(self.robot_vel[0] / 5, -1, 1), 
            np.clip(self.robot_vel[1] / 5, -1, 1),
            ball_distance, goal_distance,
            float(self.has_ball)
        ], dtype=np.float32)
        
        # Ensure no NaN/inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
    
    # To take an action
    def step(self, action):
        self.steps += 1

        # Clip action to ensure it's in valid range
        action = np.clip(action, -1.0, 1.0)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # Apply action to robot
        speed = 2.5
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
            if np.linalg.norm(to_robot) > 1e-6: # Prevent division by zero
                away_from_robot = -to_robot / np.linalg.norm(to_robot)
                target_ball_pos = self.opponent_pos + offset_distance * away_from_robot
                self.ball_pos += 0.4 * (target_ball_pos - self.ball_pos)
        else:
            # Ball is loose - apply physics
            self.ball_vel *= 0.9  # Friction
            self.ball_pos += self.ball_vel
            
            # Add some random ball movement when loose
            if np.random.random() < 0.05:
                self.ball_vel += np.random.uniform(-0.3, 0.3, 2)
        
        # Keep ball in bounds
        self.ball_pos[0] = np.clip(self.ball_pos[0], 15, 385)
        self.ball_pos[1] = np.clip(self.ball_pos[1], 15, 385)
        
        # Opponent AI
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
        """Opponent AI with different behaviours based on game state and difficulty"""
        opponent_to_ball = self.ball_pos - self.opponent_pos
        opponent_to_robot = self.robot_pos - self.opponent_pos

        # Prevent division by zero
        robot_dist = np.linalg.norm(opponent_to_robot)
        ball_dist = np.linalg.norm(opponent_to_ball)

        if self.difficulty == "easy":
            # Easy: Slow, predictable opponent
            if ball_dist > 5:
                ball_dir = opponent_to_ball / (ball_dist + 1e-6)
                self.opponent_pos += 1.0 * ball_dir
                
        elif self.difficulty == "medium":
            # Medium: More active
            if self.opponent_has_ball:
                if robot_dist > 1e-6 and robot_dist < 60:
                    escape_dir = -opponent_to_robot / robot_dist
                    self.opponent_pos += 1.5 * escape_dir
            elif ball_dist > 5:
                ball_dir = opponent_to_ball / (ball_dist + 1e-6)
                self.opponent_pos += 1.8 * ball_dir
                
        else:  # hard
            # Hard: Smart opponent (your original logic with safety)
            if self.opponent_has_ball:
                if robot_dist > 1e-6 and robot_dist < 60:
                    escape_dir = -opponent_to_robot / robot_dist
                    self.opponent_pos += 2.0 * escape_dir
                else:
                    random_move = np.random.uniform(-0.8, 0.8, 2)
                    self.opponent_pos += random_move
            elif self.has_ball:
                if robot_dist > 1e-6 and robot_dist > 30:
                    intercept_dir = opponent_to_robot / robot_dist
                    self.opponent_pos += 1.8 * intercept_dir
            else:
                if ball_dist > 5:
                    ball_dir = opponent_to_ball / (ball_dist + 1e-6)
                    self.opponent_pos += 1.8 * ball_dir
        
        # if self.opponent_has_ball:
        #     # Opponent has ball - try to keep it away from robot
        #     if np.linalg.norm(opponent_to_robot) < 60:
        #         # Robot is close - move away
        #         escape_dir = -opponent_to_robot / np.linalg.norm(opponent_to_robot)
        #         self.opponent_pos += 2.5 * escape_dir
        #     else:
        #         # Robot is far - move randomly to be unpredictable
        #         random_move = np.random.uniform(-1, 1, 2)
        #         self.opponent_pos += 1.0 * random_move
                
        # elif self.has_ball:
        #     # Robot has ball - try to intercept
        #     if np.linalg.norm(opponent_to_robot) > 30:
        #         # Move toward robot to pressure
        #         intercept_dir = opponent_to_robot / np.linalg.norm(opponent_to_robot)
        #         self.opponent_pos += 2.0 * intercept_dir
        #     else:
        #         # Already close - circle around
        #         angle = np.arctan2(opponent_to_robot[1], opponent_to_robot[0]) + 0.3
        #         circle_move = np.array([np.cos(angle), np.sin(angle)])
        #         self.opponent_pos += 1.5 * circle_move
        # else:
        #     # Ball is loose - chase it
        #     if np.linalg.norm(opponent_to_ball) > 5:
        #         ball_dir = opponent_to_ball / np.linalg.norm(opponent_to_ball)
        #         self.opponent_pos += 2.0 * ball_dir
        
        # Keep opponent in bounds
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], 20, 380)
        self.opponent_pos[1] = np.clip(self.opponent_pos[1], 20, 380)

    def _calculate_reward(self):
        reward = 0.0
        
        # Time penalty
        reward -= 0.002

        # Distance calculations
        ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        robot_x, robot_y = self.robot_pos
        # goal_distance = np.linalg.norm(self.robot_pos - self.goal_pos)

        # Goal distance that considers BOTH X and Y coordinates
        goal_center = np.array([350.0, 200.0])  # Center of goal
        goal_distance = np.linalg.norm(self.robot_pos - goal_center)

        # Check if robot is actually in front of goal (not just right side)
        robot_x, robot_y = self.robot_pos
        goal_aligned = (robot_x > 300 and 160 < robot_y < 240)  # Actually in goal area
        
        # PHASE 1: Getting the ball (most important for learning)
        if not self.has_ball:
            # Dense reward shaping for approaching ball
            if ball_distance < 30:
                reward += 1.0    # Close to ball
            elif ball_distance < 50:
                reward += 0.5    # Getting closer
            elif ball_distance < 80:
                reward += 0.2    # Moving in right direction
            
            # Penalty for being far from ball
            if ball_distance > 150:
                reward -= 0.3

        # PHASE 2: Moving with ball (second priority)
        if self.has_ball:
            # Good reward for having ball
            reward += 2.0
            
            # Define optimal shooting positions (directly in front of goal opening)
            optimal_shooting_zone = (robot_x > 290 and robot_x < 340 and robot_y > 170 and robot_y < 230)  # Directly in front of goal
            
            # Define goalpost camping zones (next to goalposts but can't score)
            near_top_goalpost = (robot_x > 320 and robot_y < 160)     # Above goal
            near_bottom_goalpost = (robot_x > 320 and robot_y > 240)  # Below goal
            goalpost_camping = near_top_goalpost or near_bottom_goalpost

            # Reward for moving toward goal center, not just right edge
            # Considers both distance AND alignment
            if optimal_shooting_zone:
                # Reward for the robot if aligned with goal
                reward += 5.0
                # Additional reward for being close to goal center when aligned
                center_distance = np.linalg.norm(self.robot_pos - goal_center)
                reward += 3.0 * (200 - center_distance) / 200
            elif goalpost_camping:
                # PENALTY for camping next to goalposts (can't score from there)
                reward -= 3.0  # Strong penalty for goalpost camping
                print(f"⚠️ GOALPOST CAMPING DETECTED at ({robot_x:.0f}, {robot_y:.0f})")
            else:
                # Robot has ball but not aligned with goal
                # Small reward for moving right, but penalty for going to wrong Y position
                rightward_progress = max(0, (robot_x - 100) / 250)  # Progress rightward
                reward += rightward_progress * 0.5
                
                # Penalty for being far from goal Y-center (prevents edge camping)
                y_distance_from_goal_center = abs(robot_y - 200)  # Distance from Y=200
                if y_distance_from_goal_center > 100:  # Far from goal center
                    reward -= 1.0  # Penalty for being at wrong Y position
            
            # Possession time bonus (but capped)
            reward += min(self.robot_possession_time * 0.01, 0.2)
        
        # Bigger reward for scoring but not too big
        if self._check_goal():
            reward += 15.0

        # Penalties
        # Collision (moderate penalty)
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < self.collision_distance:
            reward -= 1.0
            
        # Corner camping prevention
        if (self.robot_pos[0] < 60 or self.robot_pos[0] > 340 or self.robot_pos[1] < 60 or self.robot_pos[1] > 340):
            reward -= 0.5
        
        # Opponent possession penalty
        if self.opponent_has_ball:
            reward -= 0.05
            reward -= min(self.opponent_possession_time * 0.001, 0.1)

        # Clip rewards to prevent explosions
        reward = np.clip(reward, -3.0, 20.0)
        
        # Safety check
        if not np.isfinite(reward):
            reward = 0.0
            
        return float(reward)
    
        # # Reward for having ball
        # if self.has_ball:
        #     reward += 10.0 # Used to be 0.5 increments then 5.0
            
        #     # Progressive reward for keeping ball longer
        #     reward += min(self.robot_possession_time * 0.01, 0.05)

        #     # Reward for moving toward goal with ball
        #     goal_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        #     reward += 3.0 * (400 - goal_distance) / 400 # (Used to be 0.5 multiplier then 2.0 * multiplier)

        #     # Big bonus for being close to goal with ball
        #     if goal_distance < 100:
        #         reward += 10.0 # Used to be incremented by 2.0 then 5.0
        # else:
        #     # Reward for being close to ball when not having it
        #     ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        #     if ball_distance < 50:
        #         reward += 2.0  # Reward for approaching ball used to be incremented by 0.1
        #     if ball_distance < 30:
        #         reward += 5.0 # Reward for getting even closer to the ball
        #     if ball_distance > 150:
        #         reward -= 2.0 # Penalty for staying away from the ball (to avoid corner camping to minimise -ve rewards)
        
        # # Penalty for being in corners (to prevent corner camping)
        # # Check if robot is near field edges
        # if (self.robot_pos[0] < 50 or self.robot_pos[0] > 350 or 
        #     self.robot_pos[1] < 50 or self.robot_pos[1] > 350):
        #     reward -= 1.0  # Penalty for corner camping

        # if self.opponent_has_ball:
        #     reward -= 0.02  # Penalty for losing possession to opponent
        #     # Additional penalty if opponent has ball for long time
        #     reward -= min(self.opponent_possession_time * 0.001, 0.03)

        # # Check if the robot scored, then give a huge reward
        # if (self.ball_pos[0] > 340 and 160 < self.ball_pos[1] < 240):
        #     reward += 100.0
        
        # # Penalty for collision with opponent
        # if np.linalg.norm(self.robot_pos - self.opponent_pos) < 30:
        #     reward -= 1.0 # Used to be 5.0
        
        # # Small penalty for being too far from action
        # ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        # if ball_distance > 150:  # If robot is very far from ball
        #     reward -= 0.01

        # return reward
    
    def _check_goal(self):
        """Check if ball is in goal"""
        return (self.ball_pos[0] > 340 and 160 < self.ball_pos[1] < 240)
    
    def _check_terminated(self):
        # Success: ball in goal
        if self._check_goal():
            return True
        
        # Failure: collision with opponent
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < 25:
            return True

        # Failure: opponent dominates (based on difficulty)
        max_opponent_time = 50 if self.difficulty == "easy" else 30
        if self.opponent_possession_time > max_opponent_time:
            return True
        
        # # Failure: end if opponent controls ball for too long (30 steps)
        # if self.opponent_possession_time > 30:
        #     return True
        
        return False
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.field_width, self.field_height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption(f"Soccer RL - {self.difficulty.title()} Mode")
        
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
        
        # Display info
        font = pygame.font.Font(None, 36)
        info_text = f"Step: {self.steps}/{self.max_steps}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        self.window.blit(text_surface, (10, 10))
        
        if self.has_ball:
            ball_text = font.render("ROBOT HAS BALL!", True, (0, 255, 0))
            self.window.blit(ball_text, (10, 50))

            # More precise positioning feedback
            robot_x, robot_y = self.robot_pos
            optimal_shooting_zone = (robot_x > 290 and robot_x < 340 and 
                                   robot_y > 170 and robot_y < 230)
            goalpost_camping = ((robot_x > 320 and robot_y < 160) or 
                               (robot_x > 320 and robot_y > 240))
            
            if optimal_shooting_zone:
                optimal_text = font.render("OPTIMAL SHOOTING POSITION!", True, (0, 255, 0))
                self.window.blit(optimal_text, (10, 130))
            elif goalpost_camping:
                camping_text = font.render("GOALPOST CAMPING - MOVE TO CENTER!", True, (255, 0, 0))
                self.window.blit(camping_text, (10, 130))
            elif robot_x > 300:
                approach_text = font.render("NEAR GOAL - AIM FOR CENTER!", True, (255, 255, 0))
                self.window.blit(approach_text, (10, 130))

        elif self.opponent_has_ball:
            opp_text = font.render("OPPONENT HAS BALL", True, (255, 0, 0))
            self.window.blit(opp_text, (10, 50))
        
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
    env = SoccerEnv(render_mode="human")
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