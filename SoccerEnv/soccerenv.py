"""
soccerenv.py - Basic 2D Soccer Environment for FYP
This file contains the SoccerEnv class which implements a simple 2D soccer environment
where a robot needs to dribble a ball around an opponent to score in a goal.
The environment supports different difficulty levels and provides visual rendering using Pygame.

Author: Ali Riyaz
Student Number: C3412624
Last Updated: 17/08/2025
"""

#TODO: If I do end up using this, remember to fix the comments

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from .fieldconfig import FieldConfig

class SoccerEnv(gym.Env):
    """
    Simple 2D Soccer Environment for FYP
    - Robot needs to dribble ball around opponent to goal
    - Positions of the ball and 2 robots randomised at the start of a new episode
    - 
    """
    
    # Constructor to initialise the environment including action, observation spaces along with state variables
    def __init__(self, render_mode=None, difficulty="easy", config_path="field_config.yaml", testing_mode=False):
        super().__init__()

        # Load field configuration
        self.field_config = FieldConfig(config_path)
        self.difficulty = difficulty
        self.testing_mode = testing_mode
        
        # # Field dimensions (4x4 meters as per field size requirements)
        # self.field_width = 400  # pixels
        # self.field_height = 400  # pixels
        # self.goal_width = 80

        # Set up field dimensions from config
        self.field_width = self.field_config.field_width_pixels # X-axis
        self.field_height = self.field_config.field_height_pixels # Y-axis
        self.goal_width = self.field_config.goal_width_pixels
        
        # Action space: [forward/back, left/right, rotation]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: [robot_x, robot_y, robot_angle, ball_x, ball_y, 
        #                    opponent_x, opponent_y, robot_vx, robot_vy, 
        #                    ball_distance, goal_distance, has_ball]
        # TODO: MAYBE?? Remove the has_ball boolean from the observation space
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # # Ball physics parameters
        # self.ball_mass = 0.5          # Ball mass for physics calculations
        # self.ball_friction = 0.8     # Ball friction (0-1, lower = more friction) was 0.92
        # self.ball_bounce = 0.2        # Wall bounce dampening (was 0.3)
        # self.ball_max_speed = 3.0     # Maximum ball speed (OG was 8.0 then 0.5)

        # Load physics parameters from config
        physics = self.field_config.config['physics']
        self.ball_mass = physics['ball_mass']
        self.ball_friction = physics['ball_friction']
        self.ball_bounce = physics['ball_bounce']
        self.ball_max_speed = physics['ball_max_speed']
        self.push_force_multiplier = physics['push_force_multiplier']

        # Define timestep
        self.target_fps = 60  # TODO: Change according to simulation speed
        self.dt = 1.0 / self.target_fps

        # Load robot parameters from config
        robot_params = self.field_config.config['robot_parameters']
        self.robot_radius = self.field_config.meters_to_pixels(robot_params['robot_radius'])
        self.ball_radius = self.field_config.meters_to_pixels(robot_params['ball_radius'])
        self.possession_threshold = self.field_config.meters_to_pixels(robot_params['possession_threshold'])
        
        # # Robot-ball interaction parameters
        # self.robot_radius = 15        # Robot collision radius
        # self.ball_radius = 8          # Ball radius
        # self.push_force_multiplier = 1.2  # How strong robot pushes ball (was 2.0)
        # self.possession_threshold = 25    # Distance to consider "close to ball"
        
        # # Game parameters
        # TODO: See if we need these in some way or to change them and then use it
        # self.possession_distance = 40.0 # Used to be 35.0
        # self.collision_distance = 30.0

        # Load difficulty-specific parameters from config
        diff_settings = self.field_config.config['difficulty_settings'][difficulty]
        self.possession_distance = self.field_config.meters_to_pixels(diff_settings['possession_distance'])
        self.collision_distance = self.field_config.meters_to_pixels(diff_settings['collision_distance'])
        self.max_steps = diff_settings['max_steps']
        self.opponent_speed = diff_settings['opponent_speed']

        # Initialise state variables first
        self.robot_pos = None
        self.robot_angle = None
        self.robot_vel = None
        self.opponent_vel = None  # Track opponent velocity
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

        # Robot movement parameters - adjusted for testing mode
        base_robot_speed = 1.8 # Used tp be 2.5 (MIGHT BE IN PIXELS per FRAME not m/s)
        base_rotation_speed = 0.08 # Used to be 0.1

        # Cache frequently used calculations
        self._max_distance = np.sqrt(self.field_width**2 + self.field_height**2)
        self._goal_center = np.array([self.field_width, self.field_height // 2])
        self._field_center = np.array([self.field_width // 2, self.field_height // 2])
        
        # Apply testing mode speed multiplier
        if self.testing_mode or self.field_config.config.get('rendering', {}).get('testing_mode', False):
            speed_multiplier = self.field_config.config.get('rendering', {}).get('testing_speed_multiplier', 2.0)
            self.robot_speed = base_robot_speed * speed_multiplier
            self.robot_rotation_speed = base_rotation_speed * speed_multiplier
            print(f"🚀 Testing mode enabled: {speed_multiplier}x speed")
        else:
            self.robot_speed = base_robot_speed
            self.robot_rotation_speed = base_rotation_speed

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Initialize ball with its own velocity
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # # Randomize robot starting position (left third of field)
        # self.robot_pos = np.array([
        #     np.random.uniform(self.robot_radius + 20, self.field_width * 0.33),
        #     np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        # ])
        self.robot_angle = np.random.uniform(-np.pi/4, np.pi/4)  # Roughly facing right
        self.robot_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.opponent_vel = np.array([0.0, 0.0], dtype=np.float64)  # Reset opponent velocity
        
        # # Randomize ball position (center third of field, avoiding edges)
        # self.ball_pos = np.array([
        #     np.random.uniform(self.field_width * 0.3, self.field_width * 0.7),
        #     np.random.uniform(self.ball_radius + 30, self.field_height - self.ball_radius - 30)
        # ])

        # Randomize robot starting position
        self.robot_pos = np.array([
            np.random.uniform(self.robot_radius + 20, self.field_width * 0.4),
            np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        ], dtype=np.float64)

        # Place ball very close to robot (within contact distance + small margin)
        contact_distance = self.robot_radius + self.ball_radius
        ball_distance = contact_distance + np.random.uniform(5, 30)  # Just outside contact range
        ball_angle = np.random.uniform(0, 2 * np.pi)

        self.ball_pos = self.robot_pos + ball_distance * np.array([
            np.cos(ball_angle), 
            np.sin(ball_angle)
        ], dtype=np.float64)

        # Clamp to field boundaries
        self.ball_pos[0] = np.clip(self.ball_pos[0], self.ball_radius + 10, self.field_width - self.ball_radius - 10)
        self.ball_pos[1] = np.clip(self.ball_pos[1], self.ball_radius + 10, self.field_height - self.ball_radius - 10)
        
        # Randomize opponent position (right third of field)
        self.opponent_pos = np.array([
            np.random.uniform(self.field_width * 0.6, self.field_width - self.robot_radius - 20),
            np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        ], dtype=np.float64)
        
        # Add small random initial ball velocity for realism
        if np.random.random() < 0.3:  # 30% chance of initial movement
            self.ball_vel = np.random.uniform(-0.5, 0.5, 2)

        # Goal position (right side, center)
        self.goal_pos = np.array([self.field_width, self.field_height // 2])
        
        # Ensure minimum distances to avoid starting in collision
        while np.linalg.norm(self.robot_pos - self.opponent_pos) < 50:
            self.opponent_pos = np.array([
                np.random.uniform(self.field_width * 0.6, self.field_width - self.robot_radius - 20),
                np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
            ])

        # Reset game state
        self.has_ball = False
        self.opponent_has_ball = False
        self.steps = 0
        self.robot_possession_time = 0
        self.opponent_possession_time = 0

        # Reset opponent behavior
        self.opponent_behavior = np.random.choice(['aggressive', 'defensive', 'balanced'])
        self.opponent_target = None

        # TODO: Commented this out since Added testing mode to speed things up for the rendering during testing models at least when testing
        # # === ROBOT MOVEMENT PARAMETERS ===
        # self.robot_speed = 1.8          # # Base speed of the robot (was 2.5).SLOWER robot
        # self.robot_rotation_speed = 0.08  # Rotation speed multiplier (was 0.1) SLOWER rotation
        
        return self._get_obs(), {}
    
    # Function to 
    def _get_obs(self) -> np.array:
        # Normalize positions to [-1, 1] range
        norm_robot_x = np.clip((2 * self.robot_pos[0] / self.field_width) - 1, -1, 1)
        norm_robot_y = np.clip((2 * self.robot_pos[1] / self.field_height) - 1, -1, 1)
        norm_ball_x = np.clip((2 * self.ball_pos[0] / self.field_width) - 1, -1, 1)
        norm_ball_y = np.clip((2 * self.ball_pos[1] / self.field_height) - 1, -1, 1)
        norm_opponent_x = np.clip((2 * self.opponent_pos[0] / self.field_width) - 1, -1, 1)
        norm_opponent_y = np.clip((2 * self.opponent_pos[1] / self.field_height) - 1, -1, 1)
        
        # Normalize velocities
        max_vel = 100.0  # Assuming max velocity is 100 pixels/s
        norm_robot_vx = np.clip(self.robot_vel[0] / max_vel, -1, 1)
        norm_robot_vy = np.clip(self.robot_vel[1] / max_vel, -1, 1)

        # Calculate distances
        max_distance = self._max_distance  # Maximum distance in the field

        # Normalize distances to [0, 1]
        ball_distance = np.clip(np.linalg.norm(self.robot_pos - self.ball_pos) / max_distance, 0, 1)
        goal_distance = np.clip(np.linalg.norm(self.robot_pos - self.goal_pos) / max_distance, 0, 1)
    
        # Normalize angle
        norm_angle = np.clip(self.robot_angle / np.pi, -1, 1)
        
        obs = np.array([
            norm_robot_x, norm_robot_y, norm_angle,
            norm_ball_x, norm_ball_y,
            norm_opponent_x, norm_opponent_y,
            norm_robot_vx, norm_robot_vy,
            ball_distance, goal_distance,
            float(self.has_ball),  # Convert boolean to float
        ], dtype=np.float32)
    
        # Ensure no NaN/inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
    
    # To take an action
    def step(self, action):
        self.steps += 1

        # Apply robot action in the environment
        self._apply_robot_action(action)

        # Update ball movements (position, velocity, etc.) accordingly
        self._update_ball_physics()

        # Opponent AI's behavior (can be offensive, defensive, or balanced)
        self._update_opponent()
        
        # Update possession flags based on proximity (transition period)
        self._update_possession_flags()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_terminated()
        truncated = self.steps >= self.max_steps

        # Check if ball went out of bounds (simple implementation)
        ball_out_of_bounds = self._check_ball_out_of_play()
        
        # Check specific termination reasons for logging
        if terminated and not truncated:
            if self._check_goal():
                print("🎉 ROBOT SCORED - EPISODE WON!")
            elif self._check_opponent_goal():
                print("😭 OPPONENT SCORED - EPISODE LOST!")
            elif self._check_ball_out_of_play():
                print("❌ BALL OUT OF BOUNDS - EPISODE TERMINATED!")

        if ball_out_of_bounds and not terminated:
            # Reset ball to center if it goes out of bounds
            self.ball_pos = np.array([
                self.field_width * 0.5,
                self.field_height * 0.5
            ])
            self.ball_vel = np.array([0.0, 0.0])
            reward += self.field_config.config.get('reward_parameters', {}).get('out_of_bounds_penalty', -2.0)
            terminated = True  # End episode if ball goes out of bounds
            print("❌ BALL OUT OF BOUNDS - EPISODE TERMINATED!")
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _apply_robot_action(self, action):
        # Validate and clip action values
        action = np.clip(action, -1.0, 1.0)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # Parse action
        forward_back = action[0]  # -1 to 1
        left_right = action[1]    # -1 to 1  
        rotation = action[2]      # -1 to 1
        
        # Update robot angle
        self.robot_angle += rotation * self.robot_rotation_speed
        self.robot_angle = self.robot_angle % (2 * np.pi) # Normalize angle to [0, 2π)
        
        # Calculate movement
        speed = self.robot_speed
        
        # Forward/backward movement (in robot's facing direction)
        forward_vel = speed * forward_back
        forward_x = forward_vel * np.cos(self.robot_angle)
        forward_y = forward_vel * np.sin(self.robot_angle)
        
        # Left/right movement (perpendicular to robot's facing direction)
        strafe_vel = speed * left_right
        strafe_x = strafe_vel * np.cos(self.robot_angle + np.pi/2)
        strafe_y = strafe_vel * np.sin(self.robot_angle + np.pi/2)
        
        # Combine movements
        self.robot_vel = np.array([forward_x + strafe_x, forward_y + strafe_y])
        
        # Update position
        new_pos = self.robot_pos + self.robot_vel * self.dt # Based on time step and fps rate

        # Keep robot within bounds
        new_pos[0] = np.clip(new_pos[0], self.robot_radius, self.field_width - self.robot_radius)
        new_pos[1] = np.clip(new_pos[1], self.robot_radius, self.field_height - self.robot_radius)
        
        self.robot_pos = new_pos
    
    #TODO: Deal with opponent's velocity as well in the init method 
    def _update_opponent(self):
        """Enhanced opponent AI with different behaviors"""
        
        if not hasattr(self, 'opponent_behavior'):
            self.opponent_behavior = 'balanced'
        
        # Store previous position to calculate velocity
        prev_pos = self.opponent_pos.copy()

        robot_ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        opponent_ball_distance = np.linalg.norm(self.opponent_pos - self.ball_pos)
        robot_opponent_distance = np.linalg.norm(self.robot_pos - self.opponent_pos)
        
        # Different opponent behaviors
        if self.opponent_behavior == 'aggressive':
            # Always chase the ball or try to intercept robot
            if self.has_ball and robot_opponent_distance > 50:
                # Try to intercept robot's path to goal
                intercept_point = self.robot_pos + (self.goal_pos - self.robot_pos) * 0.3
                direction = intercept_point - self.opponent_pos
            elif opponent_ball_distance < robot_ball_distance:
                # Go for ball if closer
                direction = self.ball_pos - self.opponent_pos
            else:
                # Chase robot aggressively
                direction = self.robot_pos - self.opponent_pos
                
        elif self.opponent_behavior == 'defensive':
            # Stay between ball/robot and goal, more conservative
            goal_center = self._goal_center
            
            if self.has_ball:
                # Position between robot and goal
                defensive_pos = self.robot_pos + (goal_center - self.robot_pos) * 0.4
                defensive_pos[0] = min(defensive_pos[0], self.field_width - 60)  # Don't go too close to goal
                direction = defensive_pos - self.opponent_pos
            else:
                # Position between ball and goal
                defensive_pos = self.ball_pos + (goal_center - self.ball_pos) * 0.3
                defensive_pos[0] = min(defensive_pos[0], self.field_width - 60)
                direction = defensive_pos - self.opponent_pos
                
        else:  # balanced behavior
            # Mix of aggressive and defensive based on situation
            if self.opponent_has_ball:
                # Move towards goal when has ball
                direction = self.goal_pos - self.opponent_pos
            elif opponent_ball_distance < robot_ball_distance * 0.8:
                # Go for ball if significantly closer
                direction = self.ball_pos - self.opponent_pos
            elif self.has_ball and robot_opponent_distance < 80:
                # Try to tackle if robot has ball and is close
                direction = self.robot_pos - self.opponent_pos
            else:
                # Default to ball position
                direction = self.ball_pos - self.opponent_pos
        
        # Normalize direction and apply movement
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
            
        # Apply movement with opponent speed from config (with testing mode)
        # base_speed = 20.0 * self.opponent_speed #TODO: Check to remove this multiplier

        base_opponent_speed = 1.8 # Base speed for opponent in PIXELS per frame
        speed = base_opponent_speed * self.opponent_speed # Here opponent_speed is a multiplier from the config
        if self.testing_mode or self.field_config.config.get('rendering', {}).get('testing_mode', False):
            speed_multiplier = self.field_config.config.get('rendering', {}).get('testing_speed_multiplier', 2.0)
            speed = speed * speed_multiplier          

        # Add some randomness to make opponent less predictable
        if np.random.random() < 0.1:  # 10% chance for random movement
            random_angle = np.random.uniform(0, 2 * np.pi)
            direction += 0.3 * np.array([np.cos(random_angle), np.sin(random_angle)])
            direction = direction / np.linalg.norm(direction)
        
        # Calculate new position
        new_pos = self.opponent_pos + direction * speed * self.dt
        
        # Keep opponent within bounds with margin
        margin = self.robot_radius + 10
        new_pos[0] = np.clip(new_pos[0], margin, self.field_width - margin)
        new_pos[1] = np.clip(new_pos[1], margin, self.field_height - margin)
        
        # Calculate velocity based on position change
        self.opponent_vel = (new_pos - self.opponent_pos) / self.dt
        
        # Update position
        self.opponent_pos = new_pos

    # Function to calculate rewards as an episode progresses during training   
    def _calculate_reward(self) -> float:
        reward = 0.0

        # Load parameters from config
        reward_params = self.field_config.config.get('reward_parameters', {})
        robot_params = self.field_config.config.get('robot_parameters', {})
        game_thresholds = self.field_config.config.get('game_thresholds', {})
        strategic_zones = self.field_config.config.get('strategic_zones', {})
        
        # Time penalty
        reward += reward_params['time_penalty']

        # Calculate key distances  using configurable field dimensions
        robot_x, robot_y = self.robot_pos
        ball_x, ball_y = self.ball_pos
        robot_ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        opponent_ball_distance = np.linalg.norm(self.opponent_pos - self.ball_pos)
        goal_center = self._goal_center # Goal distance that considers BOTH X and Y coordinates (adapt to field size)
        ball_to_goal_distance = np.linalg.norm(self.ball_pos - goal_center)
        

        # Convert thresholds to pixels
        contact_threshold = self.field_config.meters_to_pixels(robot_params.get('contact_threshold', 0.25))
        close_threshold = self.field_config.meters_to_pixels(robot_params.get('close_threshold', 0.4))
        medium_threshold = self.field_config.meters_to_pixels(robot_params.get('medium_threshold', 0.8))
        far_threshold = self.field_config.meters_to_pixels(robot_params.get('far_threshold', 1.5))
        
        #TODO: THIS IS NOT USED Check if robot is actually in front of goal (scaled to field size)
        goal_area_x_min = self.field_width * 0.75  # 75% of field width
        goal_area_y_min = self.field_height * 0.4   # 40% of field height
        goal_area_y_max = self.field_height * 0.6   # 60% of field height
        goal_aligned = (robot_x > goal_area_x_min and goal_area_y_min < robot_y < goal_area_y_max)
        
        # PRIORITY 1: Major outcomes (also since the episode should end with either of these 3 outcomes)
        # HUGE reward for scoring
        if self._check_goal():
            # reward += reward_params['goal_scored']
            print("🎉 GOAL SCORED!")
            return reward_params.get('goal_scored', 100.0)

        # HUGE penalty for opponent scoring
        if self._check_opponent_goal():
            # reward += -reward_params['goal_scored']  # Negative of goal reward
            print("😭 OPPONENT SCORED!")
            return -reward_params.get('goal_scored', 100.0) # Negative of goal reward
            
        if self._check_ball_out_of_play():
            return reward_params.get('out_of_bounds_penalty', -10.0)

        # PRIORITY 2: Ball acquisition
        if robot_ball_distance < contact_threshold:
            reward += reward_params.get('contact_reward', 15.0)
        elif robot_ball_distance < close_threshold:
            reward += reward_params.get('close_reward', 8.0)
        elif robot_ball_distance < medium_threshold:
            reward += reward_params.get('medium_reward', 4.0)
        elif robot_ball_distance < far_threshold:
            reward += reward_params.get('far_reward', 1.0)
        else:
            reward += reward_params.get('too_far_penalty', -2.0)

        max_distance = self._max_distance  # Maximum distance in the field
        ball_proximity_reward = (max_distance - robot_ball_distance) / max_distance # Idea to maximise this reward as the robot is close to the ball
        reward += ball_proximity_reward * reward_params.get('proximity_multiplier', 10.0)  # Strong ball proximity reward

        # PHASE 2: Reward for moving towards ball (velocity-based)
        robot_speed = np.linalg.norm(self.robot_vel)
        if robot_ball_distance > close_threshold and robot_speed > 0.1:  # Only if moving fast enough
            # Calculate if robot is moving towards ball
            ball_direction = (self.ball_pos - self.robot_pos)
            if np.linalg.norm(ball_direction) > 0:
                ball_direction_norm = ball_direction / np.linalg.norm(ball_direction)
                robot_velocity_norm = self.robot_vel / robot_speed
                
                # Reward for moving towards ball
                velocity_alignment = np.dot(robot_velocity_norm, ball_direction_norm)
                if velocity_alignment > 0:
                    reward += velocity_alignment * robot_speed * 0.9  # Reward moving towards ball

        # PHASE 3: Reward for pushing ball toward goal
        
        
        # Only reward ball progress when robot is close to ball
        if robot_ball_distance < medium_threshold:
            max_goal_distance = np.linalg.norm([0, self.field_height//2] - goal_center)
            goal_progress = (max_goal_distance - ball_to_goal_distance) / max_goal_distance
            reward += goal_progress * 4.0
            
            # Extra reward if robot is "shepherding" ball toward goal
            shepherding_threshold = self.field_config.meters_to_pixels(0.35)  # 35cm
            if robot_ball_distance < shepherding_threshold:
                robot_ball_goal_angle = self._calculate_shepherding_angle()
                if robot_ball_goal_angle < 30:  # Very good angle
                    reward += 6.0
                elif robot_ball_goal_angle < 60:  # Good angle
                    reward += 4.0
                elif robot_ball_goal_angle < 90:  # Okay angle
                    reward += 2.0

        # PHASE 4: Reward for ball speed toward goal (good pushing)
        ball_speed = np.linalg.norm(self.ball_vel)
        if ball_speed > 0.1 and robot_ball_distance < medium_threshold:  # Only when close to ball
            ball_to_goal = goal_center - self.ball_pos
            if np.linalg.norm(ball_to_goal) > 1e-6:
                ball_to_goal_normalized = ball_to_goal / np.linalg.norm(ball_to_goal)
                ball_vel_normalized = self.ball_vel / ball_speed
                
                # Dot product: 1 = moving directly toward goal, -1 = away from goal
                goal_direction_alignment = np.dot(ball_vel_normalized, ball_to_goal_normalized)
                if goal_direction_alignment > 0:
                    reward += goal_direction_alignment * min(ball_speed, 2.0) * 1.5
        
        

        # === ANTI-EXPLOIT PENALTIES ===
        # 4A: STRONG penalties for edge/corner behavior (scaled to field size)
        edge_threshold = min(self.field_width, self.field_height) * 0.05  # Reduced from 0.1 to 0.05
        corner_threshold = min(self.field_width, self.field_height) * 0.08  # Reduced from 0.15 to 0.08

        # Robot near edges
        if (robot_x < edge_threshold or robot_x > (self.field_width - edge_threshold) or 
            robot_y < edge_threshold or robot_y > (self.field_height - edge_threshold)):
            reward += reward_params['edge_penalty']  # Strong penalty for robot camping edges
        
        # Ball near edges (prevents pushing ball to edges)
        if (ball_x < edge_threshold or ball_x > (self.field_width - edge_threshold) or 
            ball_y < edge_threshold or ball_y > (self.field_height - edge_threshold)):

            if self._check_ball_out_of_play():
                reward -= 2.0  # Major penalty for losing ball
        
        # 4C: Reward for ball being in center field (encourage proper play)
        field_center = np.array([self.field_width // 2, self.field_height // 2])
        ball_to_center_distance = np.linalg.norm(self.ball_pos - field_center)
        center_zone_radius = min(self.field_width, self.field_height) * 0.25  # 25% of field size
        if ball_to_center_distance < center_zone_radius:  # Ball near center
            reward += reward_params['center_field_bonus']

        # Collision (moderate penalty)
        robot_opponent_distance = np.linalg.norm(self.robot_pos - self.opponent_pos)
        if robot_opponent_distance < self.collision_distance:
            reward += reward_params['collision_penalty']
            
        # Corner camping prevention (scaled to field)
        # corner_threshold = min(self.field_width, self.field_height) * 0.15  # 15% of field size
        if (robot_x < corner_threshold or robot_x > (self.field_width - corner_threshold) or 
            robot_y < corner_threshold or robot_y > (self.field_height - corner_threshold)):
            reward += reward_params['corner_penalty']

        # Extra reward for getting ball into dangerous areas (near goal) - scaled to field
        dangerous_area_x = self.field_width * 0.75  # 75% of field width
        dangerous_area_y_min = self.field_height * 0.375  # 37.5% of field height
        dangerous_area_y_max = self.field_height * 0.625  # 62.5% of field height
        
        if ball_x > dangerous_area_x and dangerous_area_y_min < ball_y < dangerous_area_y_max:
            reward += 3.0  # Ball in scoring area
            
            # Even more if robot is still close (maintaining control)
            if robot_ball_distance < self.field_config.meters_to_pixels(0.4):  # Within 40cm
                reward += 5.0 # (USED TO BE 3.0)

        # Penalty if opponent is closer to ball (competitive element)
        if opponent_ball_distance < robot_ball_distance:
            reward -= 0.05

        # Clip rewards to prevent explosions
        reward = np.clip(reward, -2.0, 30.0)
        
        # Safety check
        if not np.isfinite(reward):
            reward = 0.0
            
        return float(reward)

    def _calculate_shepherding_angle(self) -> float:
        """Calculate the angle between robot-ball vector and ball-goal vector"""
        try:
            # Vector from robot to ball
            robot_to_ball = self.ball_pos - self.robot_pos
            
            # Vector from ball to goal (Goal on the Right side of the field)
            goal_center = self._goal_center
            ball_to_goal = goal_center - self.ball_pos
            
            # Normalize vectors
            if np.linalg.norm(robot_to_ball) < 1e-6 or np.linalg.norm(ball_to_goal) < 1e-6:
                return 90.0  # Default angle if vectors are too small
                
            robot_to_ball_norm = robot_to_ball / np.linalg.norm(robot_to_ball)
            ball_to_goal_norm = ball_to_goal / np.linalg.norm(ball_to_goal)
            
            # Calculate angle between vectors
            dot_product = np.dot(robot_to_ball_norm, ball_to_goal_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range for arccos
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            # If any calculation fails, return a neutral angle
            return 90.0
    
    # TODO: Test whether the new method makes sure the ball doesn't phase/pass through the goalposts like maybe either 2 posts of the goal
    def _update_ball_physics(self):
        """Realistic ball physics with pushing, friction, and bouncing"""
        
        # ROBOT-BALL INTERACTION (pushing)
        robot_to_ball = self.ball_pos - self.robot_pos
        robot_ball_distance = np.linalg.norm(robot_to_ball)
        
        # Check if robot is touching/pushing the ball
        contact_distance = self.robot_radius + self.ball_radius
        if robot_ball_distance < contact_distance and robot_ball_distance > 0.1:
            
            # if robot_ball_distance > 1e-6: # Avoid Division by zero
            #     push_direction = robot_to_ball / robot_ball_distance


            # Calculate push force based on robot velocity and distance
            overlap = contact_distance - robot_ball_distance
            
            # Direction of push (from robot center to ball center)
            push_direction = robot_to_ball / robot_ball_distance
            
            # Push force depends on robot velocity and overlap
            robot_speed = np.linalg.norm(self.robot_vel)
            push_force = (robot_speed * 0.3 + overlap * 0.2) * self.push_force_multiplier
            
            # Apply push to ball velocity
            push_velocity = push_direction * push_force
            self.ball_vel += push_velocity
            
            # Add some robot direction influence (robot can "guide" the ball)
            if robot_speed > 0.1:  # Only if robot is moving
                robot_direction = self.robot_vel / robot_speed
                guidance_strength = min(robot_speed * 0.15, 0.8)  # Limit guidance strength
                self.ball_vel += robot_direction * guidance_strength
        
        # OPPONENT-BALL INTERACTION 
        opponent_to_ball = self.ball_pos - self.opponent_pos
        opponent_ball_distance = np.linalg.norm(opponent_to_ball)
        
        if opponent_ball_distance < contact_distance and opponent_ball_distance > 0.1:
            overlap = contact_distance - opponent_ball_distance
            push_direction = opponent_to_ball / opponent_ball_distance
            
            # Opponent push force
            opponent_speed = np.linalg.norm(self.opponent_vel)  # Assume constant opponent "speed"
            push_force = (opponent_speed * 0.4 + overlap * 0.2) * self.push_force_multiplier
            
            push_velocity = push_direction * push_force
            self.ball_vel += push_velocity

            # Add opponent direction influence (similar to robot)
            if opponent_speed > 0.1:  # Only if opponent is moving
                opponent_direction = self.opponent_vel / opponent_speed
                guidance_strength = min(opponent_speed * 0.1, 0.5)  # Slightly weaker than robot
                self.ball_vel += opponent_direction * guidance_strength
            
        # Apply friction to ball
        self.ball_vel *= self.ball_friction
        
        # Add small random perturbations (realistic ball imperfections)
        if np.random.random() < 0.01:  # 1% chance per step
            random_perturbation = np.random.uniform(-0.1, 0.1, 2)
            self.ball_vel += random_perturbation
        
        # Limit ball speed
        ball_speed = np.linalg.norm(self.ball_vel)
        max_speed_pixels = self.field_config.meters_to_pixels(self.ball_max_speed)
        if ball_speed > max_speed_pixels:
            self.ball_vel = (self.ball_vel / ball_speed) * max_speed_pixels
        
        # Update ball position
        self.ball_pos = self.ball_pos + self.ball_vel * self.dt
        
        # # Handle wall bounces
        # self._handle_ball_wall_bounces()

        # Handle goalpost collisions
        self._handle_goalpost_collision()
        
        # Stop very slow ball (prevent infinite tiny movements)
        if ball_speed < 0.08:
            self.ball_vel *= 0.3  # Quickly dampen very slow movement
    
    def _handle_goalpost_collision(self):
        """Handle ball collision with goalposts at both ends"""
        
        # Get goal parameters
        goal_center_y = self.field_height // 2
        goal_half_width = self.goal_width // 2
        
        # --- RIGHT GOAL (Target goal) ---
        right_goal_x = self.field_width
        top_post_y = goal_center_y - goal_half_width
        bottom_post_y = goal_center_y + goal_half_width
        
        # Check collision with right goal top post
        if (abs(self.ball_pos[0] - right_goal_x) <= self.ball_radius and
            abs(self.ball_pos[1] - top_post_y) <= self.ball_radius):
            
            # Determine collision side and bounce accordingly
            if self.ball_pos[1] < top_post_y:  # Hit from above
                self.ball_pos[1] = top_post_y - self.ball_radius
                self.ball_vel[1] = -abs(self.ball_vel[1]) * self.ball_bounce
            else:  # Hit from side
                self.ball_pos[0] = right_goal_x - self.ball_radius
                self.ball_vel[0] = -abs(self.ball_vel[0]) * self.ball_bounce
            
            # Add small random component to prevent getting stuck
            self.ball_vel += np.random.uniform(-0.1, 0.1, 2)
        
        # Check collision with right goal bottom post
        if (abs(self.ball_pos[0] - right_goal_x) <= self.ball_radius and
            abs(self.ball_pos[1] - bottom_post_y) <= self.ball_radius):
            
            # Determine collision side and bounce accordingly
            if self.ball_pos[1] > bottom_post_y:  # Hit from below
                self.ball_pos[1] = bottom_post_y + self.ball_radius
                self.ball_vel[1] = abs(self.ball_vel[1]) * self.ball_bounce
            else:  # Hit from side
                self.ball_pos[0] = right_goal_x - self.ball_radius
                self.ball_vel[0] = -abs(self.ball_vel[0]) * self.ball_bounce
            
            # Add small random component to prevent getting stuck
            self.ball_vel += np.random.uniform(-0.1, 0.1, 2)
        
        # --- LEFT GOAL (Opponent goal) ---
        left_goal_x = 0
        
        # Check collision with left goal top post
        if (abs(self.ball_pos[0] - left_goal_x) <= self.ball_radius and
            abs(self.ball_pos[1] - top_post_y) <= self.ball_radius):
            
            # Determine collision side and bounce accordingly
            if self.ball_pos[1] < top_post_y:  # Hit from above
                self.ball_pos[1] = top_post_y - self.ball_radius
                self.ball_vel[1] = -abs(self.ball_vel[1]) * self.ball_bounce
            else:  # Hit from side
                self.ball_pos[0] = left_goal_x + self.ball_radius
                self.ball_vel[0] = abs(self.ball_vel[0]) * self.ball_bounce
            
            # Add small random component to prevent getting stuck
            self.ball_vel += np.random.uniform(-0.1, 0.1, 2)
        
        # Check collision with left goal bottom post
        if (abs(self.ball_pos[0] - left_goal_x) <= self.ball_radius and
            abs(self.ball_pos[1] - bottom_post_y) <= self.ball_radius):
            
            # Determine collision side and bounce accordingly
            if self.ball_pos[1] > bottom_post_y:  # Hit from below
                self.ball_pos[1] = bottom_post_y + self.ball_radius
                self.ball_vel[1] = abs(self.ball_vel[1]) * self.ball_bounce
            else:  # Hit from side
                self.ball_pos[0] = left_goal_x + self.ball_radius
                self.ball_vel[0] = abs(self.ball_vel[0]) * self.ball_bounce
            
            # Add small random component to prevent getting stuck
            self.ball_vel += np.random.uniform(-0.1, 0.1, 2)
            
    def _handle_ball_wall_bounces(self):
        """Handle ball bouncing off walls with dampening"""
        
        # Left and right walls
        if self.ball_pos[0] <= self.ball_radius:
            self.ball_pos[0] = self.ball_radius
            self.ball_vel[0] = -self.ball_vel[0] * self.ball_bounce
            # Add some randomness to prevent edge exploitation
            self.ball_vel[1] += np.random.uniform(-0.2, 0.2)
        elif self.ball_pos[0] >= self.field_width - self.ball_radius:
            # Don't bounce off right wall if it's in the goal area
            goal_center_y = self.field_height // 2
            goal_half_width = self.goal_width // 2
            if not (goal_center_y - goal_half_width <= self.ball_pos[1] <= goal_center_y + goal_half_width):
                self.ball_pos[0] = self.field_width - self.ball_radius
                self.ball_vel[0] = -self.ball_vel[0] * self.ball_bounce
                # Add randomness to prevent corner camping
                self.ball_vel[1] += np.random.uniform(-0.2, 0.2)
        
        # Top and bottom walls
        if self.ball_pos[1] <= self.ball_radius:
            self.ball_pos[1] = self.ball_radius
            self.ball_vel[1] = -self.ball_vel[1] * self.ball_bounce
            # Prevent bottom edge exploitation
            self.ball_vel[0] += np.random.uniform(-0.2, 0.2)
        elif self.ball_pos[1] >= self.field_height - self.ball_radius:
            self.ball_pos[1] = self.field_height - self.ball_radius
            self.ball_vel[1] = -self.ball_vel[1] * self.ball_bounce
            # Prevent top edge exploitation  
            self.ball_vel[0] += np.random.uniform(-0.2, 0.2)

    def _update_possession_flags(self):
        """Update possession flags based on proximity (transition method)"""
        
        robot_ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        opponent_ball_distance = np.linalg.norm(self.opponent_pos - self.ball_pos)
        
        # Simple proximity-based possession for now
        if robot_ball_distance < self.possession_threshold:
            if opponent_ball_distance > robot_ball_distance * 1.2:  # Robot is clearly closer
                self.has_ball = True
                self.opponent_has_ball = False
                self.robot_possession_time += 1
                self.opponent_possession_time = 0
        elif opponent_ball_distance < self.possession_threshold:
            if robot_ball_distance > opponent_ball_distance * 1.2:  # Opponent is clearly closer
                self.has_ball = False
                self.opponent_has_ball = True
                self.opponent_possession_time += 1
                self.robot_possession_time = 0
        else:
            # Ball is loose
            self.has_ball = False
            self.opponent_has_ball = False
            # Don't reset possession times immediately - add some "memory"
            if self.robot_possession_time > 0:
                self.robot_possession_time = max(0, self.robot_possession_time - 1)
            if self.opponent_possession_time > 0:
                self.opponent_possession_time = max(0, self.opponent_possession_time - 1)

    def _check_goal(self) -> bool:
        """Check if ball is in goal"""
        return self.field_config.is_ball_in_goal(self.ball_pos[0], self.ball_pos[1])

    def _check_opponent_goal(self) -> bool:
        """Check if ball is in opponent's goal (left goal)"""
        goal_center_y = self.field_height // 2
        goal_half_width = self.goal_width // 2
        
        return (self.ball_pos[0] <= 0 and  # Past left goal line
                goal_center_y - goal_half_width <= self.ball_pos[1] <= goal_center_y + goal_half_width)
    
    def _check_ball_out_of_play(self) -> bool:
        """Check if ball is out of play (beyond field boundaries)"""
        
        # Check top and bottom boundaries (simple - no goals here)
        if (self.ball_pos[1] < -self.ball_radius or 
            self.ball_pos[1] > self.field_height + self.ball_radius):
            return True
        
        # Check left boundary
        if self.ball_pos[0] < -self.ball_radius:
            # Check if ball is in left goal area
            goal_center_y = self.field_height // 2
            goal_half_width = self.goal_width // 2
            
            # Ball is NOT out of play if it's in the goal area
            if (goal_center_y - goal_half_width <= self.ball_pos[1] <= goal_center_y + goal_half_width):
                return False  # Ball is in left goal
            else:
                return True   # Ball is past left boundary but not in goal
        
        # Check right boundary
        if self.ball_pos[0] > self.field_width + self.ball_radius:
            # Use existing method to check if ball is in right goal
            if self.field_config.is_ball_in_goal(self.ball_pos[0], self.ball_pos[1]):
                return False  # Ball is in right goal
            else:
                return True   # Ball is past right boundary but not in goal
        
        # Ball is within all boundaries
        return False
    
    def _check_terminated(self) -> bool:
        """Check if the episode should terminate"""
        # Success: ball in goal
        if self._check_goal():
            return True

        if self._check_opponent_goal():
            return True

        # Ball out of play
        if self._check_ball_out_of_play():
            return True
        
        # Failure: collision with opponent
        if np.linalg.norm(self.robot_pos - self.opponent_pos) < self.collision_distance:
            return True

        # Failure: opponent dominates (based on difficulty)
        max_opponent_time = 50 if self.difficulty == "easy" else 30
        if self.opponent_possession_time > max_opponent_time:
            return True
        
        return False
    
    # Function to render the environment
    def render(self):
        if self.render_mode is None:
            return
        
        if self.window is None:
            pygame.init()
            # Add extra space for field margins and info display
            window_width = self.field_width + 250
            window_height = self.field_height + 200
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption(f"RoboCup Soccer RL Environment - {self.difficulty.title()} Mode")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # CRITICAL: Handle pygame events to prevent "Not Responding"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause functionality
                    input("Press Enter to continue...")
        
        # Colors
        field_green = (34, 139, 34)
        line_white = (255, 255, 255)
        goal_yellow = (255, 215, 0)
        ball_white = (255, 255, 255)
        robot_blue = (0, 100, 255)
        opponent_red = (255, 50, 50)
        possession_green = (0, 255, 0)
        
        # Fill background (darker green for border area)
        self.window.fill((20, 100, 20))
        
        # Draw field background
        field_rect = pygame.Rect(50, 50, self.field_width, self.field_height)
        pygame.draw.rect(self.window, field_green, field_rect)
        
        # Calculate line width based on field scale
        line_width = max(2, self.field_config.meters_to_pixels(0.05))
        
        # Draw field boundary
        pygame.draw.rect(self.window, line_white, 
                        (50, 50, self.field_width, self.field_height), line_width)
        
        # Draw field markings according to RoboCup specifications
        self._draw_complete_field_markings()
        
        # Draw goals
        self._draw_goals()
        
        # Draw strategic zones (if enabled for debugging)
        if hasattr(self, '_show_zones') and self._show_zones:
            self._draw_strategic_zones()
        
        # Draw ball with shadow effect
        ball_x, ball_y = int(self.ball_pos[0] + 50), int(self.ball_pos[1] + 50)
        # Shadow
        pygame.draw.circle(self.window, (200, 200, 200), 
                         (ball_x + 2, ball_y + 2), self.ball_radius)
        # Ball
        pygame.draw.circle(self.window, ball_white, 
                         (ball_x, ball_y), self.ball_radius)
        # Ball pattern (simple cross)
        pygame.draw.line(self.window, (0, 0, 0), 
                        (ball_x - self.ball_radius//2, ball_y), 
                        (ball_x + self.ball_radius//2, ball_y), 2)
        pygame.draw.line(self.window, (0, 0, 0), 
                        (ball_x, ball_y - self.ball_radius//2), 
                        (ball_x, ball_y + self.ball_radius//2), 2)
        
        # Draw robot (blue) with direction indicator
        robot_x, robot_y = int(self.robot_pos[0] + 50), int(self.robot_pos[1] + 50)
        # Robot shadow
        pygame.draw.circle(self.window, (0, 50, 100), 
                         (robot_x + 2, robot_y + 2), self.robot_radius)
        # Robot body
        pygame.draw.circle(self.window, robot_blue, 
                         (robot_x, robot_y), self.robot_radius)
        # Robot center
        pygame.draw.circle(self.window, line_white, 
                         (robot_x, robot_y), self.robot_radius // 3)
        
        # Draw robot facing direction
        direction_length = self.robot_radius + 10
        direction_end_x = robot_x + direction_length * np.cos(self.robot_angle)
        direction_end_y = robot_y + direction_length * np.sin(self.robot_angle)
        pygame.draw.line(self.window, line_white, 
                        (robot_x, robot_y), (int(direction_end_x), int(direction_end_y)), 3)
        
        # Draw opponent (red)
        opponent_x, opponent_y = int(self.opponent_pos[0] + 50), int(self.opponent_pos[1] + 50)
        # Opponent shadow
        pygame.draw.circle(self.window, (100, 25, 25), 
                         (opponent_x + 2, opponent_y + 2), self.robot_radius)
        # Opponent body
        pygame.draw.circle(self.window, opponent_red, 
                         (opponent_x, opponent_y), self.robot_radius)
        # Opponent center
        pygame.draw.circle(self.window, line_white, 
                         (opponent_x, opponent_y), self.robot_radius // 3)
        
        # Draw possession indicators
        if self.has_ball:
            pygame.draw.circle(self.window, possession_green, 
                             (robot_x, robot_y), self.robot_radius + 8, 4)
            # Possession text
            if not hasattr(self, '_font_small'):
                pygame.font.init()
                self._font_small = pygame.font.Font(None, 20)
            text = self._font_small.render("BALL", True, possession_green)
            self.window.blit(text, (robot_x - 15, robot_y - self.robot_radius - 25))
        
        if self.opponent_has_ball:
            pygame.draw.circle(self.window, (255, 100, 100), 
                             (opponent_x, opponent_y), self.robot_radius + 8, 4)
        
        # Draw velocity vectors (for debugging)
        if hasattr(self, '_show_velocities') and self._show_velocities:
            # Ball velocity
            if np.linalg.norm(self.ball_vel) > 0.1:
                vel_end_x = ball_x + self.ball_vel[0] * 5
                vel_end_y = ball_y + self.ball_vel[1] * 5
                pygame.draw.line(self.window, (255, 255, 0), 
                               (ball_x, ball_y), (vel_end_x, vel_end_y), 2)
        
        # Draw field information panel
        self._draw_field_info_panel()
        
        # Draw opponent behavior indicator
        self._draw_opponent_behavior()
        
        pygame.display.flip()

        # # Use testing FPS if in testing mode
        # rendering_config = self.field_config.config.get('rendering', {})
        # if self.testing_mode or rendering_config.get('testing_mode', False):
        #     target_fps = rendering_config.get('testing_fps', 30)
        # else:
        #     target_fps = rendering_config.get('fps', 60)
            
        # self.clock.tick(target_fps)

        # Speed up rendering for testing
        if self.testing_mode:
            self.clock.tick(120)  # Fast rendering
        else:
            self.clock.tick(60)   # Normal speed
    
    def _draw_complete_field_markings(self):
        """Draw complete RoboCup field markings"""
        line_white = (255, 255, 255)
        line_width = max(2, self.field_config.meters_to_pixels(0.05))
        
        # Field offset for drawing
        offset_x, offset_y = 50, 50
        
        # Center line
        center_x = self.field_width // 2 + offset_x
        pygame.draw.line(self.window, line_white, 
                        (center_x, offset_y), 
                        (center_x, self.field_height + offset_y), line_width)
        
        # Center circle
        center_radius = self.field_config.meters_to_pixels(
            self.field_config.real_dims['centre_circle_diameter'] / 2)
        pygame.draw.circle(self.window, line_white, 
                         (center_x, self.field_height // 2 + offset_y), 
                         center_radius, line_width)
        
        # Center mark
        pygame.draw.circle(self.window, line_white, 
                         (center_x, self.field_height // 2 + offset_y), 3)
        
        # Goal areas (both sides)
        goal_area_length = self.field_config.meters_to_pixels(
            self.field_config.real_dims['goal_area_length'])
        goal_area_width = self.field_config.meters_to_pixels(
            self.field_config.real_dims['goal_area_width'])
        
        # Right goal area
        goal_center_y = self.field_height // 2 + offset_y
        goal_area_rect_right = pygame.Rect(
            self.field_width - goal_area_length + offset_x,
            goal_center_y - goal_area_width // 2,
            goal_area_length,
            goal_area_width
        )
        pygame.draw.rect(self.window, line_white, goal_area_rect_right, line_width)
        
        # Left goal area
        goal_area_rect_left = pygame.Rect(
            offset_x,
            goal_center_y - goal_area_width // 2,
            goal_area_length,
            goal_area_width
        )
        pygame.draw.rect(self.window, line_white, goal_area_rect_left, line_width)
        
        # Penalty areas (both sides)
        penalty_area_length = self.field_config.meters_to_pixels(
            self.field_config.real_dims['penalty_area_length'])
        penalty_area_width = self.field_config.meters_to_pixels(
            self.field_config.real_dims['penalty_area_width'])
        
        # Right penalty area
        penalty_area_rect_right = pygame.Rect(
            self.field_width - penalty_area_length + offset_x,
            goal_center_y - penalty_area_width // 2,
            penalty_area_length,
            penalty_area_width
        )
        pygame.draw.rect(self.window, line_white, penalty_area_rect_right, line_width)
        
        # Left penalty area
        penalty_area_rect_left = pygame.Rect(
            offset_x,
            goal_center_y - penalty_area_width // 2,
            penalty_area_length,
            penalty_area_width
        )
        pygame.draw.rect(self.window, line_white, penalty_area_rect_left, line_width)
        
        # Penalty marks
        penalty_mark_distance = self.field_config.meters_to_pixels(
            self.field_config.real_dims['penalty_mark_distance'])
        
        # Right penalty mark
        penalty_mark_x_right = self.field_width - penalty_mark_distance + offset_x
        pygame.draw.circle(self.window, line_white, 
                         (int(penalty_mark_x_right), int(goal_center_y)), 4)
        
        # Left penalty mark
        penalty_mark_x_left = penalty_mark_distance + offset_x
        pygame.draw.circle(self.window, line_white, 
                         (int(penalty_mark_x_left), int(goal_center_y)), 4)
    
    def _draw_goals(self):
        """Draw goals at both ends"""
        goal_yellow = (255, 215, 0)
        goal_post_white = (255, 255, 255)
        
        offset_x, offset_y = 50, 50
        goal_center_y = self.field_height // 2 + offset_y
        goal_half_width = self.goal_width // 2
        goal_depth = self.field_config.goal_depth_pixels
        
        # Right goal (target goal)
        right_goal_rect = pygame.Rect(
            self.field_width + offset_x, 
            goal_center_y - goal_half_width,
            goal_depth,
            self.goal_width
        )
        pygame.draw.rect(self.window, goal_yellow, right_goal_rect)
        
        # Right goal posts
        pygame.draw.line(self.window, goal_post_white,
                        (self.field_width + offset_x, goal_center_y - goal_half_width),
                        (self.field_width + offset_x + goal_depth, goal_center_y - goal_half_width), 4)
        pygame.draw.line(self.window, goal_post_white,
                        (self.field_width + offset_x, goal_center_y + goal_half_width),
                        (self.field_width + offset_x + goal_depth, goal_center_y + goal_half_width), 4)
        pygame.draw.line(self.window, goal_post_white,
                        (self.field_width + offset_x + goal_depth, goal_center_y - goal_half_width),
                        (self.field_width + offset_x + goal_depth, goal_center_y + goal_half_width), 4)
        
        # Left goal (opponent goal)
        left_goal_rect = pygame.Rect(
            offset_x - goal_depth, 
            goal_center_y - goal_half_width,
            goal_depth,
            self.goal_width
        )
        pygame.draw.rect(self.window, (200, 200, 200), left_goal_rect)  # Gray for opponent goal

    def _draw_strategic_zones(self):
        """Draw strategic zones for debugging (semi-transparent overlays)"""
        overlay = pygame.Surface((self.field_width, self.field_height))
        overlay.set_alpha(30)  # Semi-transparent
        
        # Optimal shooting zone (green)
        if 'optimal_shooting' in self.field_config.strategic_zones:
            zone = self.field_config.strategic_zones['optimal_shooting']
            zone_rect = pygame.Rect(zone['x_min'], zone['y_min'], 
                                  zone['x_max'] - zone['x_min'], zone['y_max'] - zone['y_min'])
            pygame.draw.rect(overlay, (0, 255, 0), zone_rect)
        
        # Attacking zone (yellow)
        if 'attacking_zone' in self.field_config.strategic_zones:
            zone = self.field_config.strategic_zones['attacking_zone']
            zone_rect = pygame.Rect(zone['x_min'], zone['y_min'], 
                                  zone['x_max'] - zone['x_min'], zone['y_max'] - zone['y_min'])
            pygame.draw.rect(overlay, (255, 255, 0), zone_rect)
        
        self.window.blit(overlay, (0, 0))
    
    def _draw_field_info_panel(self):
        """Draw field information and stats"""
        if not hasattr(self, '_font'):
            pygame.font.init()
            self._font = pygame.font.Font(None, 24)
        
        # Field dimensions info
        info_text = [
            f"Field: {self.field_config.real_dims['field_length']}m x {self.field_config.real_dims['field_width']}m",
            f"Type: {self.field_config.config['field_type'].title()}",
            f"Difficulty: {self.difficulty.title()}",
            f"Steps: {self.steps}/{self.max_steps}",
            f"Ball possession: {'Robot' if self.has_ball else 'Opponent' if self.opponent_has_ball else 'None'}"
        ]
        
        y_offset = self.field_height + 60
        for i, text in enumerate(info_text):
            surface = self._font.render(text, True, (255, 255, 255))
            self.window.blit(surface, (10, y_offset + i * 20))
    
    def _draw_opponent_behavior(self):
        """Draw opponent behavior indicator"""
        if not hasattr(self, '_font_small'):
            pygame.font.init()
            self._font_small = pygame.font.Font(None, 20)
        
        behavior_text = f"Opponent: {self.opponent_behavior.title()}"
        color = {'aggressive': (255, 100, 100), 'defensive': (100, 100, 255), 'balanced': (100, 255, 100)}
        surface = self._font_small.render(behavior_text, True, color.get(self.opponent_behavior, (255, 255, 255)))
        self.window.blit(surface, (self.field_width + 60, 100))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def set_show_zones(self, show=True):
        """Enable/disable strategic zone visualization for debugging"""
        self._show_zones = show
    
    def set_show_velocities(self, show=True):
        """Enable/disable velocity vector visualization for debugging"""
        self._show_velocities = show

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
                if env._check_goal():
                    print(f"Episode ended at step {i}: GOAL SCORED!")
                elif env._check_opponent_goal():
                    print(f"Episode ended at step {i}: OPPONENT SCORED!")
                elif env._check_ball_out_of_play():
                    print(f"Episode ended at step {i}: BALL OUT OF BOUNDS!")
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