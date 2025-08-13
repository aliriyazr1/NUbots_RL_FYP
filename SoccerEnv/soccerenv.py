"""
soccerenv.py - Basic 2D Soccer Environment for FYP
This file contains the SoccerEnv class which implements a simple 2D soccer environment
where a robot needs to dribble a ball around an opponent to score in a goal.
The environment supports different difficulty levels and provides visual rendering using Pygame.

Author: Ali Riyaz
Student Number: C3412624
Last Updated: 09/08/2025
"""

#TODO: If I do end up using this, remember to fix the comments


# simple_soccer_env.py - Basic 2D Soccer Environment for FYP
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame, math, yaml, os
from pathlib import Path

class FieldConfig:
    """Class to handle field configuration loading and calculations"""
    
    def __init__(self, config_path="field_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._calculate_derived_values()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                print(f"Warning: Config file {self.config_path} not found. Using default kidsize configuration.")
                return self._get_default_config()
            
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default kidsize configuration if file loading fails"""
        return {
            'field_type': 'kidsize',
            'real_world_dimensions': {
                'kidsize': {
                    'field_length': 9.0, 'field_width': 6.0, 'goal_depth': 0.6,
                    'goal_width': 2.6, 'goal_height': 1.2, 'goal_area_length': 1.0,
                    'goal_area_width': 3.0, 'penalty_mark_distance': 1.5,
                    'centre_circle_diameter': 1.5, 'border_strip_width': 1.0,
                    'penalty_area_length': 2.0, 'penalty_area_width': 5.0,
                    'line_width': 0.05
                }
            },
            'display_dimensions': {'width': 900, 'height': 600, 'scale_factor': 100},
            'robot_parameters': {'robot_radius': 0.15, 'ball_radius': 0.08, 'possession_threshold': 0.25},
            'physics': {'ball_mass': 0.5, 'ball_friction': 0.8, 'ball_bounce': 0.2, 
                       'ball_max_speed': 3.0, 'push_force_multiplier': 1.2},
            'difficulty_settings': {
                'easy': {'possession_distance': 0.5, 'collision_distance': 0.2, 'max_steps': 150, 'opponent_speed': 0.8},
                'medium': {'possession_distance': 0.4, 'collision_distance': 0.25, 'max_steps': 200, 'opponent_speed': 1.0},
                'hard': {'possession_distance': 0.35, 'collision_distance': 0.3, 'max_steps': 250, 'opponent_speed': 1.2}
            },
            'reward_parameters': {
                'goal_scored': 15.0, 'time_penalty': -0.002, 'collision_penalty': -1.0,
                'edge_penalty': -1.0, 'corner_penalty': -0.5, 'center_field_bonus': 0.3
            }
        }
    
    def _calculate_derived_values(self):
        """Calculate derived values like pixel scale and strategic zones"""
        field_type = self.config['field_type']
        
        if field_type == 'custom':
            self.real_dims = self.config['custom_dimensions']
        else:
            self.real_dims = self.config['real_world_dimensions'][field_type]
        
        # Calculate pixel scale factors
        display_dims = self.config['display_dimensions']
        self.pixels_per_meter_x = display_dims['width'] / self.real_dims['field_length']
        self.pixels_per_meter_y = display_dims['height'] / self.real_dims['field_width']
        
        # Use consistent scale factor (smaller of the two to ensure field fits)
        self.pixels_per_meter = min(self.pixels_per_meter_x, self.pixels_per_meter_y)
        
        # Calculate actual pixel dimensions
        self.field_width_pixels = int(self.real_dims['field_length'] * self.pixels_per_meter)
        self.field_height_pixels = int(self.real_dims['field_width'] * self.pixels_per_meter)
        
        # Goal dimensions in pixels
        self.goal_width_pixels = int(self.real_dims['goal_width'] * self.pixels_per_meter)
        self.goal_depth_pixels = int(self.real_dims['goal_depth'] * self.pixels_per_meter)
        
        # Calculate strategic zones in pixel coordinates
        self._calculate_strategic_zones()
    
    def _calculate_strategic_zones(self):
        """Calculate strategic zone coordinates in pixels"""
        zones = self.config.get('strategic_zones', {})
        self.strategic_zones = {}
        
        # Calculate optimal shooting zone
        if 'optimal_shooting' in zones:
            zone = zones['optimal_shooting']
            self.strategic_zones['optimal_shooting'] = {
                'x_min': int(zone['x_min_percent'] * self.field_width_pixels),
                'x_max': int(zone['x_max_percent'] * self.field_width_pixels),
                'y_min': int(zone['y_min_percent'] * self.field_height_pixels),
                'y_max': int(zone['y_max_percent'] * self.field_height_pixels)
            }
        
        # Calculate attacking zone
        if 'attacking_zone' in zones:
            zone = zones['attacking_zone']
            self.strategic_zones['attacking_zone'] = {
                'x_min': int(zone['x_min_percent'] * self.field_width_pixels),
                'x_max': int(zone['x_max_percent'] * self.field_width_pixels),
                'y_min': int(zone['y_min_percent'] * self.field_height_pixels),
                'y_max': int(zone['y_max_percent'] * self.field_height_pixels)
            }
        
        # Calculate goal area coordinates
        goal_center_y = self.field_height_pixels // 2
        goal_half_width = self.goal_width_pixels // 2
        self.strategic_zones['goal_area'] = {
            'x_min': self.field_width_pixels - int(self.real_dims['goal_area_length'] * self.pixels_per_meter),
            'x_max': self.field_width_pixels,
            'y_min': goal_center_y - int(self.real_dims['goal_area_width'] * self.pixels_per_meter // 2),
            'y_max': goal_center_y + int(self.real_dims['goal_area_width'] * self.pixels_per_meter // 2)
        }
        
        # Calculate penalty area coordinates
        self.strategic_zones['penalty_area'] = {
            'x_min': self.field_width_pixels - int(self.real_dims['penalty_area_length'] * self.pixels_per_meter),
            'x_max': self.field_width_pixels,
            'y_min': goal_center_y - int(self.real_dims['penalty_area_width'] * self.pixels_per_meter // 2),
            'y_max': goal_center_y + int(self.real_dims['penalty_area_width'] * self.pixels_per_meter // 2)
        }
    
    def meters_to_pixels(self, meters):
        """Convert meters to pixels"""
        return int(meters * self.pixels_per_meter)
    
    def pixels_to_meters(self, pixels):
        """Convert pixels to meters"""
        return pixels / self.pixels_per_meter
    
    def is_in_zone(self, x, y, zone_name):
        """Check if coordinates are within a strategic zone"""
        if zone_name not in self.strategic_zones:
            return False
        
        zone = self.strategic_zones[zone_name]
        return (zone['x_min'] <= x <= zone['x_max'] and 
                zone['y_min'] <= y <= zone['y_max'])
    
    def is_ball_in_goal(self, ball_x, ball_y):
        """Check if ball is in goal"""
        goal_center_y = self.field_height_pixels // 2
        goal_half_width = self.goal_width_pixels // 2
        
        return (ball_x >= self.field_width_pixels and
                goal_center_y - goal_half_width <= ball_y <= goal_center_y + goal_half_width)

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
        self.field_width = self.field_config.field_width_pixels
        self.field_height = self.field_config.field_height_pixels
        self.goal_width = self.field_config.goal_width_pixels
        
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

        # # Game parameters based on difficulty
        # if difficulty == "easy":
        #     self.possession_distance = 50.0  # Easier to get ball
        #     self.collision_distance = 20.0   # More forgiving collisions
        #     self.max_steps = 150            # Shorter episodes
        # elif difficulty == "medium":
        #     self.possession_distance = 40.0
        #     self.collision_distance = 25.0
        #     self.max_steps = 200
        # else:  # hard
        #     self.possession_distance = 35.0
        #     self.collision_distance = 30.0
        #     self.max_steps = 250

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

        # Robot movement parameters - adjusted for testing mode
        base_robot_speed = 1.8 # Used tp be 2.5
        base_rotation_speed = 0.08 # Used to be 0.1
        
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
        self.ball_vel = np.array([0.0, 0.0])
        
        # Randomize robot starting position (left third of field)
        self.robot_pos = np.array([
            np.random.uniform(self.robot_radius + 20, self.field_width * 0.33),
            np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        ])
        self.robot_angle = np.random.uniform(-np.pi/4, np.pi/4)  # Roughly facing right
        self.robot_vel = np.array([0.0, 0.0])
        
        # Randomize ball position (center third of field, avoiding edges)
        self.ball_pos = np.array([
            np.random.uniform(self.field_width * 0.3, self.field_width * 0.7),
            np.random.uniform(self.ball_radius + 30, self.field_height - self.ball_radius - 30)
        ])
        
        # Randomize opponent position (right third of field)
        self.opponent_pos = np.array([
            np.random.uniform(self.field_width * 0.6, self.field_width - self.robot_radius - 20),
            np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        ])
        
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

        # TODO: Randomise starting positions
        # This is essential for the robot to learn generalisable behavior
        # # Starting positions based on difficulty
        # if self.difficulty == "easy":
        #     # Easy: Robot starts close to ball, while the opponent is far
        #     self.robot_pos = np.array([80.0, 200.0])
        #     self.ball_pos = np.array([120.0, 200.0])
        #     self.opponent_pos = np.array([300.0, 300.0])
        # elif self.difficulty == "medium":
        #     # Medium: More challenging
        #     self.robot_pos = np.array([
        #         np.random.uniform(50, 150),
        #         np.random.uniform(100, 300)
        #     ])
        #     self.ball_pos = np.array([
        #         np.random.uniform(80, 200),
        #         np.random.uniform(100, 300)
        #     ])
        #     self.opponent_pos = np.array([
        #         np.random.uniform(200, 350),
        #         np.random.uniform(100, 300)
        #     ])
        # else:  # hard
        #     # Hard: Random positions
        #     self.robot_pos = np.array([
        #         np.random.uniform(30, 180),
        #         np.random.uniform(50, 350)
        #     ])
        #     self.ball_pos = np.array([
        #         np.random.uniform(60, 320),
        #         np.random.uniform(50, 350)
        #     ])
        #     self.opponent_pos = np.array([
        #         np.random.uniform(150, 350),
        #         np.random.uniform(50, 350)
        #     ])
        
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
    def _get_obs(self):
        # Normalize positions to [-1, 1] range
        norm_robot_x = (2 * self.robot_pos[0] / self.field_width) - 1
        norm_robot_y = (2 * self.robot_pos[1] / self.field_height) - 1
        norm_ball_x = (2 * self.ball_pos[0] / self.field_width) - 1
        norm_ball_y = (2 * self.ball_pos[1] / self.field_height) - 1
        norm_opponent_x = (2 * self.opponent_pos[0] / self.field_width) - 1
        norm_opponent_y = (2 * self.opponent_pos[1] / self.field_height) - 1
        
        # Normalize velocities
        norm_robot_vx = np.clip(self.robot_vel[0] / 100, -1, 1)
        norm_robot_vy = np.clip(self.robot_vel[1] / 100, -1, 1)
        
        # Calculate distances
        ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos) / max(self.field_width, self.field_height)
        goal_distance = np.linalg.norm(self.robot_pos - self.goal_pos) / max(self.field_width, self.field_height)
        
        # Normalize angle
        norm_angle = self.robot_angle / np.pi
        
        obs = np.array([
            norm_robot_x, norm_robot_y, norm_angle,
            norm_ball_x, norm_ball_y,
            norm_opponent_x, norm_opponent_y,
            norm_robot_vx, norm_robot_vy,
            ball_distance, goal_distance,
            float(self.has_ball)
        ], dtype=np.float32)
    
        # Ensure no NaN/inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

        # # Normalise positions to [-1, 1] range with clipping
        # robot_x = np.clip((self.robot_pos[0] - 200) / 200, -1, 1)
        # robot_y = np.clip((self.robot_pos[1] - 200) / 200, -1, 1)
        # ball_x = np.clip((self.ball_pos[0] - 200) / 200, -1, 1)
        # ball_y = np.clip((self.ball_pos[1] - 200) / 200, -1, 1)
        # opponent_x = np.clip((self.opponent_pos[0] - 200) / 200, -1, 1)
        # opponent_y = np.clip((self.opponent_pos[1] - 200) / 200, -1, 1)
        
        # # Calculate distances with proper normalisation
        # ball_distance = np.clip(np.linalg.norm(self.robot_pos - self.ball_pos) / 200, 0, 1)
        # goal_distance = np.clip(np.linalg.norm(self.robot_pos - self.goal_pos) / 200, 0, 1)
        
        # # # Check ball possession for both robot and opponent
        # # robot_ball_dist = np.linalg.norm(self.robot_pos - self.ball_pos)
        # # opponent_ball_dist = np.linalg.norm(self.opponent_pos - self.ball_pos)
        # # # Possession logic: closer player gets ball, but with some "stickiness" to make it harder to steal the ball from the robot that has possession
        # # if robot_ball_dist < self.possession_distance:
        # #     if not self.opponent_has_ball or robot_ball_dist < opponent_ball_dist * 0.8:
        # #         self.has_ball = True
        # #         self.opponent_has_ball = False
        # #         self.robot_possession_time += 1
        # #         self.opponent_possession_time = 0
        # # elif opponent_ball_dist < self.possession_distance:
        # #     if not self.has_ball or opponent_ball_dist < robot_ball_dist * 0.8:
        # #         self.has_ball = False
        # #         self.opponent_has_ball = True
        # #         self.opponent_possession_time += 1
        # #         self.robot_possession_time = 0
        # # else:
        # #     # Ball is loose. Neither player has possession
        # #     self.has_ball = False
        # #     self.opponent_has_ball = False
        # #     self.robot_possession_time = 0
        # #     self.opponent_possession_time = 0

        # # All observations properly bounded
        # obs = np.array([
        #     robot_x, robot_y, 
        #     np.clip(self.robot_angle / np.pi, -1, 1),
        #     ball_x, ball_y,
        #     opponent_x, opponent_y,
        #     np.clip(self.robot_vel[0] / 5, -1, 1), 
        #     np.clip(self.robot_vel[1] / 5, -1, 1),
        #     ball_distance, goal_distance,
        #     float(self.has_ball)
        # ], dtype=np.float32)
        
        # # Ensure no NaN/inf values
        # obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        # return obs
    
    # To take an action
    def step(self, action):
        self.steps += 1

        # Apply robot action
        self._apply_robot_action(action)

        # # Clip action to ensure it's in valid range
        # action = np.clip(action, -1.0, 1.0)
        # action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # # Apply action to robot
        # forward = action[0] * self.robot_speed
        # strafe = action[1] * self.robot_speed
        # rotation = action[2] * self.robot_rotation_speed

        # # Update robot angle first
        # self.robot_angle += rotation
        # # Keep angle in reasonable range
        # self.robot_angle = np.clip(self.robot_angle, -np.pi, np.pi)
        
        # # Update robot velocity
        # self.robot_vel[0] = forward * np.cos(self.robot_angle) - strafe * np.sin(self.robot_angle)
        # self.robot_vel[1] = forward * np.sin(self.robot_angle) + strafe * np.cos(self.robot_angle)
        
        # # Update robot position
        # self.robot_pos += self.robot_vel
        
        # # Keep robot in bounds
        # self.robot_pos[0] = np.clip(self.robot_pos[0], 20, 380)
        # self.robot_pos[1] = np.clip(self.robot_pos[1], 20, 380)
        
        # # Removed since the updated one is in the update_ball_physics function
        # # # Ball physics 
        # # if self.has_ball:
        # #     # Ball follows robot
        # #     offset_distance = 20
        # #     target_ball_pos = self.robot_pos + offset_distance * np.array([
        # #         np.cos(self.robot_angle), np.sin(self.robot_angle)
        # #     ])
        # #     self.ball_pos += 0.4 * (target_ball_pos - self.ball_pos)
            
        # # elif self.opponent_has_ball:
        # #     # Ball follows opponent
        # #     offset_distance = 20
        # #     # Opponent tries to move ball away from robot
        # #     to_robot = self.robot_pos - self.opponent_pos
        # #     if np.linalg.norm(to_robot) > 1e-6: # Prevent division by zero
        # #         away_from_robot = -to_robot / np.linalg.norm(to_robot)
        # #         target_ball_pos = self.opponent_pos + offset_distance * away_from_robot
        # #         self.ball_pos += 0.4 * (target_ball_pos - self.ball_pos)
        # # else:
        # #     # Ball is loose - apply physics
        # #     self.ball_vel *= 0.9  # Friction
        # #     self.ball_pos += self.ball_vel
            
        # #     # Add some random ball movement when loose
        # #     if np.random.random() < 0.05:
        # #         self.ball_vel += np.random.uniform(-0.3, 0.3, 2)
        
        # # Keep ball in bounds
        # self.ball_pos[0] = np.clip(self.ball_pos[0], 15, 385)
        # self.ball_pos[1] = np.clip(self.ball_pos[1], 15, 385)
                
        # === NEW REALISTIC BALL PHYSICS ===
        self._update_ball_physics()
        
        # Opponent AI
        self._update_opponent()
        
        # Update possession flags based on proximity (transition period)
        self._update_possession_flags()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_terminated()
        truncated = self.steps >= self.max_steps

        # Check if ball went out of bounds (simple implementation)
        ball_out_of_bounds = (self.ball_pos[0] < 0 or self.ball_pos[0] > self.field_width or
                             self.ball_pos[1] < 0 or self.ball_pos[1] > self.field_height)
        
        if ball_out_of_bounds and not terminated:
            # Reset ball to center if it goes out of bounds
            self.ball_pos = np.array([
                self.field_width * 0.5,
                self.field_height * 0.5
            ])
            self.ball_vel = np.array([0.0, 0.0])
            reward += self.field_config.config.get('reward_parameters', {}).get('out_of_bounds_penalty', -2.0)
        
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _apply_robot_action(self, action):
        # Parse action
        forward_back = action[0]  # -1 to 1
        left_right = action[1]    # -1 to 1  
        rotation = action[2]      # -1 to 1
        
        # Update robot angle
        self.robot_angle += rotation * self.robot_rotation_speed
        self.robot_angle = self.robot_angle % (2 * np.pi)
        
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
        new_pos = self.robot_pos + self.robot_vel * 0.1  # Time step
        
        # Keep robot within bounds
        new_pos[0] = np.clip(new_pos[0], self.robot_radius, self.field_width - self.robot_radius)
        new_pos[1] = np.clip(new_pos[1], self.robot_radius, self.field_height - self.robot_radius)
        
        self.robot_pos = new_pos
    
    def _update_opponent(self):
        """Enhanced opponent AI with different behaviors"""
        
        if not hasattr(self, 'opponent_behavior'):
            self.opponent_behavior = 'balanced'
        
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
            goal_center = np.array([self.field_width, self.field_height // 2])
            
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
        base_speed = 20.0 * self.opponent_speed
        if self.testing_mode or self.field_config.config.get('rendering', {}).get('testing_mode', False):
            speed_multiplier = self.field_config.config.get('rendering', {}).get('testing_speed_multiplier', 2.0)
            speed = base_speed * speed_multiplier
        else:
            speed = base_speed
        
        # Add some randomness to make opponent less predictable
        if np.random.random() < 0.1:  # 10% chance for random movement
            random_angle = np.random.uniform(0, 2 * np.pi)
            direction += 0.3 * np.array([np.cos(random_angle), np.sin(random_angle)])
            direction = direction / np.linalg.norm(direction)
        
        self.opponent_pos += direction * speed * 0.1
        
        # Keep opponent within bounds with margin
        margin = self.robot_radius + 10
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], margin, self.field_width - margin)
        self.opponent_pos[1] = np.clip(self.opponent_pos[1], margin, self.field_height - margin)
    
        """Opponent AI with different behaviours based on game state and difficulty"""

        # opponent_to_ball = self.ball_pos - self.opponent_pos
        # opponent_to_robot = self.robot_pos - self.opponent_pos

        # # Prevent division by zero
        # robot_dist = np.linalg.norm(opponent_to_robot)
        # ball_dist = np.linalg.norm(opponent_to_ball)

        # if self.difficulty == "easy":
        #     # Easy: Slow, predictable opponent
        #     if ball_dist > 5:
        #         ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #         self.opponent_pos += 1.0 * ball_dir
                
        # elif self.difficulty == "medium":
        #     # Medium: More intelligent movement
        #     if ball_dist < 30:  # Close to ball
        #         # Try to push ball away from robot
        #         if robot_dist > 1e-6:
        #             # Move to position ball between opponent and robot
        #             ideal_pos = self.ball_pos + 20 * (self.ball_pos - self.robot_pos) / robot_dist
        #             to_ideal = ideal_pos - self.opponent_pos
        #             if np.linalg.norm(to_ideal) > 1e-6:
        #                 self.opponent_pos += 1.2 * to_ideal / np.linalg.norm(to_ideal)
        #         else:
        #             # Just move toward ball
        #             ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #             self.opponent_pos += 1.2 * ball_dir
        #     else:
        #         # Chase ball
        #         ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #         self.opponent_pos += 1.5 * ball_dir
        #     # if self.opponent_has_ball:
        #     #     if robot_dist > 1e-6 and robot_dist < 60:
        #     #         escape_dir = -opponent_to_robot / robot_dist
        #     #         self.opponent_pos += 1.5 * escape_dir
        #     # elif ball_dist > 5:
        #     #     ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #     #     self.opponent_pos += 1.8 * ball_dir

        # else:  # hard
        #     # Hard: Smart positioning and interception
        #     if ball_dist < 25:
        #         # Close to ball - try to control it
        #         if robot_dist > 30:
        #             # Robot is far, focus on ball control
        #             ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #             self.opponent_pos += 1.8 * ball_dir
        #         else:
        #             # Robot is close, play defensively
        #             # Position between robot and goal
        #             goal_pos = np.array([350.0, 200.0])
        #             robot_to_goal = goal_pos - self.robot_pos
        #             if np.linalg.norm(robot_to_goal) > 1e-6:
        #                 intercept_pos = self.robot_pos + 0.7 * robot_to_goal
        #                 to_intercept = intercept_pos - self.opponent_pos
        #                 if np.linalg.norm(to_intercept) > 1e-6:
        #                     self.opponent_pos += 1.6 * to_intercept / np.linalg.norm(to_intercept)
        #     else:
        #         # Far from ball - chase it
        #         ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #         self.opponent_pos += 2.0 * ball_dir

        #     # # Hard: Smart opponent (your original logic with safety)
        #     # if self.opponent_has_ball:
        #     #     if robot_dist > 1e-6 and robot_dist < 60:
        #     #         escape_dir = -opponent_to_robot / robot_dist
        #     #         self.opponent_pos += 2.0 * escape_dir
        #     #     else:
        #     #         random_move = np.random.uniform(-0.8, 0.8, 2)
        #     #         self.opponent_pos += random_move
        #     # elif self.has_ball:
        #     #     if robot_dist > 1e-6 and robot_dist > 30:
        #     #         intercept_dir = opponent_to_robot / robot_dist
        #     #         self.opponent_pos += 1.8 * intercept_dir
        #     # else:
        #     #     if ball_dist > 5:
        #     #         ball_dir = opponent_to_ball / (ball_dist + 1e-6)
        #     #         self.opponent_pos += 1.8 * ball_dir
        
    
        # # Keep opponent in bounds
        # self.opponent_pos[0] = np.clip(self.opponent_pos[0], 20, 380)
        # self.opponent_pos[1] = np.clip(self.opponent_pos[1], 20, 380)
        
    def _calculate_reward(self):
        reward = 0.0

        # Load reward parameters from config
        reward_params = self.field_config.config.get('reward_parameters', {
            'goal_scored': 15.0, 'time_penalty': -0.002, 'collision_penalty': -1.0,
            'edge_penalty': -1.0, 'corner_penalty': -0.5, 'center_field_bonus': 0.3
        })
        
        # Time penalty
        reward += reward_params['time_penalty']

        # Distance calculations using configurable field dimensions
        ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        robot_x, robot_y = self.robot_pos
        
        # Goal distance that considers BOTH X and Y coordinates (adapt to field size)
        goal_center = np.array([self.field_width, self.field_height // 2])
        goal_distance = np.linalg.norm(self.robot_pos - goal_center)
        
        # Check if robot is actually in front of goal (scaled to field size)
        goal_area_x_min = self.field_width * 0.75  # 75% of field width
        goal_area_y_min = self.field_height * 0.4   # 40% of field height
        goal_area_y_max = self.field_height * 0.6   # 60% of field height
        goal_aligned = (robot_x > goal_area_x_min and goal_area_y_min < robot_y < goal_area_y_max)
        
        # PHASE 1: Getting the ball (most important for learning)
        max_distance = np.sqrt(self.field_width**2 + self.field_height**2)
        ball_proximity_reward = (max_distance - ball_distance) / max_distance
        reward += ball_proximity_reward * 8.0  # Strong ball proximity reward

        # Dense reward shaping for approaching ball (scaled to field)
        close_threshold = self.field_config.meters_to_pixels(0.3)    # 30cm
        medium_threshold = self.field_config.meters_to_pixels(0.5)   # 50cm  
        far_threshold = self.field_config.meters_to_pixels(0.8)      # 80cm
        very_far_threshold = self.field_config.meters_to_pixels(1.2) # 120cm
        
        if ball_distance < close_threshold:
            reward += 8.0    # Close to ball (Used to be 1.0)
        elif ball_distance < medium_threshold:
            reward += 4.0    # Getting closer (Used to be 0.5)
        elif ball_distance < far_threshold:
            reward += 2.0    # Moving in right direction (Used to be 0.2)
        elif ball_distance < very_far_threshold:
            reward += 1.0    # Still on track
        else:
            # Penalty for being far from ball
            reward -= 0.5 # (used to be 0.3)

        # PHASE 2: Reward for moving towards ball (velocity-based)
        robot_speed = np.linalg.norm(self.robot_vel)
        if ball_distance > close_threshold and robot_speed > 0.1:  # Only if moving fast enough
            # Calculate if robot is moving towards ball
            ball_direction = (self.ball_pos - self.robot_pos)
            if np.linalg.norm(ball_direction) > 0:
                ball_direction_norm = ball_direction / np.linalg.norm(ball_direction)
                robot_velocity_norm = self.robot_vel / robot_speed
                
                # Reward for moving towards ball
                velocity_alignment = np.dot(robot_velocity_norm, ball_direction_norm)
                if velocity_alignment > 0:
                    reward += velocity_alignment * robot_speed * 0.5  # Reward moving towards ball

        # PHASE 3: Reward for pushing ball toward goal
        ball_x, ball_y = self.ball_pos
        ball_to_goal_distance = np.linalg.norm(self.ball_pos - goal_center)
        
        # Only reward ball progress when robot is close to ball
        if ball_distance < medium_threshold:
            max_goal_distance = np.linalg.norm([0, self.field_height//2] - goal_center)
            goal_progress = (max_goal_distance - ball_to_goal_distance) / max_goal_distance
            reward += goal_progress * 4.0
            
            # Extra reward if robot is "shepherding" ball toward goal
            shepherding_threshold = self.field_config.meters_to_pixels(0.35)  # 35cm
            if ball_distance < shepherding_threshold:
                robot_ball_goal_angle = self._calculate_shepherding_angle()
                if robot_ball_goal_angle < 30:  # Very good angle
                    reward += 4.0
                elif robot_ball_goal_angle < 60:  # Good angle
                    reward += 2.0
                elif robot_ball_goal_angle < 90:  # Okay angle
                    reward += 1.0

        # PHASE 4: Reward for ball speed toward goal (good pushing)
        ball_speed = np.linalg.norm(self.ball_vel)
        if ball_speed > 0.1 and ball_distance < medium_threshold:  # Only when close to ball
            ball_to_goal = goal_center - self.ball_pos
            if np.linalg.norm(ball_to_goal) > 1e-6:
                ball_to_goal_normalized = ball_to_goal / np.linalg.norm(ball_to_goal)
                ball_vel_normalized = self.ball_vel / ball_speed
                
                # Dot product: 1 = moving directly toward goal, -1 = away from goal
                goal_direction_alignment = np.dot(ball_vel_normalized, ball_to_goal_normalized)
                if goal_direction_alignment > 0:
                    reward += goal_direction_alignment * min(ball_speed, 2.0) * 1.0
        
        # HUGE reward for scoring
        if self._check_goal_scored():
            reward += reward_params['goal_scored']
            print("🎉 GOAL SCORED!")


        # === ANTI-EXPLOIT PENALTIES ===
        # 4A: STRONG penalties for edge/corner behavior (scaled to field size)
        edge_threshold = min(self.field_width, self.field_height) * 0.1  # 10% of field size
        
        # Robot near edges
        if (robot_x < edge_threshold or robot_x > (self.field_width - edge_threshold) or 
            robot_y < edge_threshold or robot_y > (self.field_height - edge_threshold)):
            reward += reward_params['edge_penalty']  # Strong penalty for robot camping edges
        
        # Ball near edges (prevents pushing ball to edges)
        if (ball_x < edge_threshold or ball_x > (self.field_width - edge_threshold) or 
            ball_y < edge_threshold or ball_y > (self.field_height - edge_threshold)):
            reward -= 0.8  # Penalty for ball being at edges
            
            # EXTRA penalty if robot is also near edge along with the ball (both camping)
            if (robot_x < edge_threshold or robot_x > (self.field_width - edge_threshold) or 
                robot_y < edge_threshold or robot_y > (self.field_height - edge_threshold)):
                reward -= 1.5  # Very strong penalty for both at edges
        
        # 4B: Penalty for ball moving along edges (detect edge following)
        if ball_speed > 0.1:
            # Check if ball is moving along edges rather than toward center
            if ((ball_x < edge_threshold or ball_x > (self.field_width - edge_threshold)) or
                (ball_y < edge_threshold or ball_y > (self.field_height - edge_threshold))):
                # Ball is at edge and moving - check if moving along edge
                if abs(self.ball_vel[0]) > abs(self.ball_vel[1]) * 2:  # Moving mostly horizontally
                    if ball_y < edge_threshold or ball_y > (self.field_height - edge_threshold):  # At top/bottom edge
                        reward -= 1.0  # Penalty for moving along top/bottom edges
                elif abs(self.ball_vel[1]) > abs(self.ball_vel[0]) * 2:  # Moving mostly vertically  
                    if ball_x < edge_threshold or ball_x > (self.field_width - edge_threshold):  # At left/right edge
                        reward -= 1.0  # Penalty for moving along left/right edges
        
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
        corner_threshold = min(self.field_width, self.field_height) * 0.15  # 15% of field size
        if (robot_x < corner_threshold or robot_x > (self.field_width - corner_threshold) or 
            robot_y < corner_threshold or robot_y > (self.field_height - corner_threshold)):
            reward += reward_params['corner_penalty']

        # Extra reward for getting ball into dangerous areas (near goal) - scaled to field
        dangerous_area_x = self.field_width * 0.75  # 75% of field width
        dangerous_area_y_min = self.field_height * 0.375  # 37.5% of field height
        dangerous_area_y_max = self.field_height * 0.625  # 62.5% of field height
        
        if ball_x > dangerous_area_x and dangerous_area_y_min < ball_y < dangerous_area_y_max:
            reward += 2.0  # Ball in scoring area
            
            # Even more if robot is still close (maintaining control)
            if ball_distance < self.field_config.meters_to_pixels(0.4):  # Within 40cm
                reward += 2.0

        # Penalty if opponent is closer to ball (competitive element)
        opponent_ball_distance = np.linalg.norm(self.opponent_pos - self.ball_pos)
        if opponent_ball_distance < ball_distance:
            reward -= 0.1

        # Clip rewards to prevent explosions
        reward = np.clip(reward, -3.0, 20.0)
        
        # Safety check
        if not np.isfinite(reward):
            reward = 0.0
            
        return float(reward)

        # # PHASE 2: Moving with ball (second priority)
        # if self.has_ball:
        #     # Good reward for having ball
        #     reward += 2.0
            
        #     # Define optimal shooting positions (directly in front of goal opening)
        #     optimal_shooting_zone = (robot_x > 290 and robot_x < 340 and robot_y > 170 and robot_y < 230)  # Directly in front of goal
            
        #     # Define goalpost camping zones (next to goalposts but can't score)
        #     near_top_goalpost = (robot_x > 320 and robot_y < 160)     # Above goal
        #     near_bottom_goalpost = (robot_x > 320 and robot_y > 240)  # Below goal
        #     goalpost_camping = near_top_goalpost or near_bottom_goalpost

        #     # Reward for moving toward goal center, not just right edge
        #     # Considers both distance AND alignment
        #     if optimal_shooting_zone:
        #         # Reward for the robot if aligned with goal
        #         reward += 5.0
        #         # Additional reward for being close to goal center when aligned
        #         center_distance = np.linalg.norm(self.robot_pos - goal_center)
        #         reward += 3.0 * (200 - center_distance) / 200
        #     elif goalpost_camping:
        #         # PENALTY for camping next to goalposts (can't score from there)
        #         reward -= 3.0  # Strong penalty for goalpost camping
        #         print(f"⚠️ GOALPOST CAMPING DETECTED at ({robot_x:.0f}, {robot_y:.0f})")
        #     else:
        #         # Robot has ball but not aligned with goal
        #         # Small reward for moving right, but penalty for going to wrong Y position
        #         rightward_progress = max(0, (robot_x - 100) / 250)  # Progress rightward
        #         reward += rightward_progress * 0.5
                
        #         # Penalty for being far from goal Y-center (prevents edge camping)
        #         y_distance_from_goal_center = abs(robot_y - 200)  # Distance from Y=200
        #         if y_distance_from_goal_center > 100:  # Far from goal center
        #             reward -= 1.0  # Penalty for being at wrong Y position
            
        #     # Possession time bonus (but capped)
        #     reward += min(self.robot_possession_time * 0.01, 0.2)
    ##########################################################
    
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


    def _calculate_shepherding_angle(self):
        """Calculate the angle between robot-ball vector and ball-goal vector"""
        try:
            # Vector from robot to ball
            robot_to_ball = self.ball_pos - self.robot_pos
            
            # Vector from ball to goal (Goal on the Right side of the field)
            goal_center = np.array([self.field_width, self.field_height // 2])
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
    
    def _update_ball_physics(self):
        """Realistic ball physics with pushing, friction, and bouncing"""
        
        # ROBOT-BALL INTERACTION (pushing)
        robot_to_ball = self.ball_pos - self.robot_pos
        robot_ball_distance = np.linalg.norm(robot_to_ball)
        
        # Check if robot is touching/pushing the ball
        contact_distance = self.robot_radius + self.ball_radius
        if robot_ball_distance < contact_distance and robot_ball_distance > 0.1:
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
                robot_direction = self.robot_vel / np.linalg.norm(self.robot_vel)
                guidance_strength = min(robot_speed * 0.15, 0.8)  # Limit guidance strength
                self.ball_vel += robot_direction * guidance_strength
        
        # OPPONENT-BALL INTERACTION 
        opponent_to_ball = self.ball_pos - self.opponent_pos
        opponent_ball_distance = np.linalg.norm(opponent_to_ball)
        
        if opponent_ball_distance < contact_distance and opponent_ball_distance > 0.1:
            overlap = contact_distance - opponent_ball_distance
            push_direction = opponent_to_ball / opponent_ball_distance
            
            # Opponent push force
            opponent_speed = 1.2  # Assume constant opponent "speed"
            push_force = (opponent_speed * 0.4 + overlap * 0.2) * self.push_force_multiplier
            
            push_velocity = push_direction * push_force
            self.ball_vel += push_velocity
        
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
        self.ball_pos += self.ball_vel
        
        # Handle wall bounces
        self._handle_ball_wall_bounces()
        
        # Stop very slow ball (prevent infinite tiny movements)
        if ball_speed < 0.08:
            self.ball_vel *= 0.3  # Quickly dampen very slow movement
    

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
    
    # def _check_goal(self):
    #     """Check if ball is in goal"""
    #     return (self.ball_pos[0] > 340 and 160 < self.ball_pos[1] < 240)

    def _check_goal(self):
        """Check if ball is in goal"""
        return self.field_config.is_ball_in_goal(self.ball_pos[0], self.ball_pos[1])
    
    def _check_terminated(self):
        # Success: ball in goal
        if self._check_goal():
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
            window_width = self.field_width + 200
            window_height = self.field_height + 150
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

        # Use testing FPS if in testing mode
        rendering_config = self.field_config.config.get('rendering', {})
        if self.testing_mode or rendering_config.get('testing_mode', False):
            target_fps = rendering_config.get('testing_fps', 30)
        else:
            target_fps = rendering_config.get('fps', 60)
            
        self.clock.tick(target_fps)
        # self.clock.tick(60)  # 60 FPS
    
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