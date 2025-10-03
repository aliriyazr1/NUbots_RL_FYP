"""
soccerenv.py - Basic 2D Soccer Environment for FYP
This file contains the SoccerEnv class which implements a simple 2D soccer environment
where a robot needs to dribble a ball around an opponent to score in a goal.
The environment supports different difficulty levels, different opponent behaviours and provides visual rendering using Pygame.

Student Name: Ali Riyaz
Student Number: C3412624
Last Updated: 30/08/2025
"""

#TODO: Remember to fix the comments!!!!!!!!!!!!!!!!!!

#TODO: BRING BACK OPPONENT RELATED CODE ONCE WE GET A GOOD REWARD FUNCTION FOR THE ROBOT TO SCORE RELIABLY
"""TODO list of changes that were made to remove the opponent:
1) get_obs function: changed norm_opponent_x and y to 0.0 instead of the calculation
2) reset() function (Also commented out the while loop to ensure the 2 robots don't start in collision)
3) calculate_reward function
4) update_ball_physics function
5) check_terminated function
6) step function to include the update_opponent function
7) update_possession_flags function
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from .fieldconfig import FieldConfig


#TODO: Figure out the reward function calculations and scaling factors
#TODO: try to better understand and come up with the math for the rewards rather than just blindlhy aiming/vibing
#TODO: Figure out if the rewards should be in bounds like normalised. Johanne used a bell curve function to tell the robot that 1 action is good and the other is bad 
# between 0 and 1
# It should have a derivative at every point. Multiplied it by 2 and subtracted by 1 to make the bounds of the reward to be -1 and 1. 
# This was so that he could couple the pitch and yaw
# so that the robot wouldn't have the correct yaw but the wrong pitch
# This was to make sure that the robot doesn't "technically" do the right thing if it gets the value as 0, So he'll get either -1 or +1 (somehow, I forgot)
# NNeed to be smart about the rewards
#TODO: Look up research papers on reward functions. Maybe to see if the total reward as a whole is also normalised or scaled?????

#TODO: Ask Claude how to test the performance of models for evaluation and testing for FYP as a whole (probably won't do deployment or C++ integration)

class ActionSmoothingWrapper:
    """Wrapper to smooth model actions - no retraining needed"""

    def __init__(self, model, smoothing_factor=0.7):
        self.model = model
        self.smoothing_factor = smoothing_factor
        self.prev_action = np.array([0.0, 0.0, 0.0])

    def predict(self, obs, deterministic=True):
        # Get raw action from trained model
        raw_action, state = self.model.predict(obs, deterministic=deterministic)

        # Smooth the action with previous action
        smoothed_action = (self.smoothing_factor * self.prev_action +
                          (1 - self.smoothing_factor) * raw_action)

        # Store for next iteration
        self.prev_action = smoothed_action.copy()

        return smoothed_action, state

    def reset(self):
        """Reset action history for new episode"""
        self.prev_action = np.array([0.0, 0.0, 0.0])


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
        """Creates smooth transition from 0 to 1 around threshold using sigmoid."""
        if sharpness is None:
            sharpness = self.transition_sharpness
        return 1.0 / (1.0 + np.exp(-sharpness * (threshold - x)))

    def gaussian_reward(self, distance: float, optimal: float = 0.0, sigma: float = 1.0) -> float:
        """Gaussian-shaped reward centered at optimal distance."""
        return np.exp(-0.5 * ((distance - optimal) / sigma) ** 2)
    
class SoccerEnv(gym.Env):
    """
    Simple 2D Soccer Environment for FYP
    - Robot needs to dribble ball around opponent to goal
    - Positions of the ball and 2 robots randomised at the start of a new episode
    - 
    """
    
    # Constructor to initialise the environment including action, observation spaces along with state variables
    def __init__(self, render_mode=None, difficulty="easy", config_path="field_config.yaml", testing_mode=False, reward_type="original"):
        super().__init__()

        # Load field configuration
        self.field_config = FieldConfig(config_path)
        self.difficulty = difficulty
        self.testing_mode = testing_mode
        self.reward_type = reward_type  # "original", "smooth", or "hybrid"

        # Initialize smooth reward function if needed
        if self.reward_type in ["smooth", "hybrid"]:
            self.smooth_reward_fn = SmoothRewardFunction()
            self.smooth_reward_fn.prev_ball_goal_distance = None
        
        # Set up field dimensions from config
        self.field_width = self.field_config.field_width_pixels # X-axis
        self.field_height = self.field_config.field_height_pixels # Y-axis
        self.goal_width = self.field_config.goal_width_pixels
        
        # Action space: [forward/back, left/right, rotation]
        # Action space: [x_velocity, y_velocity, angular_velocity] in WORLD coordinates
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

        # Cache frequently used calculations
        self._max_distance = np.sqrt(self.field_width**2 + self.field_height**2)
        self._goal_center = np.array([self.field_width, self.field_height // 2])
        self._field_center = np.array([self.field_width // 2, self.field_height // 2])

        # Robot movement parameters - adjusted for testing mode
        # base_robot_speed = 1.8 # Used tp be 2.5 (MIGHT BE IN PIXELS per FRAME not m/s)
        base_robot_speed_mps = self.field_config.config['robot_parameters']['base_robot_speed_mps'] # (THis is in m/s)
        base_rotation_speed_rps = self.field_config.config['robot_parameters']['base_rotation_speed_rps'] # Used to be 0.08 but not in radians per second BTW THIS IS 1 full rotation/1.5sec

        # Convert to pixels per frame for internal use
        self.robot_speed = base_robot_speed_mps * self.field_config.pixels_per_meter * self.dt
        self.robot_rotation_speed = base_rotation_speed_rps * self.dt

        # Calculate opponent speed (similar to robot speed calculation)
        base_opponent_speed_mps = self.field_config.config['robot_parameters']['base_opponent_speed_mps']
        opponent_speed_mps = base_opponent_speed_mps * self.opponent_speed

        # Convert to pixels per frame for internal use
        self.opponent_speed_pixels_per_frame = opponent_speed_mps * self.field_config.pixels_per_meter * self.dt


        # Apply testing mode speed multiplier
        if self.testing_mode or self.field_config.config.get('rendering', {}).get('testing_mode', False):
            speed_multiplier = self.field_config.config.get('rendering', {}).get('testing_speed_multiplier', 2.0)
            self.robot_speed = base_robot_speed_mps * speed_multiplier * self.field_config.pixels_per_meter * self.dt # should be in pixels/frame
            self.robot_rotation_speed = base_rotation_speed_rps * speed_multiplier * self.dt # Should be rad/frame
            opponent_speed_mps = opponent_speed_mps * speed_multiplier

            print(f"🚀 Testing mode enabled: {speed_multiplier}x speed")

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

        #TODO: Make the robot's position to be at the center of the field
        # self.robot_pos = np.array([
        #     self.field_width / 2,
        #     self.field_height / 2
        # ], dtype=np.float64)

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
        
        # # TODO: from list of changes to remove opponent # Randomize opponent position (right third of field)
        # self.opponent_pos = np.array([
        #     np.random.uniform(self.field_width * 0.6, self.field_width - self.robot_radius - 20),
        #     np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        # ], dtype=np.float64)
        
        # Add small random initial ball velocity for realism
        if np.random.random() < 0.3:  # 30% chance of initial movement
            self.ball_vel = np.random.uniform(-0.5, 0.5, 2)

        # Goal position (right side, center)
        self.goal_pos = np.array([self.field_width, self.field_height // 2])
        
        # Get training progress (you'll need to pass this from your training script)
        training_timesteps = getattr(self, 'relative_timesteps', 0)
        print(f"🔍 TEST: Current timesteps = {training_timesteps}")

        #TODO: WIP This is for no opponent
        # if training_timesteps < 200000:
        #     # Phase 1: No opponent - robot learns to score freely
        #     self.opponent_enabled = False
        #     self.opponent_pos = np.array([self.field_width + 100, self.field_height + 100])  # Off-field
        # else:
        # Phase 2+: Normal opponent

        self.opponent_enabled = False
        #TODO: CHeck whether any opponent pos or velocity impacts the behaviours or reward Function
        self.opponent_pos = np.array([self.field_width + 100, self.field_height + 100])  # Off-field

        #NOTE: THIS IS TEMPORARY 
        # # Ensure minimum distances to avoid starting in collision
        # while np.linalg.norm(self.robot_pos - self.opponent_pos) < 50:
        #     self.opponent_pos = np.array([
        #         np.random.uniform(self.field_width * 0.6, self.field_width - self.robot_radius - 20),
        #         np.random.uniform(self.robot_radius + 20, self.field_height - self.robot_radius - 20)
        #     ])

        self._ball_push_momentum = np.array([0.0, 0.0])
        # Reset game state
        self.has_ball = False
        self.opponent_has_ball = False
        self.steps = 0
        self.robot_possession_time = 0
        self.opponent_possession_time = 0

        # Reset progress tracking variables to prevent cross-episode contamination when calculating rewards
        if hasattr(self, '_prev_ball_to_goal_distance'):
            delattr(self, '_prev_ball_to_goal_distance')
        if hasattr(self, '_prev_robot_to_goal_distance'):
            delattr(self, '_prev_robot_to_goal_distance')
        if hasattr(self, '_prev_robot_angle'):
            delattr(self, '_prev_robot_angle')

        # Reset opponent behavior
        self.opponent_behavior = np.random.choice(['aggressive', 'defensive', 'balanced'])        
        return self._get_obs(), {}
    
    # Function to 
    def _get_obs(self) -> np.array:
        # Normalize positions to [-1, 1] range
        norm_robot_x = np.clip((2 * self.robot_pos[0] / self.field_width) - 1, -1, 1)
        norm_robot_y = np.clip((2 * self.robot_pos[1] / self.field_height) - 1, -1, 1)
        norm_ball_x = np.clip((2 * self.ball_pos[0] / self.field_width) - 1, -1, 1)
        norm_ball_y = np.clip((2 * self.ball_pos[1] / self.field_height) - 1, -1, 1)

        #TODO: BRING BACK OPPONENT RELATED CODE ONCE WE GET A GOOD REWARD FUNCTION FOR THE ROBOT TO SCORE RELIABLY
        # norm_opponent_x = np.clip((2 * self.opponent_pos[0] / self.field_width) - 1, -1, 1)
        # norm_opponent_y = np.clip((2 * self.opponent_pos[1] / self.field_height) - 1, -1, 1)
        norm_opponent_x = 0.0
        norm_opponent_y = 0.0
        
        # Normalize velocities # should hopefully be in pixels/fame, pls verify
        # This came from manually mathing that max value of forward_x or strafe_x is 4.175 pixels/frame (from init method calculating self.robot_speed 
        # self_robot_speed = base_robot_speed_mps * self.field_config.pixels_per_meter * self.dt = 2.5 × 100 × 0.0167  = 4.175 pixels/frame), then add forward_x and strafe_x, max becomes 4.175 + 4.175 = 8.35
        # which is self.robot_speed times 2?
        max_vel = self.robot_speed * 2.0 
        #TODO: Check whether to normalise component or normalise magnitude
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

        # # Opponent AI's behavior (can be offensive, defensive, or balanced)
        # self._update_opponent()
        
        # Update possession flags based on proximity (transition period)
        self._update_possession_flags()

        # Calculate reward based on selected reward type
        if self.reward_type == "smooth":
            reward = self._calculate_smooth_reward()
        elif self.reward_type == "hybrid":
            reward = self._calculate_hybrid_reward()
        else:
            reward = self._calculate_reward()  # Original reward function

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
        """
        Apply robot action using WORLD COORDINATE DYNAMICS
        
        States: (x, y, θ)
        Actions: (ẋ, ẏ, θ̇) - world frame velocities
        
        Discrete dynamics:
        x[k+1] = x[k] + ẋ * dt
        y[k+1] = y[k] + ẏ * dt  
        θ[k+1] = θ[k] + θ̇ * dt
        
        Angle Convention (Standard Mathematical):
        - θ = 0: Robot faces +X direction (right/toward goal)
        - θ = π/2: Robot faces +Y direction (down in screen coords)
        - θ = π: Robot faces -X direction (left/away from goal)  
        - θ = 3π/2: Robot faces -Y direction (up in screen coords)
        - Positive angular velocity (θ̇ > 0): Counter-clockwise rotation
        - Negative angular velocity (θ̇ < 0): Clockwise rotation
        """
        # Validate and clip action values
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        # Parse action as world frame velocities
        x_velocity_action = action[0]  # World X velocity command [-1, 1]
        y_velocity_action = action[1]  # World Y velocity command [-1, 1]  
        angular_velocity_action = action[2]  # Angular velocity command [-1, 1]
        
        # Convert action commands to actual velocities
        max_linear_speed = self.robot_speed  # pixels/frame
        max_angular_speed = self.robot_rotation_speed  # rad/frame
        
        # World Coordinate Velocities
        world_x_velocity = x_velocity_action * max_linear_speed
        world_y_velocity = y_velocity_action * max_linear_speed
        angular_velocity = angular_velocity_action * max_angular_speed
        
        # Store velocities for observation
        self.robot_vel = np.array([world_x_velocity, world_y_velocity])
        
        # Discrete Dynamics Update
        # x[k+1] = x[k] + ẋ * dt (dt=1 since velocities already in pixels/frame)
        new_x = self.robot_pos[0] + world_x_velocity
        new_y = self.robot_pos[1] + world_y_velocity
        new_angle = self.robot_angle + angular_velocity
        
        # Normalize angle to [0, 2π)
        new_angle = new_angle % (2 * np.pi)
        
        # Keep robot within bounds
        new_x = np.clip(new_x, self.robot_radius, self.field_width - self.robot_radius)
        new_y = np.clip(new_y, self.robot_radius, self.field_height - self.robot_radius)
        
        # Update state
        self.robot_pos = np.array([new_x, new_y])
        self.robot_angle = new_angle
        
        # DEBUG: Print angle information (remove after verification)
        if hasattr(self, '_debug_angles') and self._debug_angles:
            angle_deg = np.degrees(self.robot_angle) % 360
            print(f"🧭 Robot angle: {angle_deg:.1f}° ({self.robot_angle:.3f} rad) | "
                  f"Action angular_vel: {angular_velocity_action:+.2f}")
    
    #TODO: Deal with opponent's velocity as well in the init method 
    def _update_opponent(self):
        """Enhanced opponent AI with different behaviors"""
        # At start of _update_opponent():
        if not getattr(self, 'opponent_enabled', True):
            return  # Skip opponent updates if disabled

        if not hasattr(self, 'opponent_behavior'):
            self.opponent_behavior = 'balanced'
        
        # # Store previous position to calculate velocity
        # prev_pos = self.opponent_pos.copy()

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

        speed = self.opponent_speed_pixels_per_frame       

        # Add some randomness to make opponent less predictable
        if np.random.random() < 0.1:  # 10% chance for random movement
            random_angle = np.random.uniform(0, 2 * np.pi)
            direction += 0.3 * np.array([np.cos(random_angle), np.sin(random_angle)])
            direction = direction / np.linalg.norm(direction)
        
        # Calculate new position
        new_pos = self.opponent_pos + direction * speed
        
        # Keep opponent within bounds with margin
        margin = self.robot_radius + 10
        new_pos[0] = np.clip(new_pos[0], margin, self.field_width - margin)
        new_pos[1] = np.clip(new_pos[1], margin, self.field_height - margin)
        
        # Calculate velocity based on position change
        self.opponent_vel = (new_pos - self.opponent_pos) / self.dt
        
        # Update position
        self.opponent_pos = new_pos

    def _calculate_reward(self):
        """
        Multi-objective continuous reward function with anti-exploitation measures.

        Mathematical Components:
        1. Sigmoid: σ(x) = 1/(1 + e^(-k(x-threshold))) for smooth transitions
        2. Gaussian: G(x) = e^(-0.5((x-μ)/σ)²) for optimal positioning
        3. Exponential: e^(-x/scale) for distance-based penalties
        4. Potential: Φ(s') - Φ(s) for policy-invariant shaping

        References:
        1. "Policy Invariance Under Reward Transformations" (Ng et al., 1999)
        2. "Deep Reinforcement Learning with Smooth Policy" (Gu et al., 2016)
        3. "Multi-Objective Reinforcement Learning" (Roijers & Whiteson, 2017)
        4. "The Mirage of Action-Dependent Baselines" (Tucker et al., 2018)
        """
        # Load all parameters from config
        reward_params = self.field_config.config.get('reward_parameters', {})
        robot_params = self.field_config.config.get('robot_parameters', {})
        strategic_zones = self.field_config.config.get('strategic_zones', {})

        # Multi-objective weights
        possession_weight = reward_params.get('possession_weight', 2.0)
        goal_weight = reward_params.get('goal_weight', 3.0)
        efficiency_weight = reward_params.get('efficiency_weight', 1.5)
        safety_weight = reward_params.get('safety_weight', 1.0)

        # Transition parameters
        possession_sharpness = reward_params.get('possession_sharpness', 8.0)
        zone_sharpness = reward_params.get('zone_sharpness', 5.0)
        avoidance_sharpness = reward_params.get('avoidance_sharpness', 10.0)

        # Convert thresholds from meters to pixels
        contact_threshold = self.field_config.meters_to_pixels(robot_params.get('contact_threshold', 0.25))
        possession_threshold = self.field_config.meters_to_pixels(robot_params.get('possession_threshold', 0.4))
        collision_threshold = self.field_config.meters_to_pixels(reward_params.get('collision_distance_threshold', 0.3))

        # Initialize reward components tracking
        reward = 0.0
        self.reward_components = {}  # Track individual components for debugging

        # Calculate key distances and states
        robot_ball_distance = np.linalg.norm(self.robot_pos - self.ball_pos)
        goal_center = self._goal_center
        ball_to_goal_distance = np.linalg.norm(self.ball_pos - goal_center)
        robot_to_goal_distance = np.linalg.norm(self.robot_pos - goal_center)

        # Curriculum learning - check if opponent is enabled
        opponent_enabled = getattr(self, 'opponent_enabled', True)
        if opponent_enabled:
            opponent_ball_distance = np.linalg.norm(self.opponent_pos - self.ball_pos)
            robot_opponent_distance = np.linalg.norm(self.robot_pos - self.opponent_pos)
        else:
            opponent_ball_distance = float('inf')
            robot_opponent_distance = float('inf')

        # === TERMINAL OUTCOMES (Discrete for rare events) ===
        if self._check_goal():
            if reward_params.get('debug_prints_enabled', False):
                print(f"🎯 GOAL SCORED! Reward: {reward_params.get('goal_scored_reward', 150.0)}")
            return reward_params.get('goal_scored_reward', 150.0)

        if self._check_opponent_goal() and opponent_enabled:
            if reward_params.get('debug_prints_enabled', False):
                print(f"😱 OPPONENT SCORED! Penalty: {reward_params.get('opponent_goal_penalty', -100.0)}")
            return reward_params.get('opponent_goal_penalty', -100.0)

        if self._check_ball_out_of_play():
            if reward_params.get('debug_prints_enabled', False):
                print(f"⚽ BALL OUT! Penalty: {reward_params.get('ball_out_bounds_penalty', -50.0)}")
            return reward_params.get('ball_out_bounds_penalty', -50.0)

        # ============================================================
        # COMPONENT 1: BALL POSSESSION (Continuous Sigmoid)
        # Mathematical: σ(x) = 1/(1 + e^(-k(threshold - distance)))
        # ============================================================
        possession_confidence = 1.0 / (1.0 + np.exp(-possession_sharpness * (possession_threshold - robot_ball_distance)))
        self.reward_components['possession_confidence'] = possession_confidence

        # Gaussian reward for optimal ball distance (peaks at contact)
        optimal_ball_sigma = contact_threshold * reward_params.get('optimal_ball_sigma_multiplier', 1.5)
        ball_proximity_reward = np.exp(-0.5 * ((robot_ball_distance / optimal_ball_sigma) ** 2))
        possession_reward = possession_weight * ball_proximity_reward * 10.0
        reward += possession_reward
        self.reward_components['ball_proximity'] = possession_reward

        # ============================================================
        # COMPONENT 2: GOAL PROGRESS (Potential-Based Shaping)
        # Mathematical: Φ(s') - Φ(s) where Φ(s) = -distance_to_goal
        # Policy invariant according to Ng et al., 1999
        # ============================================================
        if reward_params.get('use_potential_shaping', True):
            # Initialize potential if not exists
            if not hasattr(self, '_prev_potential'):
                self._prev_potential = -ball_to_goal_distance

            current_potential = -ball_to_goal_distance
            potential_diff = reward_params.get('potential_discount', 0.99) * current_potential - self._prev_potential
            potential_reward = goal_weight * potential_diff * reward_params.get('potential_scale', 10.0)

            # Scale by possession confidence (more reward when in control)
            potential_reward *= (0.5 + 0.5 * possession_confidence)
            reward += potential_reward
            self.reward_components['goal_progress'] = potential_reward
            self._prev_potential = current_potential

        # Exponential reward for goal approach
        goal_approach_scale = self.field_config.meters_to_pixels(reward_params.get('goal_approach_scale', 2.0))
        goal_approach_reward = goal_weight * np.exp(-ball_to_goal_distance / goal_approach_scale) * 5.0
        reward += goal_approach_reward * possession_confidence
        self.reward_components['goal_approach'] = goal_approach_reward * possession_confidence

        # ============================================================
        # COMPONENT 3: OPPONENT AVOIDANCE (Smooth Exponential)
        # Mathematical: -e^(-distance/scale) for smooth repulsion
        # ============================================================
        if opponent_enabled and robot_opponent_distance < collision_threshold * 3:
            collision_scale = self.field_config.meters_to_pixels(reward_params.get('collision_decay_scale', 0.3))
            collision_penalty = -safety_weight * 5.0 * np.exp(-robot_opponent_distance / collision_scale)

            # Reduce penalty in attacking zone with possession
            robot_x_fraction = self.robot_pos[0] / self.field_width
            attacking_third_start = strategic_zones.get('attacking_third_start', 0.67)
            attacking_confidence = 1.0 / (1.0 + np.exp(-zone_sharpness * (robot_x_fraction - attacking_third_start)))

            # Courage factor: reduce collision penalty when attacking with ball
            courage_factor = 1.0 - (attacking_confidence * possession_confidence * 0.7)
            collision_penalty *= courage_factor

            reward += collision_penalty
            self.reward_components['collision_avoidance'] = collision_penalty

            # Competitive factor (smooth tanh transition)
            competitive_diff = opponent_ball_distance - robot_ball_distance
            competitive_factor = np.tanh(competitive_diff / contact_threshold)
            competitive_reward = possession_weight * competitive_factor * 2.0
            reward += competitive_reward
            self.reward_components['competitive_factor'] = competitive_reward

        # ============================================================
        # COMPONENT 4: MOVEMENT EFFICIENCY (Anti-Exploitation)
        # ============================================================
        robot_speed = np.linalg.norm(self.robot_vel)
        robot_angular_speed = abs(getattr(self, '_prev_robot_angle', self.robot_angle) - self.robot_angle)
        self._prev_robot_angle = self.robot_angle

        # Anti-spinning penalty (continuous sigmoid transition)
        spin_threshold = reward_params.get('spin_detection_threshold', 0.2)
        rotation_threshold = reward_params.get('spin_rotation_threshold', 0.5)

        if robot_speed < spin_threshold and robot_angular_speed > rotation_threshold:
            spin_penalty = -efficiency_weight * reward_params.get('spinning_penalty', 7.0)
            spin_penalty *= (1.0 - np.tanh(robot_speed / spin_threshold))  # Smooth transition
            reward += spin_penalty
            self.reward_components['anti_spinning'] = spin_penalty

        # Movement toward objective (continuous alignment)
        if robot_speed > 0.1:
            # When has possession, reward movement toward goal
            if possession_confidence > 0.5:
                goal_direction = goal_center - self.robot_pos
                if np.linalg.norm(goal_direction) > 1e-6:
                    goal_direction_norm = goal_direction / np.linalg.norm(goal_direction)
                    robot_velocity_norm = self.robot_vel / robot_speed
                    alignment = np.dot(robot_velocity_norm, goal_direction_norm)
                    movement_reward = efficiency_weight * np.tanh(2.0 * alignment) * 3.0
                    reward += movement_reward
                    self.reward_components['goal_alignment'] = movement_reward
            else:
                # When seeking ball, reward movement toward it
                ball_direction = self.ball_pos - self.robot_pos
                if np.linalg.norm(ball_direction) > 1e-6:
                    ball_direction_norm = ball_direction / np.linalg.norm(ball_direction)
                    robot_velocity_norm = self.robot_vel / robot_speed
                    alignment = np.dot(robot_velocity_norm, ball_direction_norm)
                    movement_reward = efficiency_weight * np.tanh(2.0 * alignment) * 2.0
                    reward += movement_reward
                    self.reward_components['ball_seeking'] = movement_reward

        # ============================================================
        # COMPONENT 5: ANTI-EXPLOITATION PENALTIES
        # ============================================================

        # Prevent ball holding (stationary with ball)
        if possession_confidence > 0.8:
            ball_speed = np.linalg.norm(self.ball_vel)
            if not hasattr(self, '_ball_holding_timer'):
                self._ball_holding_timer = 0

            if ball_speed < 0.1 and robot_speed < 0.2:
                self._ball_holding_timer += 1
                holding_threshold = reward_params.get('ball_holding_time_threshold', 20)
                if self._ball_holding_timer > holding_threshold:
                    holding_penalty = -reward_params.get('ball_holding_penalty_rate', 0.5)
                    holding_penalty *= min((self._ball_holding_timer - holding_threshold) / 10.0,
                                          reward_params.get('ball_holding_max_penalty', 10.0))
                    reward += holding_penalty
                    self.reward_components['ball_holding_penalty'] = holding_penalty
            else:
                self._ball_holding_timer = 0

        # Prevent wall hugging (smooth exponential penalty)
        wall_distance = self.field_config.meters_to_pixels(reward_params.get('wall_hugging_distance', 0.3))
        edge_distances = [
            self.robot_pos[0],
            self.field_width - self.robot_pos[0],
            self.robot_pos[1],
            self.field_height - self.robot_pos[1]
        ]
        min_edge_distance = min(edge_distances)

        if min_edge_distance < wall_distance:
            wall_scale = self.field_config.meters_to_pixels(reward_params.get('wall_decay_scale', 0.5))
            wall_penalty = -reward_params.get('wall_hugging_penalty_rate', 1.0) * np.exp(-min_edge_distance / wall_scale)

            # Track wall hugging time
            if not hasattr(self, '_wall_hugging_timer'):
                self._wall_hugging_timer = 0
            self._wall_hugging_timer += 1

            if self._wall_hugging_timer > reward_params.get('wall_hugging_time_threshold', 10):
                wall_penalty *= reward_params.get('wall_hugging_penalty_multiplier', 2.0)

            reward += wall_penalty
            self.reward_components['wall_penalty'] = wall_penalty
        else:
            self._wall_hugging_timer = 0

        # Prevent backward movement with ball
        if possession_confidence > 0.7:
            ball_velocity_x = self.ball_vel[0] if len(self.ball_vel) > 0 else 0
            if ball_velocity_x < reward_params.get('backward_threshold', -0.1):
                backward_penalty = -reward_params.get('backward_movement_penalty', 2.0) * abs(ball_velocity_x)
                reward += backward_penalty
                self.reward_components['backward_penalty'] = backward_penalty

        # Prevent aimless wandering (check progress periodically)
        if not hasattr(self, '_progress_check_counter'):
            self._progress_check_counter = 0
            self._last_check_position = self.robot_pos.copy()

        self._progress_check_counter += 1
        check_interval = reward_params.get('progress_check_interval', 50)

        if self._progress_check_counter >= check_interval:
            distance_traveled = np.linalg.norm(self.robot_pos - self._last_check_position)
            min_progress = self.field_config.meters_to_pixels(reward_params.get('min_progress_distance', 0.5))

            if distance_traveled < min_progress:
                wandering_penalty = -reward_params.get('no_progress_penalty', 5.0)
                reward += wandering_penalty
                self.reward_components['no_progress_penalty'] = wandering_penalty

            self._progress_check_counter = 0
            self._last_check_position = self.robot_pos.copy()

        # ============================================================
        # COMPONENT 6: STRATEGIC POSITIONING (Gaussian Optimal Zones)
        # ============================================================
        robot_x_fraction = self.robot_pos[0] / self.field_width
        robot_y_fraction = self.robot_pos[1] / self.field_height

        # Smooth zone rewards using sigmoid transitions
        attacking_third_start = strategic_zones.get('attacking_third_start', 0.67)
        zone_confidence = 1.0 / (1.0 + np.exp(-zone_sharpness * (robot_x_fraction - attacking_third_start)))

        # Gaussian reward for optimal shooting position
        optimal_x = 0.8  # 80% of field length
        optimal_y = 0.5  # Center of field
        position_sigma = reward_params.get('optimal_position_sigma', 0.8)

        x_diff = (robot_x_fraction - optimal_x) / position_sigma
        y_diff = (robot_y_fraction - optimal_y) / position_sigma
        optimal_position_reward = np.exp(-0.5 * (x_diff**2 + y_diff**2))

        strategic_reward = goal_weight * zone_confidence * optimal_position_reward * possession_confidence * 3.0
        reward += strategic_reward
        self.reward_components['strategic_positioning'] = strategic_reward

        # ============================================================
        # COMPONENT 7: CURRICULUM LEARNING ADJUSTMENT
        # ============================================================
        if not opponent_enabled:
            # Solo training bonus to encourage exploration
            solo_bonus_multiplier = reward_params.get('solo_training_bonus', 1.2)
            reward *= solo_bonus_multiplier
            self.reward_components['solo_training_bonus'] = reward * (solo_bonus_multiplier - 1.0)

        # ============================================================
        # COMPONENT 8: TIME PENALTY (Efficiency)
        # ============================================================
        time_penalty = -reward_params.get('time_step_penalty', 0.01)
        reward += time_penalty
        self.reward_components['time_penalty'] = time_penalty

        # ============================================================
        # DEBUG PRINTING
        # ============================================================
        if reward_params.get('debug_prints_enabled', False):
            if self.steps % reward_params.get('debug_print_interval', 100) == 0:
                print(f"\n=== Reward Debug (Step {self.steps}) ===")
                print(f"Total Reward: {reward:.3f}")
                print(f"Possession Confidence: {possession_confidence:.3f}")
                print(f"Robot-Ball Distance: {robot_ball_distance:.2f}")
                print(f"Ball-Goal Distance: {ball_to_goal_distance:.2f}")
                if reward_params.get('reward_component_tracking', True):
                    print("\nComponent Breakdown:")
                    for comp_name, comp_value in self.reward_components.items():
                        if abs(comp_value) > 0.01:  # Only show non-zero components
                            print(f"  {comp_name}: {comp_value:.3f}")

        # ============================================================
        # FINAL PROCESSING: SOFT CLIPPING
        # ============================================================
        if not np.isfinite(reward):
            reward = reward_params.get('invalid_state_penalty', -0.5)
            if reward_params.get('debug_prints_enabled', False):
                print(f"WARNING: Non-finite reward detected, using default: {reward}")

        # Soft clipping using tanh for continuity
        if reward_params.get('use_soft_clipping', True):
            clip_scale = reward_params.get('soft_clip_scale', 20.0)
            reward = clip_scale * np.tanh(reward / clip_scale)
        else:
            # Hard clipping as fallback
            reward_min = reward_params.get('reward_min_bound', -30.0)
            reward_max = reward_params.get('reward_max_bound', 150.0)
            reward = np.clip(reward, reward_min, reward_max)

        return float(reward)

    def _calculate_smooth_reward(self):
        """
        Enhanced smooth continuous reward with comprehensive anti-exploitation measures.

        Mathematical Framework:
        - Sigmoid transitions: σ(x) = 1/(1 + e^(-k(x-c))) for smooth state changes
        - Gaussian distributions: G(x,μ,σ) = e^(-0.5((x-μ)/σ)²) for optimal zones
        - Exponential decay: e^(-x/λ) for distance-based penalties
        - Hyperbolic tangent: tanh(x) for bounded smooth transitions
        - Logarithmic growth: log(1 + x) for diminishing returns

        Anti-Exploitation Coverage:
        1. Stationary: Velocity-based penalties, progress tracking
        2. Spinning: Angular velocity monitoring with linear velocity check
        3. Wall-hugging: Exponential boundary penalties
        4. Ball-holding: Time-based escalating penalties
        5. Backward movement: Directional penalties
        6. Oscillation: Pattern detection via history
        7. Time-wasting: Decreasing time bonus

        References:
        - "Reward Shaping in RL" (Mataric, 1994)
        - "Intrinsic Motivation Systems" (Oudeyer et al., 2007)
        - "Curiosity-driven Exploration" (Pathak et al., 2017)
        - "Anti-Gaming Mechanisms in ML" (Amodei et al., 2016)
        """
        # Load configuration parameters
        reward_params = self.field_config.config.get('reward_parameters', {})
        robot_params = self.field_config.config.get('robot_parameters', {})

        # Core weights for multi-objective optimisation (reduced for better scaling)
        w_possess = reward_params.get('smooth_possession_weight', 1.0)
        w_goal = reward_params.get('smooth_goal_weight', 1.5)
        w_control = reward_params.get('smooth_control_weight', 2.0)
        w_safety = reward_params.get('smooth_safety_weight', 3.0)  # Increased for stronger penalties
        w_progress = reward_params.get('smooth_progress_weight', 2.0)

        # Shaping parameters
        k_sharp = reward_params.get('smooth_sharpness', 10.0)  # Sigmoid sharpness
        σ_zone = reward_params.get('smooth_zone_sigma', 0.15)  # Gaussian sigma
        λ_decay = reward_params.get('smooth_decay_rate', 0.3)  # Exponential decay

        # Distance thresholds (in pixels)
        contact_dist = self.field_config.meters_to_pixels(0.2)
        possess_dist = self.field_config.meters_to_pixels(0.35)
        danger_dist = self.field_config.meters_to_pixels(0.4)

        # Initialize tracking
        reward = 0.0
        self.reward_components = {}

        # State calculations
        robot_ball_vec = self.ball_pos - self.robot_pos
        robot_ball_dist = np.linalg.norm(robot_ball_vec)
        goal_pos = self._goal_center
        ball_goal_vec = goal_pos - self.ball_pos
        ball_goal_dist = np.linalg.norm(ball_goal_vec)
        robot_goal_dist = np.linalg.norm(goal_pos - self.robot_pos)

        # Velocity magnitudes
        robot_speed = np.linalg.norm(self.robot_vel)
        ball_speed = np.linalg.norm(self.ball_vel)
        angular_speed = abs(getattr(self, '_last_angle', self.robot_angle) - self.robot_angle)
        self._last_angle = self.robot_angle

        # Opponent state (if enabled)
        opp_enabled = getattr(self, 'opponent_enabled', True)
        if opp_enabled:
            opp_ball_dist = np.linalg.norm(self.opponent_pos - self.ball_pos)
            robot_opp_dist = np.linalg.norm(self.opponent_pos - self.robot_pos)
        else:
            opp_ball_dist = float('inf')
            robot_opp_dist = float('inf')

        # ========== TERMINAL STATES (Immediate Returns) ==========
        if self._check_goal():
            return reward_params.get('smooth_goal_reward', 200.0)

        if opp_enabled and self._check_opponent_goal():
            return reward_params.get('smooth_opponent_goal_penalty', -150.0)

        if self._check_ball_out_of_play():
            return reward_params.get('smooth_out_bounds_penalty', -75.0)

        # ========== COMPONENT 1: POSSESSION DYNAMICS ==========
        # Smooth sigmoid for possession confidence
        possess_conf = 1.0 / (1.0 + np.exp(-k_sharp * (possess_dist - robot_ball_dist)))
        self.reward_components['possession_confidence'] = possess_conf

        # Gaussian reward for optimal contact distance (reduced magnitude)
        contact_reward = w_possess * 3.0 * np.exp(-0.5 * ((robot_ball_dist - contact_dist) / (contact_dist * 0.5))**2)
        reward += contact_reward * (1.0 + 0.5 * possess_conf)  # Boost when confident
        self.reward_components['contact_quality'] = contact_reward

        # Competitive advantage (smooth difference)
        if opp_enabled and opp_ball_dist < possess_dist * 2:
            compete_advantage = np.tanh((opp_ball_dist - robot_ball_dist) / contact_dist)
            compete_reward = w_possess * 5.0 * compete_advantage
            reward += compete_reward
            self.reward_components['competitive_edge'] = compete_reward

        # ========== COMPONENT 2: GOAL APPROACH DYNAMICS ==========
        # Potential-based shaping with decay (reduced magnitude)
        if not hasattr(self, '_smooth_potential'):
            self._smooth_potential = -ball_goal_dist

        potential_current = -ball_goal_dist
        potential_diff = 0.99 * potential_current - self._smooth_potential
        potential_reward = w_goal * potential_diff * 3.0 * (0.3 + 0.7 * possess_conf)
        reward += potential_reward
        self._smooth_potential = potential_current
        self.reward_components['goal_potential'] = potential_reward

        # Exponential goal proximity bonus (reduced)
        goal_proximity = w_goal * 2.0 * np.exp(-ball_goal_dist / (self.field_width * 0.3))
        reward += goal_proximity * possess_conf
        self.reward_components['goal_proximity'] = goal_proximity * possess_conf

        # ========== COMPONENT 3: MOVEMENT QUALITY ==========
        # Prevent stationary behaviour
        if robot_speed < 0.05:  # Nearly stationary
            if not hasattr(self, '_stationary_timer'):
                self._stationary_timer = 0
            self._stationary_timer += 1

            if self._stationary_timer > 5:
                # Escalating penalty (stronger)
                stationary_penalty = -w_control * 3.0 * min(self._stationary_timer / 5.0, 10.0)
                reward += stationary_penalty
                self.reward_components['stationary_penalty'] = stationary_penalty
        else:
            self._stationary_timer = 0

            # Reward purposeful movement
            if possess_conf > 0.5:
                # Moving toward goal with ball
                if ball_goal_dist > contact_dist:
                    goal_dir = ball_goal_vec / ball_goal_dist
                    vel_dir = self.robot_vel / robot_speed if robot_speed > 0 else np.zeros(2)
                    alignment = np.dot(goal_dir, vel_dir)
                    direction_reward = w_control * 4.0 * np.tanh(2.0 * alignment)
                    reward += direction_reward
                    self.reward_components['goal_alignment'] = direction_reward
            else:
                # Moving toward ball
                if robot_ball_dist > contact_dist:
                    ball_dir = robot_ball_vec / robot_ball_dist
                    vel_dir = self.robot_vel / robot_speed if robot_speed > 0 else np.zeros(2)
                    alignment = np.dot(ball_dir, vel_dir)
                    direction_reward = w_control * 3.0 * np.tanh(2.0 * alignment)
                    reward += direction_reward
                    self.reward_components['ball_seeking'] = direction_reward

        # ========== COMPONENT 4: ANTI-SPINNING ==========
        # Detect and penalise pure rotation without translation
        spin_ratio = angular_speed / (robot_speed + 0.01)  # Avoid division by zero
        if spin_ratio > 3.0:  # High rotation relative to movement (lower threshold)
            spin_penalty = -w_control * 5.0 * np.tanh(spin_ratio - 3.0)  # Stronger penalty
            reward += spin_penalty
            self.reward_components['anti_spin'] = spin_penalty

        # ========== COMPONENT 5: BOUNDARY PENALTIES ==========
        # Exponential penalties near walls
        wall_margin = self.field_config.meters_to_pixels(0.4)
        edge_dists = [
            self.robot_pos[0],  # Left wall
            self.field_width - self.robot_pos[0],  # Right wall
            self.robot_pos[1],  # Top wall
            self.field_height - self.robot_pos[1]  # Bottom wall
        ]
        min_edge = min(edge_dists)

        if min_edge < wall_margin:
            wall_penalty = -w_safety * 5.0 * np.exp(-min_edge / (wall_margin * 0.3))  # Stronger penalty

            # Track wall time for escalation
            if not hasattr(self, '_wall_timer'):
                self._wall_timer = 0
            self._wall_timer += 1

            if self._wall_timer > 10:  # Faster escalation
                wall_penalty *= min(3.0, 1.0 + self._wall_timer / 15.0)  # Stronger escalation

            reward += wall_penalty
            self.reward_components['wall_avoidance'] = wall_penalty
        else:
            self._wall_timer = 0

        # ========== COMPONENT 6: BALL CONTROL PENALTIES ==========
        # Prevent ball holding without progress
        if possess_conf > 0.7:
            if ball_speed < 0.1 and robot_speed < 0.1:
                if not hasattr(self, '_holding_timer'):
                    self._holding_timer = 0
                self._holding_timer += 1

                if self._holding_timer > 10:
                    hold_penalty = -w_progress * 1.0 * min(self._holding_timer / 20.0, 3.0)
                    reward += hold_penalty
                    self.reward_components['ball_holding'] = hold_penalty
            else:
                self._holding_timer = 0

            # Penalise backward ball movement
            ball_vel_x = self.ball_vel[0] if len(self.ball_vel) > 0 else 0
            if ball_vel_x < -0.1:  # Moving away from goal
                backward_penalty = -w_progress * 2.0 * abs(ball_vel_x)
                reward += backward_penalty
                self.reward_components['backward_ball'] = backward_penalty

        # ========== COMPONENT 7: COLLISION AVOIDANCE ==========
        if opp_enabled and robot_opp_dist < danger_dist * 1.5:
            # Exponential collision penalty
            collision_penalty = -w_safety * 4.0 * np.exp(-robot_opp_dist / (danger_dist * 0.5))

            # Reduce penalty when attacking with ball
            if possess_conf > 0.7 and self.robot_pos[0] > self.field_width * 0.6:
                collision_penalty *= 0.4  # Brave in attacking third

            reward += collision_penalty
            self.reward_components['collision_avoid'] = collision_penalty

        # ========== COMPONENT 8: PROGRESS TRACKING ==========
        # Track overall progress to prevent loops
        if not hasattr(self, '_progress_history'):
            self._progress_history = []
            self._last_progress_pos = self.robot_pos.copy()

        if len(self._progress_history) % 30 == 0:  # Check every 30 steps
            progress_dist = np.linalg.norm(self.robot_pos - self._last_progress_pos)
            min_expected = self.field_config.meters_to_pixels(0.3)

            if progress_dist < min_expected:
                loop_penalty = -w_progress * 2.0
                reward += loop_penalty
                self.reward_components['loop_detection'] = loop_penalty

            self._last_progress_pos = self.robot_pos.copy()

        self._progress_history.append(self.robot_pos.copy())
        if len(self._progress_history) > 100:
            self._progress_history.pop(0)

        # ========== COMPONENT 9: OSCILLATION DETECTION ==========
        # Detect back-and-forth patterns
        if len(self._progress_history) >= 10:
            recent_positions = self._progress_history[-10:]
            position_variance = np.var([pos[0] for pos in recent_positions])

            if position_variance < 10.0 and robot_speed > 0.1:  # Moving but not progressing
                oscillation_penalty = -w_control * 1.5
                reward += oscillation_penalty
                self.reward_components['oscillation'] = oscillation_penalty

        # ========== COMPONENT 10: TIME EFFICIENCY ==========
        # Logarithmic time penalty for diminishing returns
        time_factor = -w_progress * 0.02 * np.log(1 + self.steps / 100.0)
        reward += time_factor
        self.reward_components['time_efficiency'] = time_factor

        # ========== COMPONENT 11: STRATEGIC POSITIONING ==========
        # Gaussian reward for optimal field position
        robot_x_norm = self.robot_pos[0] / self.field_width
        robot_y_norm = self.robot_pos[1] / self.field_height

        # Optimal attacking position
        optimal_x = 0.75  # 3/4 field length
        optimal_y = 0.5   # Center line

        position_quality = np.exp(-0.5 * (((robot_x_norm - optimal_x) / σ_zone)**2 +
                                          ((robot_y_norm - optimal_y) / (σ_zone * 1.5))**2))

        strategic_reward = w_goal * 2.0 * position_quality * possess_conf
        reward += strategic_reward
        self.reward_components['strategic_position'] = strategic_reward

        # ========== FINAL PROCESSING ==========
        # Handle invalid states
        if not np.isfinite(reward):
            reward = -1.0

        # Smooth clipping with tanh (reduced magnitude for better discrimination)
        max_magnitude = 15.0
        reward = max_magnitude * np.tanh(reward / max_magnitude)

        # Debug output
        if reward_params.get('smooth_debug_enabled', False):
            if self.steps % 50 == 0:
                print(f"\n=== Smooth Reward (Step {self.steps}) ===")
                print(f"Total: {reward:.3f}")
                print(f"Possession: {possess_conf:.3f}, Ball Dist: {robot_ball_dist:.1f}")
                print(f"Goal Dist: {ball_goal_dist:.1f}, Robot Speed: {robot_speed:.2f}")
                for name, value in sorted(self.reward_components.items()):
                    if abs(value) > 0.01:
                        print(f"  {name}: {value:+.3f}")

        return float(reward)

    def _calculate_hybrid_reward(self):
        """
        Hybrid reward combining discrete events with continuous shaping.

        Strategy: Use discrete rewards for rare events (goals, out of bounds)
        and continuous shaping for common behaviours.

        This approach balances:
        - Clear objectives (discrete terminal rewards)
        - Smooth learning signals (continuous guidance)
        - Anti-exploitation measures (penalty accumulation)

        References:
        - "Hybrid Reward Architecture" (van Seijen et al., 2017)
        - "Combining Discrete and Continuous Rewards" (Baird, 1995)
        """
        reward_params = self.field_config.config.get('reward_parameters', {})

        # Check terminal conditions first (discrete)
        if self._check_goal():
            return reward_params.get('hybrid_goal_reward', 100.0)

        if self._check_opponent_goal():
            return reward_params.get('hybrid_opponent_goal', -75.0)

        if self._check_ball_out_of_play():
            return reward_params.get('hybrid_out_bounds', -25.0)

        # Otherwise use smooth continuous reward
        base_reward = self._calculate_smooth_reward()

        # Scale continuous portion
        hybrid_scale = reward_params.get('hybrid_continuous_scale', 0.5)
        return base_reward * hybrid_scale

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
            # Calculate push force based on robot velocity and distance
            overlap = contact_distance - robot_ball_distance
            
            # Direction of push (from robot center to ball center)
            push_direction = robot_to_ball / robot_ball_distance
            
            # Push force depends on robot velocity and overlap
            robot_speed = np.linalg.norm(self.robot_vel)
            push_force = (robot_speed * 0.3 + overlap * 0.2) * self.push_force_multiplier
            
            # NEW: Gradual force buildup prevents jerky movement
            if not hasattr(self, '_ball_push_momentum'):
                self._ball_push_momentum = np.array([0.0, 0.0])

            # Apply push to ball velocity
            push_velocity = push_direction * push_force

            #  # Smooth momentum change (prevents instant ball jumps)
            momentum_change_rate = 0.3  # Lower = smoother, higher = more responsive
            self._ball_push_momentum += (push_velocity - self._ball_push_momentum) * momentum_change_rate
            self.ball_vel += self._ball_push_momentum

            # self.ball_vel += push_velocity
            
            # Add some robot direction influence (robot can "guide" the ball)
            if robot_speed > 0.1:  # Only if robot is moving
                robot_direction = self.robot_vel / robot_speed
                guidance_strength = min(robot_speed * 0.15, 0.8)  # Limit guidance strength
                self.ball_vel += robot_direction * guidance_strength
        
        # # OPPONENT-BALL INTERACTION 
        # opponent_to_ball = self.ball_pos - self.opponent_pos
        # opponent_ball_distance = np.linalg.norm(opponent_to_ball)
        
        # if opponent_ball_distance < contact_distance and opponent_ball_distance > 0.1:
        #     overlap = contact_distance - opponent_ball_distance
        #     push_direction = opponent_to_ball / opponent_ball_distance
            
        #     # Opponent push force
        #     opponent_speed = np.linalg.norm(self.opponent_vel)  # Assume constant opponent "speed"
        #     push_force = (opponent_speed * 0.4 + overlap * 0.2) * self.push_force_multiplier
            
        #     push_velocity = push_direction * push_force
        #     self.ball_vel += push_velocity

        #     # Add opponent direction influence (similar to robot)
        #     if opponent_speed > 0.1:  # Only if opponent is moving
        #         opponent_direction = self.opponent_vel / opponent_speed
        #         guidance_strength = min(opponent_speed * 0.1, 0.5)  # Slightly weaker than robot
        #         self.ball_vel += opponent_direction * guidance_strength
            
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
        # elif opponent_ball_distance < self.possession_threshold:
        #     if robot_ball_distance > opponent_ball_distance * 1.2:  # Opponent is clearly closer
        #         self.has_ball = False
        #         self.opponent_has_ball = True
        #         self.opponent_possession_time += 1
        #         self.robot_possession_time = 0
        else:
            # Ball is loose
            self.has_ball = False
            self.opponent_has_ball = False
            # Don't reset possession times immediately - add some "memory"
            if self.robot_possession_time > 0:
                self.robot_possession_time = max(0, self.robot_possession_time - 1)
            # if self.opponent_possession_time > 0:
            #     self.opponent_possession_time = max(0, self.opponent_possession_time - 1)

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
        
        #TODO: REmoved this for now # Failure: collision with opponent
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
        field_rect = pygame.Rect(50, 50, int(self.field_width), int(self.field_height))
        pygame.draw.rect(self.window, field_green, field_rect)
        
        # Calculate line width based on field scale
        line_width = max(2, self.field_config.meters_to_pixels(0.05))
        
        # Draw field boundary
        pygame.draw.rect(self.window, line_white, 
                        (50, 50, int(self.field_width), int(self.field_height)), int(line_width))
        
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
        
        # Draw robot facing direction (world coordinate system)
        # Using standard mathematical convention:
        # θ=0 faces +X (right/toward goal), positive ω = counter-clockwise
        direction_length = self.robot_radius + 10
        direction_end_x = robot_x + direction_length * np.cos(self.robot_angle)
        direction_end_y = robot_y + direction_length * np.sin(self.robot_angle)  # Positive Y is down in screen coords
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
            text = self._font_small.render("ROBOT", True, possession_green)
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
        
        # # Draw opponent behavior indicator
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
        line_width = max(2, self.field_config.meters_to_pixels_int(0.05))
        
        # Field offset for drawing
        offset_x, offset_y = 50, 50
        
        # Center line
        center_x = self.field_width // 2 + offset_x
        pygame.draw.line(self.window, line_white, 
                        (center_x, offset_y), 
                        (center_x, self.field_height + offset_y), line_width)
        
        # Center circle
        center_radius = self.field_config.meters_to_pixels_int(
            self.field_config.real_dims['centre_circle_diameter'] / 2)
        pygame.draw.circle(self.window, line_white, 
                         (center_x, self.field_height // 2 + offset_y), 
                         center_radius, line_width)
        
        # Center mark
        pygame.draw.circle(self.window, line_white, 
                         (center_x, self.field_height // 2 + offset_y), 3)
        
        # Goal areas (both sides)
        goal_area_length = self.field_config.meters_to_pixels_int(
            self.field_config.real_dims['goal_area_length'])
        goal_area_width = self.field_config.meters_to_pixels_int(
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
        penalty_area_length = self.field_config.meters_to_pixels_int(
            self.field_config.real_dims['penalty_area_length'])
        penalty_area_width = self.field_config.meters_to_pixels_int(
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
        penalty_mark_distance = self.field_config.meters_to_pixels_int(
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