"""
SOCCER ENVIRONMENT FOLLOWING OFFICIAL GYMNASIUM TUTORIAL
========================================================

This follows the exact structure from:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

Adapted for a simple soccer dribbling scenario that matches your FYP requirements.

Key Components (following tutorial):
1. Import necessary modules
2. Define class inheriting from gym.Env
3. Define __init__ with spaces
4. Implement reset() method  
5. Implement step() method
6. Implement render() method (optional)
7. Register environment (optional)
"""

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class SoccerEnv(gym.Env):
    """
    Simple Soccer Environment following Gymnasium tutorial structure
    
    Description:
        A robot must dribble a ball around an opponent to reach a goal.
        This demonstrates the key concepts from your FYP:
        - Ball possession
        - Obstacle avoidance  
        - Goal-directed behavior
        - Reward engineering
    
    Observation Space:
        Type: Box(6,)
        Num    Observation               Min                  Max
        0      Robot X coordinate        0                    size-1
        1      Robot Y coordinate        0                    size-1  
        2      Ball X coordinate         0                    size-1
        3      Ball Y coordinate         0                    size-1
        4      Opponent X coordinate     0                    size-1
        5      Opponent Y coordinate     0                    size-1
        6      Has ball (0 or 1)         0                    1
    
    Action Space:
        Type: Discrete(4)
        Num   Action
        0     Move up
        1     Move down
        2     Move left  
        3     Move right
    
    Reward:
        Reward is given for:
        +100 for reaching goal with ball
        +1 for each step with ball possession
        +5 for moving closer to goal with ball
        -1 for losing ball
        -10 for colliding with opponent
        -0.1 for each timestep (encourages efficiency)
    
    Episode Termination:
        Episode terminates when:
        - Robot reaches goal with ball (success)
        - Robot collides with opponent (failure)
        - 200 timesteps elapsed (timeout)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # Size of the square grid
        self.window_size = 512  # Size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = spaces.Box(0, size - 1, shape=(7,), dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up  
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Get current observation following tutorial pattern"""
        return np.array([
            self._robot_location[0],
            self._robot_location[1],
            self._ball_location[0], 
            self._ball_location[1],
            self._opponent_location[0],
            self._opponent_location[1],
            float(self._has_ball)
        ], dtype=np.float32)

    def _get_info(self):
        """Get additional info following tutorial pattern"""
        return {
            "distance_to_goal": np.linalg.norm(self._robot_location - self._goal_location),
            "has_ball": self._has_ball,
            "steps": self._steps
        }

    def reset(self, seed=None, options=None):
        """Reset environment following tutorial pattern"""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the robot's location uniformly at random (left side)
        self._robot_location = np.array([0, self.np_random.integers(0, self.size)])

        # Ball starts near robot
        self._ball_location = self._robot_location + np.array([1, 0])
        self._ball_location = np.clip(self._ball_location, 0, self.size - 1)

        # Opponent in middle area
        self._opponent_location = np.array([
            self.np_random.integers(2, self.size - 2),
            self.np_random.integers(0, self.size)
        ])

        # Goal on right side
        self._goal_location = np.array([self.size - 1, self.size // 2])
        
        # Game state
        self._has_ball = False
        self._steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Execute one step following tutorial pattern"""
        self._steps += 1
        
        # Store previous state for reward calculation
        prev_distance = np.linalg.norm(self._robot_location - self._goal_location)
        prev_has_ball = self._has_ball

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.clip(
            self._robot_location + direction, 0, self.size - 1
        )
        
        # Check if new location collides with opponent
        if not np.array_equal(new_location, self._opponent_location):
            self._robot_location = new_location

        # Update ball possession
        distance_to_ball = np.linalg.norm(self._robot_location - self._ball_location)
        if distance_to_ball <= 1.0:
            self._has_ball = True
            self._ball_location = self._robot_location.copy()  # Ball follows robot
        else:
            # Ball stays where it was if robot moves away
            pass

        # Calculate reward
        reward = self._calculate_reward(prev_distance, prev_has_ball)

        # Check if episode terminates
        terminated = self._check_terminated()
        
        # Truncate if episode is too long
        truncated = self._steps >= 200

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_distance, prev_has_ball):
        """Calculate reward based on current state"""
        reward = 0.0
        
        # Time penalty (encourages efficiency)
        reward -= 0.1
        
        # Ball possession rewards
        if self._has_ball:
            reward += 1.0  # Continuous reward for maintaining ball possession
            
            # Progress toward goal reward
            current_distance = np.linalg.norm(self._robot_location - self._goal_location)
            if current_distance < prev_distance:
                reward += 5.0  # Bonus for getting closer to goal with ball
        else:
            if prev_has_ball:
                reward -= 1.0  # Penalty for losing ball possession
        
        # Check for goal reached (following tutorial pattern)
        if np.array_equal(self._robot_location, self._goal_location) and self._has_ball:
            reward += 100.0  # Big reward for reaching goal with ball
        
        # Check for collision
        if np.array_equal(self._robot_location, self._opponent_location):
            reward -= 10.0  # Penalty for hitting opponent
        
        return reward

    def _check_terminated(self):
        """Check if episode should terminate"""
        # Success: reached goal with ball
        if np.array_equal(self._robot_location, self._goal_location) and self._has_ball:
            return True
        
        # Failure: collision with opponent  
        if np.array_equal(self._robot_location, self._opponent_location):
            return True
            
        return False

    def render(self):
        """Render following tutorial pattern"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render frame following tutorial pattern"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the goal
        pygame.draw.rect(
            canvas,
            (255, 215, 0),  # Gold color for goal
            pygame.Rect(
                pix_square_size * self._goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw opponent 
        pygame.draw.circle(
            canvas,
            (255, 0, 0),  # Red for opponent
            (self._opponent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Draw ball
        ball_color = (0, 100, 0) if self._has_ball else (0, 0, 0)  # Green if possessed, black otherwise
        pygame.draw.circle(
            canvas,
            ball_color,
            (self._ball_location + 0.5) * pix_square_size,
            pix_square_size / 6,
        )

        # Now we draw the robot
        robot_color = (0, 0, 255) if not self._has_ball else (173, 216, 230)  # Blue or light blue
        pygame.draw.circle(
            canvas,
            robot_color,
            (self._robot_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Close rendering following tutorial pattern"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# TESTING FUNCTIONS (following tutorial examples)
def test_environment():
    """Test the environment with random actions (like tutorial)"""
    print("🎮 Testing Soccer Environment (Following Gymnasium Tutorial)")
    print("=" * 60)
    
    env = SoccerEnv(render_mode="human")
    observation, info = env.reset(seed=42)
    
    print(f"Initial observation: {observation}")
    print(f"Initial info: {info}")
    
    for i in range(100):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, Info={info}")

        if terminated or truncated:
            print(f"Episode finished after {i+1} steps")
            observation, info = env.reset()
            break

    env.close()

def train_on_custom_env():
    """Train agents on the custom environment"""
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    import matplotlib.pyplot as plt
    
    print("\n🧠 Training Agents on Custom Soccer Environment")
    print("=" * 50)
    
    # Create environment without rendering for faster training
    env = SoccerEnv()
    
    results = {}
    
    # Train DQN (since we have discrete actions)
    print("🚀 Training DQN...")
    dqn_model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
    dqn_model.learn(total_timesteps=100000)
    
    print("📊 Evaluating DQN...")
    dqn_mean, dqn_std = evaluate_policy(dqn_model, env, n_eval_episodes=20)
    results["DQN"] = (dqn_mean, dqn_std)
    print(f"DQN Results: {dqn_mean:.2f} ± {dqn_std:.2f}")
    
    # Train PPO
    print("\n🚀 Training PPO...")
    ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    ppo_model.learn(total_timesteps=100000)
    
    print("📊 Evaluating PPO...")
    ppo_mean, ppo_std = evaluate_policy(ppo_model, env, n_eval_episodes=20)
    results["PPO"] = (ppo_mean, ppo_std)
    print(f"PPO Results: {ppo_mean:.2f} ± {ppo_std:.2f}")
    
    # Plot comparison
    algorithms = list(results.keys())
    means = [results[algo][0] for algo in algorithms]
    stds = [results[algo][1] for algo in algorithms]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title("Algorithm Performance Comparison\nCustom Soccer Environment")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("soccer_env_algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 FINAL COMPARISON:")
    print(f"DQN: {dqn_mean:.2f} ± {dqn_std:.2f}")
    print(f"PPO: {ppo_mean:.2f} ± {ppo_std:.2f}")
    
    # Save models
    dqn_model.save("soccer_dqn")
    ppo_model.save("soccer_ppo")
    print("\n💾 Models saved as 'soccer_dqn.zip' and 'soccer_ppo.zip'")
    
    return results

def watch_trained_agent():
    """Watch a trained agent play (like tutorial examples)"""
    from stable_baselines3 import PPO
    
    try:
        print("👀 Loading trained agent and watching it play...")
        env = SoccerEnv(render_mode="human")
        model = PPO.load("soccer_ppo")
        
        obs, info = env.reset()
        for i in range(200):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i % 20 == 0:
                print(f"Step {i}: Reward={reward:.2f}, Has ball={info['has_ball']}")
            
            if terminated or truncated:
                print(f"Episode ended at step {i}")
                obs, info = env.reset()
        
        env.close()
        
    except FileNotFoundError:
        print("❌ No trained model found. Run train_on_custom_env() first.")

if __name__ == "__main__":
    # Test the environment
    print("Testing environment with random actions...")
    test_environment()
    
    # Uncomment to train agents
    print("\nTraining agents...")
    train_on_custom_env()
    
    # Uncomment to watch trained agent
    print("\nWatching trained agent...")
    watch_trained_agent()


#     #####################################################################
#     """
# ENHANCED SOCCER ENVIRONMENT FOR NUBOTS RL STRATEGY
# ============================================================

# This implementation follows the requirements from the SENG4001A Interim Report
# and incorporates the design specifications for the NUbots reinforcement learning
# strategy project.

# Key improvements based on the design document:
# 1. 12-element state space as specified in Section 7.1
# 2. Improved reward function aligned with project objectives
# 3. Enhanced physics simulation for more realistic behavior
# 4. Support for variable opponent strategies
# 5. Performance metrics collection
# 6. Domain randomization for better transfer learning

# """

# import numpy as np
# import pygame
# import gymnasium as gym
# from gymnasium import spaces
# from typing import Dict, Tuple, Optional, Any
# import random
# import math


# class SoccerEnv(gym.Env):
#     """
#     Enhanced Soccer Environment for NUbots RL Strategy
    
#     This environment implements the specifications from the SENG4001A project:
#     - 12-element state space (Section 7.1)
#     - 4x4 meter field with configurable dimensions
#     - Realistic physics simulation
#     - Opponent behavior variations
#     - Performance metrics tracking
    
#     State Space (12 elements):
#         [0-2]: Robot pose (x, y, θ)
#         [3-4]: Ball position (x, y) 
#         [5-6]: Opponent position (x, y)
#         [7-8]: Robot velocity (vx, vy)
#         [9]: Distance to ball
#         [10]: Distance to goal
#         [11]: Game phase indicator
    
#     Action Space:
#         3D continuous: [x_velocity, y_velocity, angular_velocity] ∈ [-1.0, 1.0]
    
#     Reward Structure:
#         +100: Goal reached with ball possession
#         +1: Maintaining ball possession per step
#         +5: Moving closer to goal with ball
#         -1: Losing ball possession
#         -10: Collision with opponent
#         -0.1: Time penalty (encourages efficiency)
#     """
    
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
#     def __init__(
#         self,
#         render_mode: Optional[str] = None,
#         field_length: float = 4.0,
#         field_width: float = 4.0,
#         enable_domain_randomization: bool = True,
#         opponent_strategy: str = "dynamic",
#         max_episode_steps: int = 200
#     ):
#         super().__init__()
        
#         # Field dimensions (configurable as per requirements)
#         self.field_length = field_length
#         self.field_width = field_width
#         self.max_episode_steps = max_episode_steps
        
#         # Physics parameters
#         self.robot_radius = 0.15  # 15cm robot radius
#         self.ball_radius = 0.065  # Standard soccer ball
#         self.max_velocity = 1.0   # Max robot velocity (m/s)
#         self.ball_friction = 0.95 # Ball deceleration factor
        
#         # Domain randomization settings
#         self.enable_domain_randomization = enable_domain_randomization
#         self.opponent_strategy = opponent_strategy
        
#         # State space: 12 elements as specified in Section 7.1
#         self.observation_space = spaces.Box(
#             low=np.array([
#                 -self.field_length/2,  # robot_x
#                 -self.field_width/2,   # robot_y  
#                 -np.pi,                # robot_theta
#                 -self.field_length/2,  # ball_x
#                 -self.field_width/2,   # ball_y
#                 -self.field_length/2,  # opponent_x
#                 -self.field_width/2,   # opponent_y
#                 -self.max_velocity,    # velocity_x
#                 -self.max_velocity,    # velocity_y
#                 0.0,                   # ball_distance
#                 0.0,                   # goal_distance
#                 0.0                    # game_phase
#             ]),
#             high=np.array([
#                 self.field_length/2,   # robot_x
#                 self.field_width/2,    # robot_y
#                 np.pi,                 # robot_theta
#                 self.field_length/2,   # ball_x
#                 self.field_width/2,    # ball_y
#                 self.field_length/2,   # opponent_x
#                 self.field_width/2,    # opponent_y
#                 self.max_velocity,     # velocity_x
#                 self.max_velocity,     # velocity_y
#                 self.field_length,     # ball_distance (max diagonal)
#                 self.field_length,     # goal_distance
#                 1.0                    # game_phase
#             ]),
#             dtype=np.float32
#         )
        
#         # Action space: 3D continuous control
#         self.action_space = spaces.Box(
#             low=np.array([-1.0, -1.0, -1.0]),
#             high=np.array([1.0, 1.0, 1.0]),
#             dtype=np.float32
#         )
        
#         # Rendering setup
#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode
#         self.window = None
#         self.clock = None
#         self.window_size = 800
        
#         # Performance metrics (as specified in evaluation design)
#         self.reset_metrics()
        
#     def reset_metrics(self):
#         """Reset performance tracking metrics"""
#         self.episode_metrics = {
#             "ball_possession_time": 0,
#             "total_distance_traveled": 0,
#             "goal_progress": 0,
#             "collisions": 0,
#             "falls": 0,
#             "decision_count": 0,
#             "successful_maneuvers": 0
#         }
        
#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
#         """Reset environment with domain randomization support"""
#         super().reset(seed=seed)
        
#         # Reset metrics
#         self.reset_metrics()
#         self.steps = 0
        
#         # Domain randomization for field dimensions (±20% as per requirements)
#         if self.enable_domain_randomization:
#             length_variation = self.np_random.uniform(-0.2, 0.2)
#             width_variation = self.np_random.uniform(-0.2, 0.2)
#             self.current_field_length = self.field_length * (1 + length_variation)
#             self.current_field_width = self.field_width * (1 + width_variation)
#         else:
#             self.current_field_length = self.field_length
#             self.current_field_width = self.field_width
            
#         # Initialize robot position (left side)
#         self.robot_position = np.array([
#             -self.current_field_length/2 + 0.5,
#             self.np_random.uniform(-self.current_field_width/4, self.current_field_width/4)
#         ])
#         self.robot_theta = self.np_random.uniform(-np.pi/4, np.pi/4)
#         self.robot_velocity = np.array([0.0, 0.0])
        
#         # Initialize ball position (near robot)
#         ball_offset = np.array([
#             self.np_random.uniform(0.2, 0.4),
#             self.np_random.uniform(-0.1, 0.1)
#         ])
#         self.ball_position = self.robot_position + ball_offset
#         self.ball_velocity = np.array([0.0, 0.0])
        
#         # Initialize opponent (middle to right side)
#         self.opponent_position = np.array([
#             self.np_random.uniform(-0.5, self.current_field_length/2 - 0.5),
#             self.np_random.uniform(-self.current_field_width/2 + 0.5, self.current_field_width/2 - 0.5)
#         ])
#         self.opponent_velocity = np.array([0.0, 0.0])
        
#         # Goal position (right side)
#         self.goal_position = np.array([self.current_field_length/2, 0.0])
        
#         # Game state
#         self.has_ball = False
#         self.game_phase = 0.0  # 0=normal play, could be extended for penalties, etc.
        
#         # Update ball possession
#         self._update_ball_possession()
        
#         observation = self._get_observation()
#         info = self._get_info()
        
#         if self.render_mode == "human":
#             self._render_frame()
            
#         return observation, info
        
#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         """Execute one environment step"""
#         self.steps += 1
#         self.episode_metrics["decision_count"] += 1
        
#         # Validate and clip action
#         action = np.clip(action, -1.0, 1.0)
        
#         # Store previous state for reward calculation
#         prev_robot_pos = self.robot_position.copy()
#         prev_ball_distance = np.linalg.norm(self.robot_position - self.ball_position)
#         prev_goal_distance = np.linalg.norm(self.robot_position - self.goal_position)
#         prev_has_ball = self.has_ball
        
#         # Update robot state
#         self._update_robot(action)
        
#         # Update opponent
#         self._update_opponent()
        
#         # Update ball physics
#         self._update_ball_physics()
        
#         # Update ball possession
#         self._update_ball_possession()
        
#         # Calculate reward
#         reward = self._calculate_reward(prev_goal_distance, prev_has_ball, prev_robot_pos)
        
#         # Check termination conditions
#         terminated = self._check_terminated()
#         truncated = self.steps >= self.max_episode_steps
        
#         # Update metrics
#         self._update_metrics(prev_robot_pos)
        
#         observation = self._get_observation()
#         info = self._get_info()
        
#         if self.render_mode == "human":
#             self._render_frame()
            
#         return observation, reward, terminated, truncated, info
        
#     def _update_robot(self, action: np.ndarray):
#         """Update robot position and orientation based on action"""
#         dt = 1.0 / 50.0  # 50Hz simulation as per requirements
        
#         # Extract movement commands
#         x_vel_cmd, y_vel_cmd, angular_vel_cmd = action
        
#         # Apply velocity limits and acceleration constraints
#         max_accel = 2.0  # m/s²
#         max_angular_accel = 3.0  # rad/s²
        
#         # Update linear velocity
#         target_velocity = np.array([x_vel_cmd, y_vel_cmd]) * self.max_velocity
#         vel_diff = target_velocity - self.robot_velocity
#         vel_diff = np.clip(vel_diff, -max_accel * dt, max_accel * dt)
#         self.robot_velocity += vel_diff
        
#         # Update angular velocity (simplified)
#         self.angular_velocity = angular_vel_cmd * 2.0  # rad/s
        
#         # Update position and orientation
#         self.robot_position += self.robot_velocity * dt
#         self.robot_theta += self.angular_velocity * dt
        
#         # Keep theta in [-π, π]
#         self.robot_theta = np.arctan2(np.sin(self.robot_theta), np.cos(self.robot_theta))
        
#         # Apply field boundaries with collision detection
#         self.robot_position[0] = np.clip(
#             self.robot_position[0], 
#             -self.current_field_length/2 + self.robot_radius,
#             self.current_field_length/2 - self.robot_radius
#         )
#         self.robot_position[1] = np.clip(
#             self.robot_position[1],
#             -self.current_field_width/2 + self.robot_radius,
#             self.current_field_width/2 - self.robot_radius
#         )
        
#     def _update_opponent(self):
#         """Update opponent behavior based on strategy"""
#         dt = 1.0 / 50.0
        
#         if self.opponent_strategy == "static":
#             return
#         elif self.opponent_strategy == "simple":
#             # Simple ball-following behavior
#             ball_direction = self.ball_position - self.opponent_position
#             ball_distance = np.linalg.norm(ball_direction)
#             if ball_distance > 0.1:
#                 self.opponent_velocity = ball_direction / ball_distance * 0.3
#             else:
#                 self.opponent_velocity = np.array([0.0, 0.0])
#         elif self.opponent_strategy == "dynamic":
#             # More sophisticated opponent behavior
#             to_ball = self.ball_position - self.opponent_position
#             to_robot = self.robot_position - self.opponent_position
            
#             if self.has_ball:
#                 # Try to intercept robot
#                 self.opponent_velocity = to_robot / (np.linalg.norm(to_robot) + 1e-6) * 0.5
#             else:
#                 # Go for the ball
#                 self.opponent_velocity = to_ball / (np.linalg.norm(to_ball) + 1e-6) * 0.4
                
#         # Update opponent position
#         self.opponent_position += self.opponent_velocity * dt
        
#         # Apply field boundaries
#         self.opponent_position[0] = np.clip(
#             self.opponent_position[0],
#             -self.current_field_length/2 + self.robot_radius,
#             self.current_field_length/2 - self.robot_radius
#         )
#         self.opponent_position[1] = np.clip(
#             self.opponent_position[1],
#             -self.current_field_width/2 + self.robot_radius,
#             self.current_field_width/2 - self.robot_radius
#         )
        
#     def _update_ball_physics(self):
#         """Update ball position with physics simulation"""
#         dt = 1.0 / 50.0
        
#         # Apply friction to ball
#         self.ball_velocity *= self.ball_friction
        
#         # Update ball position
#         self.ball_position += self.ball_velocity * dt
        
#         # Ball-field boundary collisions
#         if (self.ball_position[0] <= -self.current_field_length/2 + self.ball_radius or
#             self.ball_position[0] >= self.current_field_length/2 - self.ball_radius):
#             self.ball_velocity[0] *= -0.8  # Some energy loss
            
#         if (self.ball_position[1] <= -self.current_field_width/2 + self.ball_radius or
#             self.ball_position[1] >= self.current_field_width/2 - self.ball_radius):
#             self.ball_velocity[1] *= -0.8
            
#         # Clamp ball to field
#         self.ball_position[0] = np.clip(
#             self.ball_position[0],
#             -self.current_field_length/2 + self.ball_radius,
#             self.current_field_length/2 - self.ball_radius
#         )
#         self.ball_position[1] = np.clip(
#             self.ball_position[1],
#             -self.current_field_width/2 + self.ball_radius,
#             self.current_field_width/2 - self.ball_radius
#         )
        
#     def _update_ball_possession(self):
#         """Update ball possession and handle ball-robot interactions"""
#         robot_to_ball = self.ball_position - self.robot_position
#         distance_to_ball = np.linalg.norm(robot_to_ball)
        
#         # Ball possession threshold
#         possession_threshold = self.robot_radius + self.ball_radius + 0.1
        
#         if distance_to_ball <= possession_threshold:
#             if not self.has_ball:
#                 self.has_ball = True
                
#             # Ball follows robot when in possession
#             desired_ball_pos = (self.robot_position + 
#                               np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)]) * 
#                               (self.robot_radius + self.ball_radius + 0.05))
            
#             # Smooth ball movement towards desired position
#             self.ball_velocity = (desired_ball_pos - self.ball_position) * 5.0
#         else:
#             if self.has_ball and distance_to_ball > possession_threshold * 1.5:
#                 self.has_ball = False
                
#         # Check opponent ball possession
#         opponent_to_ball = self.ball_position - self.opponent_position
#         if np.linalg.norm(opponent_to_ball) <= possession_threshold:
#             self.has_ball = False
            
#     def _calculate_reward(self, prev_goal_distance: float, prev_has_ball: bool, prev_pos: np.ndarray) -> float:
#         """Calculate reward based on current state and actions"""
#         reward = 0.0
        
#         # Time penalty (encourages efficiency)
#         reward -= 0.1
        
#         # Ball possession reward
#         if self.has_ball:
#             reward += 1.0  # Continuous possession reward
#             self.episode_metrics["ball_possession_time"] += 1
            
#             # Progress toward goal reward
#             current_goal_distance = np.linalg.norm(self.robot_position - self.goal_position)
#             if current_goal_distance < prev_goal_distance:
#                 progress_reward = (prev_goal_distance - current_goal_distance) * 10.0
#                 reward += progress_reward
#                 self.episode_metrics["goal_progress"] += progress_reward
#         else:
#             # Penalty for losing ball
#             if prev_has_ball:
#                 reward -= 1.0
                
#         # Goal reached with ball possession
#         goal_distance = np.linalg.norm(self.robot_position - self.goal_position)
#         if goal_distance < 0.3 and self.has_ball:
#             reward += 100.0
#             self.episode_metrics["successful_maneuvers"] += 1
            
#         # Collision penalty
#         robot_opponent_distance = np.linalg.norm(self.robot_position - self.opponent_position)
#         if robot_opponent_distance < (self.robot_radius * 2):
#             reward -= 10.0
#             self.episode_metrics["collisions"] += 1
            
#         # Boundary penalty (soft constraint)
#         if (abs(self.robot_position[0]) > self.current_field_length/2 - 0.2 or
#             abs(self.robot_position[1]) > self.current_field_width/2 - 0.2):
#             reward -= 0.5
            
#         return reward
        
#     def _check_terminated(self) -> bool:
#         """Check if episode should terminate"""
#         # Success: reached goal with ball
#         goal_distance = np.linalg.norm(self.robot_position - self.goal_position)
#         if goal_distance < 0.3 and self.has_ball:
#             return True
            
#         # Failure: major collision
#         robot_opponent_distance = np.linalg.norm(self.robot_position - self.opponent_position)
#         if robot_opponent_distance < self.robot_radius:
#             return True
            
#         # Ball stationary for too long
#         if np.linalg.norm(self.ball_velocity) < 0.01 and not self.has_ball:
#             return True
            
#         return False
        
#     def _get_observation(self) -> np.ndarray:
#         """Get 12-element observation as specified in design document"""
#         ball_distance = np.linalg.norm(self.robot_position - self.ball_position)
#         goal_distance = np.linalg.norm(self.robot_position - self.goal_position)
        
#         observation = np.array([
#             self.robot_position[0],      # [0] robot_x
#             self.robot_position[1],      # [1] robot_y  
#             self.robot_theta,            # [2] robot_theta
#             self.ball_position[0],       # [3] ball_x
#             self.ball_position[1],       # [4] ball_y
#             self.opponent_position[0],   # [5] opponent_x
#             self.opponent_position[1],   # [6] opponent_y
#             self.robot_velocity[0],      # [7] velocity_x
#             self.robot_velocity[1],      # [8] velocity_y
#             ball_distance,               # [9] ball_distance
#             goal_distance,               # [10] goal_distance
#             self.game_phase              # [11] game_phase
#         ], dtype=np.float32)
        
#         return observation
        
#     def _get_info(self) -> Dict[str, Any]:
#         """Get additional environment information"""
#         return {
#             "has_ball": self.has_ball,
#             "steps": self.steps,
#             "metrics": self.episode_metrics.copy(),
#             "field_dimensions": (self.current_field_length, self.current_field_width),
#             "success_criteria": {
#                 "goal_reached": np.linalg.norm(self.robot_position - self.goal_position) < 0.3 and self.has_ball,
#                 "ball_possession_rate": self.episode_metrics["ball_possession_time"] / max(self.steps, 1),
#                 "collision_count": self.episode_metrics["collisions"]
#             }
#         }
        
#     def _update_metrics(self, prev_pos: np.ndarray):
#         """Update performance metrics"""
#         distance_traveled = np.linalg.norm(self.robot_position - prev_pos)
#         self.episode_metrics["total_distance_traveled"] += distance_traveled
        
#     def _render_frame(self):
#         """Render the environment"""
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()
            
#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((34, 139, 34))  # Green field
        
#         # Scale factor for rendering
#         scale = self.window_size / max(self.current_field_length, self.current_field_width)
#         center_x, center_y = self.window_size // 2, self.window_size // 2
        
#         def world_to_screen(pos):
#             return (
#                 int(center_x + pos[0] * scale),
#                 int(center_y - pos[1] * scale)  # Flip Y axis
#             )
            
#         # Draw field boundaries
#         field_rect = pygame.Rect(
#             center_x - self.current_field_length * scale // 2,
#             center_y - self.current_field_width * scale // 2,
#             self.current_field_length * scale,
#             self.current_field_width * scale
#         )
#         pygame.draw.rect(canvas, (255, 255, 255), field_rect, 3)
        
#         # Draw goal
#         goal_pos = world_to_screen(self.goal_position)
#         pygame.draw.circle(canvas, (255, 215, 0), goal_pos, int(20 * scale))
        
#         # Draw opponent
#         opponent_pos = world_to_screen(self.opponent_position)
#         pygame.draw.circle(canvas, (255, 0, 0), opponent_pos, int(self.robot_radius * scale))
        
#         # Draw ball
#         ball_pos = world_to_screen(self.ball_position)
#         ball_color = (255, 255, 255) if not self.has_ball else (0, 255, 0)
#         pygame.draw.circle(canvas, ball_color, ball_pos, int(self.ball_radius * scale))
        
#         # Draw robot
#         robot_pos = world_to_screen(self.robot_position)
#         robot_color = (0, 0, 255) if not self.has_ball else (0, 255, 255)
#         pygame.draw.circle(canvas, robot_color, robot_pos, int(self.robot_radius * scale))
        
#         # Draw robot orientation
#         orientation_end = world_to_screen(
#             self.robot_position + 0.2 * np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])
#         )
#         pygame.draw.line(canvas, (255, 255, 255), robot_pos, orientation_end, 2)
        
#         # Draw metrics
#         if self.render_mode == "human":
#             font = pygame.font.Font(None, 24)
#             metrics_text = [
#                 f"Step: {self.steps}",
#                 f"Ball Possession: {self.has_ball}",
#                 f"Distance to Goal: {np.linalg.norm(self.robot_position - self.goal_position):.2f}m",
#                 f"Collisions: {self.episode_metrics['collisions']}"
#             ]
            
#             for i, text in enumerate(metrics_text):
#                 text_surface = font.render(text, True, (255, 255, 255))
#                 canvas.blit(text_surface, (10, 10 + i * 25))
        
#         if self.render_mode == "human":
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#             self.clock.tick(self.metadata["render_fps"])
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )
            
#     def close(self):
#         """Close the environment"""
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()


# # TESTING AND EVALUATION FUNCTIONS
# def test_enhanced_environment():
#     """Test the enhanced environment with improved features"""
#     print("🎮 Testing Enhanced Soccer Environment")
#     print("=" * 50)
    
#     env = SoccerEnv(
#         render_mode="human",
#         enable_domain_randomization=True,
#         opponent_strategy="dynamic"
#     )
    
#     observation, info = env.reset(seed=42)
#     print(f"Initial observation shape: {observation.shape}")
#     print(f"Observation space: {env.observation_space}")
#     print(f"Action space: {env.action_space}")
#     print(f"Initial info: {info}")
    
#     total_reward = 0
#     for step in range(200):
#         # Random action for testing
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
        
#         if step % 20 == 0:
#             print(f"Step {step}: Reward={reward:.3f}, Has ball={info['has_ball']}, "
#                   f"Success criteria={info['success_criteria']}")
            
#         if terminated or truncated:
#             print(f"Episode finished at step {step}")
#             print(f"Final metrics: {info['metrics']}")
#             print(f"Total reward: {total_reward:.2f}")
#             break
            
#     env.close()


# def benchmark_environment_performance():
#     """Benchmark environment performance for training"""
#     print("\n⚡ Benchmarking Environment Performance")
#     print("=" * 40)
    
#     env = SoccerEnv(render_mode=None)  # No rendering for speed
    
#     import time
#     start_time = time.time()
    
#     num_episodes = 100
#     total_steps = 0
    
#     for episode in range(num_episodes):
#         obs, info = env.reset()
#         episode_steps = 0
        
#         while episode_steps < 200:
#             action = env.action_space.sample()
#             obs, reward, terminated, truncated, info = env.step(action)
#             episode_steps += 1
#             total_steps += 1
            
#             if terminated or truncated:
#                 break
                
#     end_time = time.time()
#     duration = end_time - start_time
    
#     print(f"Completed {num_episodes} episodes in {duration:.2f} seconds")
#     print(f"Average steps per episode: {total_steps / num_episodes:.1f}")
#     print(f"Steps per second: {total_steps / duration:.1f}")
#     print(f"Episodes per second: {num_episodes / duration:.1f}")
    
#     env.close()


# if __name__ == "__main__":
#     # Test the enhanced environment
#     test_enhanced_environment()
    
#     # Benchmark performance
#     benchmark_environment_performance()