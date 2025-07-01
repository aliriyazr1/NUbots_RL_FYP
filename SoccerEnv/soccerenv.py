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
    dqn_model.learn(total_timesteps=50000)
    
    print("📊 Evaluating DQN...")
    dqn_mean, dqn_std = evaluate_policy(dqn_model, env, n_eval_episodes=20)
    results["DQN"] = (dqn_mean, dqn_std)
    print(f"DQN Results: {dqn_mean:.2f} ± {dqn_std:.2f}")
    
    # Train PPO
    print("\n🚀 Training PPO...")
    ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    ppo_model.learn(total_timesteps=50000)
    
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