# =====================================================
# PPO HYPERPARAMETERS - COMPLETE GUIDE
# =====================================================

class PPOHyperparameters:
    """Every single PPO hyperparameter explained"""
    
    def __init__(self):
        self.hyperparameters = {
            
            # ============= LEARNING RATE =============
            "learning_rate": {
                "your_value": 1e-3,
                "description": "How big steps the neural network takes when learning",
                "good_values": "1e-4 to 1e-3",
                "bad_values": "> 1e-2 (too unstable), < 1e-5 (too slow)",
                "effects": {
                    "too_high": "Training explodes, loss goes to infinity, robot acts randomly",
                    "too_low": "Robot learns extremely slowly, may never improve",
                    "just_right": "Steady improvement in performance"
                },
                "recommendation_for_you": "1e-3 (your current value is good!)",
                "why": "Soccer needs moderate learning - not too fast (unstable) not too slow"
            },
            
            # ============= ROLLOUT PARAMETERS =============
            "n_steps": {
                "your_value": 4096,
                "description": "How many steps to collect before each training update",
                "good_values": "2048-8192 for continuous control",
                "bad_values": "< 512 (too little data), > 16384 (too slow updates)",
                "effects": {
                    "too_low": "Updates based on too little experience, unstable learning",
                    "too_high": "Very slow training, stale experiences",
                    "just_right": "Good balance of fresh data and stable updates"
                },
                "recommendation_for_you": "4096 (perfect for your soccer task)",
                "why": "Soccer episodes are moderately long, need enough data per update"
            },
            
            "batch_size": {
                "your_value": 128,
                "description": "How many experiences to use in each gradient update",
                "good_values": "64-512 depending on n_steps",
                "bad_values": "< 32 (too noisy), > n_steps (invalid)",
                "constraint": "Must be <= n_steps and preferably n_steps should be divisible by batch_size",
                "effects": {
                    "too_low": "Very noisy gradients, unstable learning",
                    "too_high": "Less frequent updates, might miss important patterns",
                    "just_right": "Smooth, stable gradient updates"
                },
                "recommendation_for_you": "256 (increase from 128)",
                "why": "With n_steps=4096, you can afford larger batches for stability"
            },
            
            "n_epochs": {
                "your_value": 8,
                "description": "How many times to reuse collected data for training",
                "good_values": "4-20 for most tasks",
                "bad_values": "< 3 (underutilizing data), > 20 (overfitting old data)",
                "effects": {
                    "too_low": "Wasting collected experience, slow learning",
                    "too_high": "Overfitting to old experiences, policy becomes stale",
                    "just_right": "Efficiently uses data without overfitting"
                },
                "recommendation_for_you": "10 (increase from 8)",
                "why": "Soccer is complex, can benefit from more epochs"
            },
            
            # ============= DISCOUNT AND ADVANTAGE ESTIMATION =============
            "gamma": {
                "your_value": 0.99,
                "description": "How much to value future rewards vs immediate rewards",
                "good_values": "0.95-0.999",
                "bad_values": "< 0.9 (too shortsighted), > 0.999 (numerical issues)",
                "effects": {
                    "too_low": "Robot only cares about immediate rewards, no strategy",
                    "too_high": "Robot obsesses over distant future, ignores present",
                    "just_right": "Good balance of immediate and future planning"
                },
                "recommendation_for_you": "0.99 (perfect!)",
                "why": "Soccer needs long-term planning to reach goal"
            },
            
            "gae_lambda": {
                "your_value": 0.95,
                "description": "Controls bias-variance tradeoff in advantage estimation",
                "good_values": "0.9-0.98",
                "bad_values": "< 0.8 (too biased), > 0.99 (too much variance)",
                "effects": {
                    "too_low": "Biased advantage estimates, poor policy updates",
                    "too_high": "High variance, noisy learning",
                    "just_right": "Accurate advantage estimates"
                },
                "recommendation_for_you": "0.95 (excellent choice!)",
                "why": "Good default that works well for most continuous control tasks"
            },
            
            # ============= PPO-SPECIFIC CLIPPING =============
            "clip_range": {
                "your_value": 0.15,
                "description": "Prevents policy from changing too drastically in one update",
                "good_values": "0.1-0.3",
                "bad_values": "< 0.05 (too conservative), > 0.5 (defeats purpose)",
                "effects": {
                    "too_low": "Policy barely changes, very slow learning",
                    "too_high": "Policy can change drastically, unstable training",
                    "just_right": "Steady, stable policy improvements"
                },
                "recommendation_for_you": "0.2 (increase from 0.15)",
                "why": "Standard value, your 0.15 is slightly conservative"
            },
            
            # ============= EXPLORATION =============
            "ent_coef": {
                "your_value": 0.1,
                "description": "Encourages exploration by penalizing too-confident actions",
                "good_values": "0.01-0.1",
                "bad_values": "< 0.001 (no exploration), > 0.5 (too random)",
                "effects": {
                    "too_low": "Robot gets stuck in local optima, no exploration",
                    "too_high": "Robot acts randomly, never learns consistent strategy",
                    "just_right": "Good exploration while learning"
                },
                "recommendation_for_you": "0.05 (decrease from 0.1)",
                "why": "Soccer needs some exploration but not too much randomness"
            },
            
            # ============= VALUE FUNCTION =============
            "vf_coef": {
                "your_value": 0.5,
                "description": "How much to weight value function loss vs policy loss",
                "good_values": "0.25-1.0",
                "bad_values": "< 0.1 (poor value estimates), > 2.0 (dominates policy)",
                "effects": {
                    "too_low": "Poor value function, bad advantage estimates",
                    "too_high": "Value function dominates, policy learning suffers",
                    "just_right": "Balanced learning of both policy and value"
                },
                "recommendation_for_you": "0.5 (perfect!)",
                "why": "Standard balanced value that works well"
            },
            
            # ============= GRADIENT CLIPPING =============
            "max_grad_norm": {
                "your_value": 0.5,
                "description": "Prevents exploding gradients by clipping large updates",
                "good_values": "0.5-2.0",
                "bad_values": "< 0.1 (too restrictive), > 10 (ineffective)",
                "effects": {
                    "too_low": "Gradients clipped too aggressively, slow learning",
                    "too_high": "Doesn't prevent gradient explosions",
                    "just_right": "Prevents instability while allowing good learning"
                },
                "recommendation_for_you": "0.5 (excellent!)",
                "why": "Good default that prevents training instabilities"
            }
        }
        
        # ============= MISSING PARAMETERS YOU SHOULD CONSIDER =============
        self.missing_parameters = {
            "clip_range_vf": {
                "description": "Clipping for value function updates (like clip_range for policy)",
                "default": "None (no clipping)",
                "recommendation": "None or same as clip_range",
                "why_add": "Can help stabilize value function learning"
            },
            
            "target_kl": {
                "description": "Stop early if policy changes too much (KL divergence limit)",
                "default": "None",
                "recommendation": "0.01-0.05",
                "why_add": "Prevents policy from changing too drastically"
            },
            
            "normalize_advantage": {
                "description": "Normalize advantages to have mean 0, std 1",
                "default": "True",
                "recommendation": "True",
                "why_add": "More stable training, better gradient scaling"
            }
        }

# =====================================================
# DDPG HYPERPARAMETERS - COMPLETE GUIDE  
# =====================================================

class DDPGHyperparameters:
    """Every single DDPG hyperparameter explained"""
    
    def __init__(self):
        self.hyperparameters = {
            
            # ============= LEARNING RATES =============
            "learning_rate": {
                "your_value": 1e-3,
                "description": "Learning rate for both actor and critic networks",
                "good_values": "1e-4 to 1e-3",
                "bad_values": "> 1e-2 (unstable), < 1e-5 (too slow)",
                "recommendation_for_you": "1e-3 (good choice!)",
                "note": "DDPG can handle slightly higher learning rates than PPO"
            },
            
            # ============= REPLAY BUFFER =============
            "buffer_size": {
                "your_value": 200000,
                "description": "How many past experiences to remember for training",
                "good_values": "100k-1M for complex tasks",
                "bad_values": "< 10k (too little memory), > 10M (memory issues)",
                "effects": {
                    "too_low": "Forgets important experiences too quickly",
                    "too_high": "Uses very old, irrelevant experiences",
                    "just_right": "Good mix of recent and diverse experiences"
                },
                "recommendation_for_you": "500000 (increase from 200k)",
                "why": "Soccer is complex, benefits from more diverse experiences"
            },
            
            "learning_starts": {
                "your_value": 5000,
                "description": "How many random steps before starting to learn",
                "good_values": "1k-10k depending on complexity",
                "bad_values": "< 500 (not enough exploration), > 50k (too much delay)",
                "effects": {
                    "too_low": "Starts learning from very limited experience",
                    "too_high": "Takes too long before any learning begins",
                    "just_right": "Good initial exploration before learning"
                },
                "recommendation_for_you": "10000 (increase from 5k)",
                "why": "Soccer needs good initial exploration of state space"
            },
            
            "batch_size": {
                "your_value": 256,
                "description": "How many experiences to sample from buffer for each update",
                "good_values": "128-512",
                "bad_values": "< 64 (too noisy), > 1024 (too slow)",
                "recommendation_for_you": "256 (perfect!)",
                "why": "Good balance of stability and computational efficiency"
            },
            
            # ============= TARGET NETWORK UPDATES =============
            "tau": {
                "your_value": 0.01,
                "description": "How fast to update target networks (soft update rate)",
                "good_values": "0.001-0.01",
                "bad_values": "< 0.0001 (too slow updates), > 0.1 (too fast, unstable)",
                "effects": {
                    "too_low": "Target networks barely change, learning is slow",
                    "too_high": "Target networks change too fast, training unstable",
                    "just_right": "Stable learning with steady target updates"
                },
                "recommendation_for_you": "0.005 (decrease from 0.01)",
                "why": "Slightly more conservative for better stability"
            },
            
            # ============= DISCOUNT FACTOR =============
            "gamma": {
                "your_value": 0.99,
                "description": "Same as PPO - how much to value future rewards",
                "recommendation_for_you": "0.99 (perfect!)",
                "why": "Same reasoning as PPO - soccer needs long-term planning"
            },
            
            # ============= EXPLORATION NOISE =============
            "action_noise": {
                "your_value": None,
                "description": "Noise added to actions for exploration",
                "current_problem": "You're not using any exploration noise!",
                "recommendation": "Add OrnsteinUhlenbeckActionNoise or NormalActionNoise",
                "why_critical": "DDPG NEEDS noise for exploration, otherwise gets stuck"
            }
        }
        
        # ============= MISSING CRITICAL PARAMETERS =============
        self.missing_critical = {
            "train_freq": {
                "description": "How often to train the networks (every N steps)",
                "default": "1 (every step)",
                "recommendation": "1-4",
                "why_important": "Controls how often network gets updated"
            },
            
            "gradient_steps": {
                "description": "How many gradient steps per training call",
                "default": "1",
                "recommendation": "1-2",
                "why_important": "More steps = more learning per update"
            },
            
            "actor_lr": {
                "description": "Separate learning rate for actor network",
                "default": "Same as learning_rate",
                "recommendation": "1e-4 (often lower than critic)",
                "why_important": "Actor often needs more careful updates"
            },
            
            "critic_lr": {
                "description": "Separate learning rate for critic network", 
                "default": "Same as learning_rate",
                "recommendation": "1e-3",
                "why_important": "Critic can handle faster learning"
            }
        }

# =====================================================
# RECOMMENDED CONFIGURATIONS FOR YOUR FYP
# =====================================================

class RecommendedConfigurations:
    """Optimized hyperparameters specifically for your soccer task"""
    
    def optimized_ppo_config(self):
        """Improved PPO configuration for your soccer environment"""
        return PPO(
            policy="MlpPolicy",
            env=train_env,
            
            # Learning rate - keep your current value
            learning_rate=1e-3,
            
            # Rollout parameters - minor improvements
            n_steps=4096,                    # Keep current
            batch_size=256,                  # INCREASE from 128
            n_epochs=10,                     # INCREASE from 8
            
            # Discount and advantage
            gamma=0.99,                      # Keep current - perfect
            gae_lambda=0.95,                 # Keep current - perfect
            
            # PPO clipping
            clip_range=0.2,                  # INCREASE from 0.15 (standard)
            clip_range_vf=None,              # ADD: Optional value function clipping
            
            # Exploration
            ent_coef=0.05,                   # DECREASE from 0.1 (less randomness)
            
            # Value function and stability
            vf_coef=0.5,                     # Keep current - perfect
            max_grad_norm=0.5,               # Keep current - perfect
            
            # ADD: Additional stability parameters
            normalize_advantage=True,         # ADD: Normalize advantages
            target_kl=0.01,                  # ADD: Early stopping if policy changes too much
            
            # Network architecture
            policy_kwargs=dict(
                net_arch=[128, 128, 64],
                activation_fn=torch.nn.ReLU,
            ),
            
            verbose=1,
            device="cpu"
        )
    
    def optimized_ddpg_config(self):
        """Improved DDPG configuration with critical missing pieces"""
        
        # CRITICAL: Add exploration noise (you're missing this!)
        from stable_baselines3.common.noise import NormalActionNoise
        import numpy as np
        
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=0.1 * np.ones(n_actions)  # 10% noise for exploration
        )
        
        return DDPG(
            policy="MlpPolicy",
            env=train_env,
            
            # Learning rates - consider separate rates
            learning_rate=1e-3,              # Keep current
            # actor_lr=1e-4,                 # ADD: Slower actor learning
            # critic_lr=1e-3,                # ADD: Faster critic learning
            
            # Replay buffer - increase for better sampling
            buffer_size=500000,              # INCREASE from 200k
            learning_starts=10000,           # INCREASE from 5k
            batch_size=256,                  # Keep current - good
            
            # Target network updates - more conservative
            tau=0.005,                       # DECREASE from 0.01 (more stable)
            
            # Discount factor
            gamma=0.99,                      # Keep current - perfect
            
            # CRITICAL: Add exploration noise
            action_noise=action_noise,       # ADD: Essential for DDPG exploration!
            
            # Training frequency
            train_freq=1,                    # ADD: Train every step
            gradient_steps=1,                # ADD: One gradient step per training
            
            # Network architecture - your current is good
            policy_kwargs=dict(
                net_arch=[256, 256, 128],    # Keep current
                activation_fn=torch.nn.ReLU,
            ),
            
            verbose=1,
            device="cpu"
        )

# =====================================================
# HYPERPARAMETER TUNING TIPS
# =====================================================

class HyperparameterTuningTips:
    """How to systematically improve your hyperparameters"""
    
    def tuning_order(self):
        """What to tune first for maximum impact"""
        return {
            "Priority 1 (Most Impact)": [
                "learning_rate",
                "action_noise (DDPG)",
                "n_steps (PPO)",
                "buffer_size (DDPG)"
            ],
            "Priority 2 (Moderate Impact)": [
                "batch_size",
                "ent_coef (PPO)",
                "tau (DDPG)",
                "clip_range (PPO)"
            ],
            "Priority 3 (Fine-tuning)": [
                "n_epochs (PPO)",
                "gae_lambda (PPO)",
                "max_grad_norm"
            ]
        }
    
    def warning_signs(self):
        """How to tell if hyperparameters are wrong"""
        return {
            "Learning too slow": "Increase learning_rate, decrease clip_range (PPO)",
            "Training unstable": "Decrease learning_rate, add max_grad_norm",
            "No exploration": "Increase ent_coef (PPO), add action_noise (DDPG)",
            "Too random": "Decrease ent_coef (PPO), reduce action_noise (DDPG)",
            "Overfitting": "Decrease n_epochs (PPO), increase buffer_size (DDPG)"
        }

# =====================================================
# CRITICAL ISSUE WITH YOUR CURRENT DDPG
# =====================================================

"""
🚨 CRITICAL: Your DDPG has NO exploration noise!

action_noise=None  # This is BAD for DDPG!

DDPG is deterministic and WILL get stuck without noise.
You MUST add exploration noise for DDPG to work properly.

Add this immediately:
from stable_baselines3.common.noise import NormalActionNoise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

Then use: action_noise=action_noise in your DDPG constructor
"""




#######################################################################################
# RoboCup Humanoid League Field Configuration
# Based on RC-HL-2025-Rules.pdf specifications
# Updated to be consistent with all code requirements

# Field Type: Choose between 'kidsize' or 'adultsize' or 'custom'
field_type: "kidsize"

# Real-world field dimensions (in meters) - from RoboCup rules
real_world_dimensions:
  kidsize:
    field_length: 9.0          # Touch line length (A)
    field_width: 6.0           # Goal line length (B)
    goal_depth: 0.6            # Goal depth (C)
    goal_width: 2.6            # Goal width (D)
    goal_height: 1.2           # Goal height
    goal_area_length: 1.0      # Goal area length (E)
    goal_area_width: 3.0       # Goal area width (F)
    penalty_mark_distance: 1.5 # Penalty mark distance from goal (G)
    centre_circle_diameter: 1.5 # Centre circle diameter (H)
    border_strip_width: 1.0    # Minimum border strip width (I)
    penalty_area_length: 2.0   # Penalty area length (J)
    penalty_area_width: 5.0    # Penalty area width (K)
    line_width: 0.05           # Field line width (~5cm)

# Simulation display dimensions (in pixels)
# The simulation will scale real-world dimensions to these pixel dimensions
display_dimensions:
  width: 900                   # Pixels for field width (Used to be 600)
  height: 600                  # Pixels for field height (Used to be 400)
  scale_factor: 100            # Pixels per meter (auto-calculated)

# Robot and ball parameters (in meters - real world scale)
robot_parameters:
  robot_radius: 0.15           # Robot collision radius (15cm)
  ball_radius: 0.08            # Ball radius (8cm - FIFA standard)

  possession_threshold: 0.25   # Distance to consider "close to ball" (25cm) 
  contact_distance: 0.25      # Can push ball
  close_distance: 0.4         # Distance between the robot and the ball to be considered 'close'
  medium_distance: 0.8        # Distance between the robot and the ball to be considered 'medium distance'
  far_distance: 1.5           # Distance between the robot and the ball to be considered 'far'

  good_shepherding_angle: 45         # Degrees
  okay_shepherding_angle: 90         # Degrees
  ball_speed_minimum: 0.1            # For rewards
  robot_speed_minimum: 0.1           # For movement rewards
  prediction_steps: 5                # Steps ahead for ball prediction    

# Physics parameters
physics:
  ball_mass: 0.5               # Ball mass for physics calculations
  ball_friction: 0.8           # Ball friction (0-1, lower = more friction)
  ball_bounce: 0.2             # Wall bounce dampening
  ball_max_speed: 3.0          # Maximum ball speed (m/s)
  push_force_multiplier: 1.2   # How strong robot pushes ball

# Game difficulty settings
difficulty_settings:
  easy:
    possession_distance: 0.5   # Easier to get ball (50cm)
    collision_distance: 0.2    # More forgiving collisions (20cm)
    max_steps: 500             # Shorter episodes (Used to be 150)
    opponent_speed: 0.8        # Slower opponent
    
  medium:
    possession_distance: 0.4   # Medium ball possession range (40cm)
    collision_distance: 0.25   # Medium collision distance (25cm)
    max_steps: 800             # Medium episode length (Used to be 200)
    opponent_speed: 1.0        # Normal opponent speed
    
  hard:
    possession_distance: 0.35  # Harder to get ball (35cm)
    collision_distance: 0.3    # Less forgiving collisions (30cm)
    max_steps: 1000             # Longer episodes (Used to be 250)
    opponent_speed: 1.2        # Faster opponent

# Strategic zones (as percentages of field dimensions)
# These define key areas for reward calculation and AI decision making
strategic_zones:
  # Optimal shooting zone (directly in front of goal)
  optimal_shooting:
    x_min_percent: 0.725       # 72.5% of field length (close to goal)
    y_min_percent: 0.35       # Used to be 42.5% of field width (centered on goal)
    y_max_percent: 0.65       # Used to be 57.5% of field width
    
  # Dangerous attacking zone (wider area near goal)
  attacking_zone:
    x_min_percent: 0.6         # 60% of field length
    x_max_percent: 1.0         # Full field length
    y_min_percent: 0.25        # 25% of field width
    y_max_percent: 0.75        # 75% of field width
    
  # Defensive zone (own half)
  defensive_zone:
    x_min_percent: 0.0         # Start of field
    x_max_percent: 0.5         # Half field
    y_min_percent: 0.0         # Full width
    y_max_percent: 1.0
    
  # Midfield zone
  midfield_zone:
    x_min_percent: 0.3         # 30% of field length
    x_max_percent: 0.7         # 70% of field length
    y_min_percent: 0.0         # Full width
    y_max_percent: 1.0
    
  # Goal area coordinates (calculated from real dimensions)
  goal_area:
    # These will be calculated automatically based on goal dimensions
    use_real_dimensions: true
    
  # Penalty area coordinates (calculated from real dimensions)
  penalty_area:
    # These will be calculated automatically based on penalty area dimensions
    use_real_dimensions: true

# Reward function parameters - Updated to match code expectations
reward_parameters:
  # Primary rewards
  goal_scored: 100.0            # Reward for scoring a goal (used to be 15.0 but trying something new)
  ball_possession: 12.0         # Reward per step for having ball
  progress_to_goal: 8.0        # Reward for moving ball towards goal (Used to be 0.5)
  shooting_zone_bonus: 10.0     # Bonus for being in optimal shooting zone with ball (used to be 1.0 but humouring AI)
  contact_reward: 15.0
  close_reward: 8.0
  medium_reward: 4.0
  far_reward: 1.0
  too_far_penalty: -2.0
  proximity_multiplier: 10.0
  competitive_advantage_multiplier: 5.0
  competitive_disadvantage_multiplier: -2.0
  
  # Penalties
  collision_penalty: -4.0      # Penalty for colliding with opponent (Used to be -1.0)
  out_of_bounds_penalty: -10.0  # Penalty for ball going out of bounds
  time_penalty: -0.001         # Small penalty per step to encourage efficiency (matches code)
  edge_camping_penalty: -0.2           # Strong penalty for robot camping edges (matches code) (used to be -1.0)
  corner_penalty: -0.2         # Penalty for staying in corners too long (matches code) (used to be -0.5)
  defensive_zone_penalty: -0.1 # Penalty for staying in defensive zone too long
  opponent_possession_penalty: -0.1  # New: penalty when opponent has ball
  opponent_long_possession_penalty: -2.0
  opponent_very_long_possession_penalty: -5.0
  
  # Bonuses
  possession_time_bonus: 0.05
  possession_time_max: 3.0

  center_field_bonus: 0.3      # Reward for ball being in center field (matches code)
  attacking_zone_bonus: 0.05   # Small bonus for being in attacking zones  
  dangerous_area_bonus: 10.0   
  max_ball_speed_reward: 3.0
  good_shepherding_reward: 8.0 # Bonus for good ball shepherding positioning (Used to be 2.0)
  okay_shepherding_reward: 4.0
  ball_speed_multiplier: 2.0        # Bonus for moving ball towards goal with speed (Used to be 0.5)
  movement_alignment_multiplier: 1.5
  
# Game thresholds (steps, angles, speeds)
game_thresholds:
  opponent_possession_warning: 30
  opponent_possession_critical: 60
  shepherding_good_angle: 45
  shepherding_okay_angle: 90
  ball_speed_minimum: 0.1
  robot_speed_minimum: 0.1
  prediction_steps: 5
  edge_threshold_percent: 0.05 # 5% of field size for edge detection

#TODO: (CHECK WHETHER USED) Training parameters (probably not used by training scripts)
training_parameters:
  # PPO specific
  ppo_learning_rate: 0.001
  ppo_n_steps: 4096
  ppo_batch_size: 128
  ppo_n_epochs: 8
  ppo_gamma: 0.99
  ppo_gae_lambda: 0.95
  ppo_clip_range: 0.15
  ppo_ent_coef: 0.1
  ppo_vf_coef: 0.5
  ppo_max_grad_norm: 0.5
  
  # DDPG specific
  ddpg_learning_rate: 0.001
  ddpg_buffer_size: 200000
  ddpg_learning_starts: 5000
  ddpg_batch_size: 256
  ddpg_tau: 0.01
  ddpg_gamma: 0.99
  
  # General training
  total_timesteps: 400000
  eval_freq: 10000
  plot_freq: 2000
  save_freq: 50000

# Progressive difficulty thresholds
difficulty_progression:
  medium_threshold: 100000     # Switch to medium after 100k steps
  hard_threshold: 250000       # Switch to hard after 250k steps
  performance_threshold_medium: 5.0   # Average reward needed for medium
  performance_threshold_hard: 8.0     # Average reward needed for hard

# Rendering and visualization settings
rendering:
  window_width_offset: 200     # Extra width for UI panels
  window_height_offset: 150    # Extra height for UI panels
  fps: 60                      # Target FPS for rendering
  show_zones: false            # Show strategic zones by default
  show_velocities: false       # Show velocity vectors by default
  show_field_info: true        # Show field information panel
  show_opponent_behavior: true # Show opponent behavior indicator
  # Testing mode settings (faster gameplay for testing)
  testing_mode: true           # Enable faster speeds for testing
  testing_speed_multiplier: 3.0  # Speed up everything by this factor
  testing_fps: 30              # Lower FPS for testing (smoother on slower PCs)

# Custom field dimensions (only used if field_type is 'custom')
custom_dimensions:
  field_length: 4.0            # Custom field length in meters
  field_width: 4.0             # Custom field width in meters
  goal_depth: 0.6              # Custom goal depth in meters
  goal_width: 0.8              # Custom goal width in meters
  goal_height: 1.0             # Custom goal height in meters
  goal_area_length: 0.5        # Custom goal area length in meters
  goal_area_width: 1.5         # Custom goal area width in meters
  penalty_mark_distance: 1.0   # Custom penalty mark distance in meters
  centre_circle_diameter: 1.0  # Custom centre circle diameter in meters
  border_strip_width: 0.5      # Custom border strip width in meters
  penalty_area_length: 1.0     # Custom penalty area length in meters
  penalty_area_width: 2.5      # Custom penalty area width in meters
  line_width: 0.05             # Custom line width in meters