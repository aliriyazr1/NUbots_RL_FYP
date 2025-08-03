# NUbots_RL_FYP (Soccer RL Training System)
# Soccer RL Training System

A comprehensive reinforcement learning system for training soccer-playing robots using PPO and DDPG algorithms with curriculum learning and difficulty progression.

## 📁 Project Structure

```
your_project_folder/
├── SoccerEnv/
│   └── soccerenv.py          # Core environment (Gymnasium-compatible)
├── train_soccer_rl.py        # Training script with curriculum learning
├── test_trained_model.py     # Model testing and visualisation
├── models/                   # Created automatically - stores best checkpoints
├── logs/                     # Created automatically - stores training logs
├── soccer_rl_comparison.png  # Created automatically - performance plots
└── README.md                 # This file
```

## 🎯 Quick Start

### 1. Training a Model
```bash
python train_soccer_rl.py
```
**Available Options:**
1. **Train PPO only** (Recommended as: 1) it trains faster than DDPG 2) Is it the better algorithm TODO: Figure it out me )
2. **Train DDPG only** (Alternative algorithm)
3. **Train both algorithms** (Full comparison)
4. **Evaluate existing models** (Test already trained models)
5. **Full pipeline** (Train both + automatic evaluation)

### 2. Testing Your Trained Model
```bash
python test_trained_model.py
```
**Available Options:**
1. **Test PPO model** (Watch PPO agent play)
2. **Test DDPG model** (Watch DDPG agent play)
3. **Compare both models** (Side-by-side comparison across difficulties)
4. **Test random baseline** (See what random actions look like)

### 3. Testing the Environment
```bash
cd SoccerEnv
python soccerenv.py
```
Runs a visual test of the environment with simple ball-chasing strategy.

## 📋 File Descriptions

### `SoccerEnv/soccerenv.py` - The Soccer Environment
**Purpose:** Defines the 2D soccer simulation environment where the robot learns.

**Key Features:**
- **Difficulty-based curriculum:** `easy`, `medium`, `hard` with progressive challenge
- **Anti-edge camping rewards:** Robot learns to aim for goal center, not field edges
- **Proper observation bounds:** All values normalised to [-1, 1] to prevent NaN crashes
- **Visual feedback:** Goal alignment indicators and possession status

**Environment Parameters by Difficulty:**
```python
# Easy Mode - Learning basics
max_steps = 150
possession_distance = 50.0    # Easier to get ball
collision_distance = 20.0     # More forgiving collisions

# Medium Mode - Intermediate challenge
max_steps = 200
possession_distance = 40.0
collision_distance = 25.0

# Hard Mode - Advanced play
max_steps = 250
possession_distance = 35.0    # Harder to get ball
collision_distance = 30.0     # Less forgiving
```

**Action Space:** 3D continuous actions (all in range [-1, 1])
- `action[0]`: Forward/backward movement
- `action[1]`: Left/right strafe movement
- `action[2]`: Rotation (turning)

**Observation Space:** 12D vector (all normalised to [-1, 1])
- Robot position (x, y) - normalised field coordinates
- Robot angle - current facing direction
- Ball position (x, y) - ball location on field
- Opponent position (x, y) - enemy robot location
- Robot velocity (x, y) - current movement speed
- Ball distance - how far robot is from ball
- Goal distance - how far robot is from goal center
- Ball possession flag - whether robot has ball (0 or 1)

**Reward System (Fixed to Prevent Edge Camping):**
- **Ball possession:** +2.0 (encourages getting ball)
- **Goal alignment:** +5.0 when robot with ball is in goal scoring area
- **Goal distance progress:** Progressive reward for approaching goal center
- **Edge camping penalty:** -1.0 for staying near field boundaries
- **Wrong Y-position penalty:** -1.0 for going to top/bottom edges instead of goal
- **Collision penalty:** -1.0 for hitting opponent
- **Scoring reward:** +15.0 for successfully scoring goal

### `train_soccer_rl.py` - Training Script
**Purpose:** Trains RL models using curriculum learning with automatic difficulty progression.

**Training Options Explained:**

#### Option 1: Train PPO Only
- **Algorithm:** Proximal Policy Optimization
- **Duration:** ~25-35 minutes (250k timesteps) (TODO: fix this duration)
- **Curriculum:** Automatic Easy→Medium→Hard progression
- **Best for:** Most users, most stable results

#### Option 2: Train DDPG Only  
- **Algorithm:** Deep Deterministic Policy Gradient
- **Duration:** ~20-30 minutes (250k timesteps) (TODO: fix this duration)
- **Difficulty:** Fixed medium difficulty
- **Best for:** Comparing against PPO, alternative approach

#### Option 3: Train Both Algorithms
- **Combines:** Both PPO and DDPG training
- **Duration:** ~45-60 minutes total
- **Result:** Complete comparison data
- **Best for:** Research, thorough evaluation

#### Option 4: Evaluate Existing Models
- **Purpose:** Test previously trained models
- **Duration:** ~5-10 minutes
- **Requirements:** Existing .zip model files
- **Includes:** Random baseline comparison

#### Option 5: Full Pipeline
- **Process:** Train both → Automatic evaluation → Comparison plots
- **Duration:** ~60-75 minutes total
- **Result:** Complete analysis with performance charts
- **Best for:** Complete experimental runs

**Curriculum Learning Features:**
- **Automatic progression:** Both training AND evaluation environments advance together
- **Progress tracking:** Real-time messages showing difficulty increases
- **Threshold-based:** Easy→Medium at avg reward >6, Medium→Hard at avg reward >10
- **Environment synchronisation:** Training and evaluation difficulties stay matched

**Hyperparameters (Optimised for Stability):**
```python
# PPO Settings
learning_rate = 1e-3         # Higher for faster learning
n_steps = 2048              # Larger rollouts
batch_size = 64             # Stable batch size
n_epochs = 5                # Sufficient updates
gamma = 0.97               # Slightly lower discount
clip_range = 0.15          # Tighter clipping
ent_coef = 0.1             # Higher exploration

# DDPG Settings  
learning_rate = 1e-3        # Standard for DDPG
buffer_size = 100000        # Large replay buffer
learning_starts = 2000      # More initial exploration
batch_size = 128           # Larger batches
```

### `test_trained_model.py` - Model Testing & Analysis
**Purpose:** Load and visually test trained models with comprehensive performance analysis.

**Testing Options Explained:**

#### Option 1: Test PPO Model
- **Model:** `soccer_rl_ppo_final.zip`
- **Episodes:** 5 test games on medium difficulty
- **Metrics:** Goal rate, ball possession, progress analysis
- **Visual:** Watch robot play with real-time feedback

#### Option 2: Test DDPG Model
- **Model:** `soccer_rl_ddpg_final.zip`  
- **Episodes:** 5 test games on medium difficulty
- **Comparison:** Performance vs PPO benchmarks
- **Analysis:** Same detailed metrics as PPO

#### Option 3: Compare Both Models
- **Process:** Tests both PPO and DDPG across all difficulties
- **Episodes:** 3 episodes per model per difficulty (18 total)
- **Difficulties:** Easy, Medium, Hard progression
- **Output:** Comprehensive comparison summary

#### Option 4: Test Random Baseline
- **Purpose:** Establish performance floor
- **Behavior:** Completely random actions
- **Duration:** One episode with env.max_steps
- **Use:** Verify trained models outperform random actions

**Analysis Metrics Provided:**
- **Goal Scoring Rate:** Percentage of episodes ending in successful goals
- **Ball Possession Rate:** Percentage of episodes where robot gets ball
- **Average Episode Reward:** Overall performance metric
- **Progress to Goal:** Maximum advancement toward goal area
- **Ball Possession Time:** Total steps robot controlled ball
- **FYP Success Assessment:** Meets >60% goal scoring requirement

**Performance Interpretation:**
```
🏆 EXCELLENT Performance:
- Goal scoring: ≥60% (meets FYP requirement)
- Ball possession: ≥80%
- Average reward: ≥15

👍 GOOD Performance:
- Goal scoring: 30-59%
- Ball possession: 50-79%
- Average reward: 5-14

⚠️ NEEDS WORK:
- Goal scoring: <30%
- Ball possession: <50%
- Average reward: <5
```

## 💾 Generated Files and Directories

### Automatically Created Directories
```bash
models/                           # Created when training starts
├── best_model.zip               # Best performing model during training
└── (timestamped checkpoints)    # Various evaluation snapshots

logs/                            # Created when training starts
├── evaluations.npz             # Evaluation metrics over time
├── progress.csv                # Episode rewards and statistics
└── (TensorBoard files)         # Detailed training logs
```
TODO: Confirm whether progress.csv and other TensorBoard files get created

### Model Files (Saved in Root Directory)

#### PPO Models (Primary Recommendation)
```bash
# Main models from training options
soccer_rl_ppo_final.zip              # Option 1: Standard PPO training

# Backup/recovery models
soccer_rl_ppo_partial.zip            # Saved if training interrupted
```

#### DDPG Models (Alternative Algorithm)
```bash
# Main models  
soccer_rl_ddpg_final.zip             # Option 2: Standard DDPG training

# Backup models
soccer_rl_ddpg_partial.zip           # Saved if training interrupted
```

#### Comparison and Analysis Files
```bash
soccer_rl_comparison.png            # Performance comparison chart
                                    # (Generated by Option 4 or 5)
```

## 🎮 Complete Usage Workflows

### Beginner Workflow (Recommended)
```bash
# 1. Train your first model (25-35 minutes)
python train_soccer_rl.py
# Choose: 1 (Train PPO only)
# Watch for: "🎯 PROGRESSING TO MEDIUM DIFFICULTY!" messages

# 2. Test your trained model (5 minutes)
python test_trained_model.py  
# Choose: 1 (Test PPO model)
# Look for: >30% goal scoring rate for good performance

# 3. If performance is poor, try more training or different settings
```

### Advanced Research Workflow
```bash
# 1. Complete algorithm comparison (60-75 minutes)
python train_soccer_rl.py
# Choose: 5 (Full pipeline)
# Result: Both models trained + automatic comparison

# 2. Detailed cross-difficulty testing (15 minutes)
python test_trained_model.py
# Choose: 3 (Compare both models)
# Result: Performance across easy/medium/hard difficulties

# 3. Analyse results and iterate
# Check soccer_rl_comparison.png for performance charts
```

### Quick Testing Workflow
```bash
# 1. Test environment setup (2 minutes)
cd SoccerEnv
python soccerenv.py
# Verify: Environment runs without errors

# 2. Quick model test (if models exist) (3 minutes)
python test_trained_model.py
# Choose: 1 (Test PPO) for quick verification

# 3. Compare with baseline (2 minutes)  
python test_trained_model.py
# Choose: 4 (Random baseline) to see untrained performance
```

## 📊 Expected Training Progress & Timelines

### Curriculum Learning Progression (Option 1)
```
🎯 EASY MODE (Steps 0-30k, ~8-12 minutes):
├── 0-10k:    Robot learns basic movement and ball approach
├── 10k-20k:  Consistent ball possession (>50%)
└── 20k-30k:  Occasional goal attempts
    📈 Target: avg reward >6 to progress

🎯 MEDIUM MODE (Steps 30k-80k, ~15-20 minutes):
├── 30k-40k:  Adapting to smarter opponent
├── 40k-60k:  Learning goal-directed movement  
├── 60k-80k:  Regular goal scoring (20-40%)
└── 80k+:     Consistent performance
    📈 Target: avg reward >10 to progress

🎯 HARD MODE (Steps 80k-250k, ~20-30 minutes):
├── 80k-120k: Advanced opponent strategies
├── 120k-200k: Optimizing goal scoring (40-60%+)
└── 200k-250k: Peak performance refinement
    📈 Target: >60% goal rate (FYP requirement)
```

### DDPG Training Timeline (Option 2)
```
🎯 MEDIUM MODE ONLY (Steps 0-250k, ~20-30 minutes):
├── 0-50k:    Learning basic ball control
├── 50k-150k: Developing goal-scoring strategies
├── 150k-250k: Performance optimization
    📈 Target: Competitive with PPO performance
```

### Full Pipeline Timeline (Option 5)
```
🎯 COMPLETE WORKFLOW (~60-75 minutes total):
├── PPO Training:     ~30-40 minutes (with curriculum)
├── DDPG Training:    ~20-30 minutes (medium difficulty)
├── Evaluation:       ~5-10 minutes (automated)
└── Plot Generation:  ~1-2 minutes
    📊 Result: Complete performance analysis
```

## 🔧 Troubleshooting Guide

### Common Training Issues

#### Training Crashes with NaN Error
```bash
# Error: "ValueError: Expected parameter loc... tensor([[nan, nan, nan]..."
# Cause: Rewards too high or learning rate too aggressive
# Solution: Environment has built-in reward clipping (-3.0 to 20.0)
# Check: Training script uses stable learning rates (1e-3)
```

#### No Curriculum Progression Messages
```bash
# Problem: No "🎯 PROGRESSING TO X DIFFICULTY!" messages after 30k steps
# Cause: ProgressTracker not updating both environments
# Solution: Current train_soccer_rl.py updates both train and eval environments
# Check: Look for progress messages every 10k steps
```

#### Robot Camps at Field Edges
```bash
# Problem: Robot goes to field edges instead of goal center
# Cause: Poor reward function encouraging edge behavior
# Solution: Environment has anti-edge camping penalties
# Check: Watch for "GOAL ALIGNED!" messages during testing
```

#### Low Goal Scoring Rate (<20%)
```bash
# Causes: Episodes too long, poor reward shaping, wrong hyperparameters
# Solutions:
# 1. Use curriculum learning (start easy mode)
# 2. Verify reward function includes goal alignment bonuses
# 3. Check episode length (150-250 steps based on difficulty)
# 4. Ensure training reaches 100k+ timesteps
```

### Model Loading Issues

#### Model File Not Found
```bash
# Error: "Model file 'soccer_rl_ppo_final.zip' not found!"
# Solution: Check exact filename, complete training first
# Default names: 
#   - soccer_rl_ppo_final.zip (from Option 1)
#   - soccer_rl_ddpg_final.zip (from Option 2)
```

#### Import Error for SoccerEnv
```bash
# Error: "ModuleNotFoundError: No module named 'SoccerEnv.soccerenv'"
# Solution: Ensure SoccerEnv/ folder exists with soccerenv.py inside
# Structure: SoccerEnv/soccerenv.py (not just soccerenv.py)
```

### Performance Issues

#### Training Too Slow
```bash
# Causes: Large networks, long episodes, high-resolution rendering
# Solutions:
# 1. Use provided hyperparameters (optimized for speed/stability)
# 2. Run training without visual rendering
# 3. Reduce total_timesteps for quick testing (100k instead of 250k)
```

#### Poor Visual Performance During Testing
```bash
# Causes: Pygame rendering issues, slow computer
# Solutions:
# 1. Adjust time.sleep() values in test_trained_model.py
# 2. Reduce number of test episodes
# 3. Test without visual rendering first
```

## 💡 Tips for Success

### Training Best Practices
1. **Start with Option 1** (PPO curriculum) - Most reliable results
2. **Monitor progress messages** - Should see difficulty progression
3. **Check visual behavior** - Robot should move toward goal center
4. **Be patient** - Good performance typically after 100k+ steps
5. **Save partial models** - Training auto-saves if interrupted

### Testing Best Practices  
1. **Test across difficulties** - Ensures robust learning
2. **Check ball possession first** - Should be >60% before expecting goals
3. **Watch for goal alignment** - Robot should aim for goal center
4. **Compare with random baseline** - Trained model should be much better

### Performance Optimization
1. **Curriculum learning** - Much more effective than single difficulty
2. **Proper reward inspection** - Watch for positive trend in episode rewards
3. **Visual verification** - Robot behavior should look intelligent
4. **Multiple test runs** - Performance can vary between episodes

## 🎯 Success Indicators & Benchmarks

### During Training ✅
**Good Signs:**
- Progress messages every 10k steps: "📈 Step X: Recent avg reward = Y"
- Curriculum advancement: "🎯 PROGRESSING TO MEDIUM/HARD DIFFICULTY!"
- Increasing episode rewards over time (trending upward)
- No NaN errors or training crashes
- Training completes without interruption

**Warning Signs ❌:**
- No progress messages after 20k steps
- Consistently negative or decreasing rewards
- NaN errors causing crashes
- No curriculum progression after 50k steps
- Training stuck in easy mode

### During Testing ✅
**Excellent Performance (FYP Success):**
- Goal scoring rate: ≥60%
- Ball possession rate: ≥80%
- Average episode reward: ≥15
- Robot consistently moves toward goal center
- "GOAL ALIGNED!" messages appear frequently

**Good Performance:**
- Goal scoring rate: 30-59%
- Ball possession rate: 50-79%  
- Average episode reward: 5-14
- Robot gets ball consistently
- Some goal-directed movement

**Needs Improvement:**
- Goal scoring rate: <30%
- Ball possession rate: <50%
- Average episode reward: <5
- Robot behavior looks random
- Frequent edge camping

### File Structure Verification ✅
**Correct Setup:**
```
✅ SoccerEnv/soccerenv.py exists
✅ train_soccer_rl.py in root directory
✅ test_trained_model.py in root directory
✅ Can run: python train_soccer_rl.py without import errors
✅ Can run: cd SoccerEnv && python soccerenv.py
```

## 📞 Getting Help

### First Steps for Issues:
1. **Verify file structure** - Ensure SoccerEnv/ folder with soccerenv.py
2. **Test environment first** - Run `cd SoccerEnv && python soccerenv.py`
3. **Check dependencies** - Ensure stable_baselines3, pygame, gymnasium installed
4. **Review error messages** - Most errors indicate missing files or import issues

### Training Issues:
1. **Start with quick test** - Use reduced timesteps (50k) to verify setup
2. **Monitor progress** - Look for curriculum progression messages
3. **Check disk space** - Models and logs require storage space
4. **Verify performance** - Test with random baseline first

### Model Performance Issues:
1. **Training duration** - Ensure sufficient timesteps (200k+ recommended)
2. **Curriculum completion** - Verify robot reached hard difficulty
3. **Visual inspection** - Watch robot behavior during testing
4. **Comparison testing** - Use Option 3 to compare across difficulties

This comprehensive guide covers most aspects of the soccer RL training system, from basic usage to advanced troubleshooting, ensuring successful training and testing of soccer-playing robots!