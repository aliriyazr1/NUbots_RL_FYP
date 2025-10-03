# Reward Function Unit Tests

## Overview

Comprehensive unit tests for the reward function in the Soccer RL FYP project.

**Student:** Ali Riyaz (C3412624)
**Module:** `test_reward_function.py`
**Test Framework:** pytest

## Test Coverage

### 1. TestRewardFunctionBounds
Tests that reward values stay within expected bounds during gameplay.

- **test_reward_bounds_random_actions**: Verifies rewards stay roughly in [-10, +10] range over 1000 random steps (excluding terminal conditions)
- **test_reward_no_nan_or_inf**: Ensures reward function never returns NaN or infinity values

**Reference:** Testing plan requirement 4.1.2 - Reward bounds validation

### 2. TestRewardTerminalConditions
Tests terminal events produce appropriate large rewards/penalties.

- **test_goal_scoring_maximum_reward**: Validates scoring gives reward ≈+100 to +150
- **test_ball_out_of_bounds_penalty**: Confirms ball going out incurs significant penalty ≤-20

**Reference:** Terminal condition handling from reward function design

### 3. TestAntiExploitationMeasures
Tests that the reward function prevents degenerate learning behaviours.

- **test_stationary_ball_holding_penalty**: Verifies holding ball stationary for >30 steps incurs increasing penalty
- **test_wall_hugging_penalty**: Confirms staying near walls is not advantageous

**Theoretical basis:** Reward shaping should discourage stationary policies (Roijers & Whiteson, 2017)

### 4. TestRewardComponents
Tests individual reward components for expected behaviour.

- **test_ball_possession_reward**: Validates Gaussian reward for being close to ball
- **test_goal_progress_reward**: Confirms potential-based shaping rewards goal progress
- **test_reward_scenarios**: Parametrised tests for common game situations

**Mathematical basis:**
- Gaussian: G(d) = e^(-0.5(d/σ)²) for optimal positioning
- Potential shaping: Φ(s') - Φ(s) (Ng et al., 1999)

### 5. TestRewardScaling
Tests reward component balance and smoothness.

- **test_reward_component_balance**: Ensures no single component dominates excessively
- **test_reward_continuity**: Validates smooth reward transitions (no large discontinuities)

**Reference:** Multi-objective RL principles (Roijers & Whiteson, 2017)

## Running the Tests

### Prerequisites

```bash
# Install pytest if not already installed
pip install pytest

# Ensure you're in the project root directory
cd /home/aliriyazr1/NUbots_RL_FYP
```

### Run All Tests

```bash
# Verbose output with test details
pytest tests/unit/test_reward_function.py -v

# With detailed output including print statements
pytest tests/unit/test_reward_function.py -v -s

# Generate coverage report
pytest tests/unit/test_reward_function.py --cov=src.environments.soccerenv --cov-report=html
```

### Run Specific Test Classes

```bash
# Test only bounds checking
pytest tests/unit/test_reward_function.py::TestRewardFunctionBounds -v

# Test only anti-exploitation measures
pytest tests/unit/test_reward_function.py::TestAntiExploitationMeasures -v -s
```

### Run Individual Tests

```bash
# Test stationary ball holding penalty
pytest tests/unit/test_reward_function.py::TestAntiExploitationMeasures::test_stationary_ball_holding_penalty -v -s

# Test goal scoring reward
pytest tests/unit/test_reward_function.py::TestRewardTerminalConditions::test_goal_scoring_maximum_reward -v -s
```

## Test Output Interpretation

### Expected Output Format

```
tests/unit/test_reward_function.py::TestRewardFunctionBounds::test_reward_bounds_random_actions
Non-terminal reward statistics:
  Min: -8.42
  Max: 15.37
  Mean: 2.14
  Std: 4.21
PASSED

tests/unit/test_reward_function.py::TestAntiExploitationMeasures::test_stationary_ball_holding_penalty
✓ Stationary holding penalty working:
  Early average (steps 5-15): 3.21
  Late average (steps 35-45): -2.45
  Penalty increase: 5.66
PASSED
```

### Success Criteria

- **All tests pass**: Reward function meets testing plan requirements
- **Bounds respected**: Non-terminal rewards within reasonable range
- **Anti-exploitation works**: Penalties increase over time for bad behaviours
- **Smoothness verified**: No large discontinuous jumps in rewards

## Test Helper Functions

### `_place_entities(robot_pos, ball_pos, opponent_pos=None)`

Helper function to place robot, ball, and opponent at specific positions for deterministic testing.

**Parameters:**
- `robot_pos`: [x, y] position in pixels
- `ball_pos`: [x, y] position in pixels
- `opponent_pos`: [x, y] position in pixels (optional)

**Usage:**
```python
self._place_entities(
    robot_pos=[300, 300],
    ball_pos=[310, 300]
)
```

## Academic Contribution

These tests validate:

1. **Theoretical reward design** - Confirms mathematical functions (sigmoid, Gaussian, exponential) work as intended
2. **Anti-exploitation measures** - Demonstrates systematic approach to preventing degenerate policies
3. **Continuous optimisation** - Validates smoothness required for gradient-based RL algorithms
4. **Multi-objective balance** - Ensures no single objective dominates learning

**For thesis documentation:**
- Test results provide quantitative evidence of reward function design
- Statistical validation supports methodology chapter
- Systematic testing demonstrates rigorous engineering approach

## References

1. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

2. Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. *Synthesis Lectures on Artificial Intelligence and Machine Learning*.

3. Schulman, J., et al. (2017). Proximal policy optimisation algorithms. *arXiv preprint*.

## Notes

- Tests use `render_mode=None` for headless operation (faster execution)
- Config file loaded from `configs/field_config.yaml`
- Tests designed to be deterministic where possible
- Some tests use controlled entity placement for reproducibility
- Random tests use fixed seed for consistent results (if needed, set `np.random.seed(42)` in test)

## Troubleshooting

### Import Errors
```bash
# Ensure src is in path
export PYTHONPATH="${PYTHONPATH}:/home/aliriyazr1/NUbots_RL_FYP/src"
```

### Config File Not Found
```bash
# Run from project root
cd /home/aliriyazr1/NUbots_RL_FYP
pytest tests/unit/test_reward_function.py -v
```

### Pygame Warnings
Pygame warnings about `pkg_resources` can be safely ignored. They don't affect test execution.