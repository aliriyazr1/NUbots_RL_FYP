# How to Test Your Models - Summary

## ✅ Changes Made

All test files now support loading your pre-trained models:

1. **test_model_evaluation.py** - Added DDPG support
   - `test_trained_vs_random_baseline_pretrained` - Tests PPO model
   - `test_trained_vs_random_baseline_pretrained_ddpg` - Tests DDPG model

2. **test_performance_benchmarks.py** - All tests now use pretrained models
   - `test_ball_possession_improvement` - Uses your model
   - `test_collision_frequency_reduction` - Uses your model
   - `test_goal_approach_success_improvement` - Uses your model

All tests automatically use PPO or DDPG model (whichever you provide).

## 🚀 Easiest Way to Run Tests

### Option 1: Use the One-Line Script

```bash
cd /home/aliriyazr1/NUbots_RL_FYP
./test_my_ddpg_model.sh
```

This runs all performance benchmarks on your DDPG model!

### Option 2: Set Environment Variable Once

```bash
cd /home/aliriyazr1/NUbots_RL_FYP
source .venv/bin/activate

# Set this once
export DDPG_MODEL_PATH=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip

# Then run any tests
python -m pytest tests/integration/test_model_evaluation.py -v
python -m pytest tests/integration/test_performance_benchmarks.py -v
python -m pytest tests/integration/ -v
```

### Option 3: Specify Model Each Time

```bash
cd /home/aliriyazr1/NUbots_RL_FYP
source .venv/bin/activate

python -m pytest tests/integration/test_performance_benchmarks.py \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s
```

## 📊 What Tests Are Available

### Model Evaluation Tests
```bash
# Test DDPG vs random baseline
python -m pytest tests/integration/test_model_evaluation.py::TestTrainedModelPerformance::test_trained_vs_random_baseline_pretrained_ddpg \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s
```

### Performance Benchmarks
```bash
# Ball possession benchmark
python -m pytest tests/integration/test_performance_benchmarks.py::TestBallPossessionBenchmark::test_ball_possession_improvement \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s

# Collision avoidance benchmark
python -m pytest tests/integration/test_performance_benchmarks.py::TestCollisionAvoidanceBenchmark::test_collision_frequency_reduction \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s

# Goal approach benchmark
python -m pytest tests/integration/test_performance_benchmarks.py::TestGoalApproachBenchmark::test_goal_approach_success_improvement \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s
```

### All Performance Benchmarks
```bash
python -m pytest tests/integration/test_performance_benchmarks.py \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s
```

## 🎯 Quick Commands Reference

```bash
# 1. Navigate to project
cd /home/aliriyazr1/NUbots_RL_FYP

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Run the easy script
./test_my_ddpg_model.sh

# OR set environment variable
export DDPG_MODEL_PATH=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip

# Then run whatever tests you want
python -m pytest tests/integration/test_model_evaluation.py -v
python -m pytest tests/integration/test_performance_benchmarks.py -v
```

## 💡 Understanding the Output

When you run tests, you'll see:

- **PASSED** ✅ - Test succeeded
- **FAILED** ❌ - Test failed (model didn't meet benchmark)
- **SKIPPED** ⏭️ - Test skipped (no model provided)

For benchmarks, you'll see output like:
```
Ball Possession Rate Results:
  Trained DDPG:    0.0234 ± 0.0123
  Random baseline: 0.0189 ± 0.0098
  Improvement:     +23.81%
  Target:          ≥15.00%
  ✓ Target achieved!
```

## 🔍 Troubleshooting

**"pytest: command not found"**
→ Activate virtual environment: `source .venv/bin/activate`

**"No pre-trained DDPG model available"**
→ Add `--ddpg-model=path/to/model.zip` or set `DDPG_MODEL_PATH`

**Tests are slow**
→ Normal! Each benchmark runs 20 episodes. Use `-v -s` to see progress.

## 📝 For Your Thesis/Report

Document which model you tested:
```
Performance benchmarks were conducted on the DDPG model trained
in experiment multi_model_training_20250910_213419
(retrained_ddpg_model_20250910_213419.zip).

Command used:
python -m pytest tests/integration/test_performance_benchmarks.py \
  --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  -v -s > benchmark_results.txt 2>&1
```
