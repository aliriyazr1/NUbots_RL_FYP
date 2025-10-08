# Integration Tests - Quick Start

## 🚀 Easiest Way: Use the GUI!

```bash
./tests/integration/run_tests_with_gui.sh
```

This opens a graphical interface to:
- Browse for specific model files
- Select from your trainGUI experiments
- Run tests with one click

**See:** [GUI_TESTING_GUIDE.md](GUI_TESTING_GUIDE.md) for full GUI documentation

---

## 📋 Other Methods

### 1. Command-Line Model Selection

```bash
# Test with specific model path
pytest --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip

# Test with both PPO and DDPG
pytest --ppo-model=path/to/ppo.zip --ddpg-model=path/to/ddpg.zip
```

### 2. Environment Variables (For Repeated Testing)

```bash
export PPO_MODEL_PATH=experiments/archives/.../my_ppo.zip
export DDPG_MODEL_PATH=experiments/archives/.../retrained_ddpg_model_20250910_213419.zip
pytest
```

### 3. Experiment Selection

```bash
# List all experiments
pytest --list-experiments

# Use specific experiment
pytest --experiment=multi_model_training_20250910_213419

# Use latest experiment
pytest --use-latest-experiment
```

---

## 📚 Available Test Suites

| Test Suite | Description | Run Command |
|------------|-------------|-------------|
| **Model Evaluation** | Test trained vs random, cross-difficulty | `pytest test_model_evaluation.py` |
| **Performance Benchmarks** | Ball possession, collisions, goal approach | `pytest test_performance_benchmarks.py` |
| **Statistical Validation** | PPO vs DDPG comparison, reproducibility | `pytest test_statistical_validation.py` |
| **Training Integration** | End-to-end training pipeline tests | `pytest test_training_integration.py` |

---

## 📖 Full Documentation

- **[GUI_TESTING_GUIDE.md](GUI_TESTING_GUIDE.md)** - Graphical interface guide
- **[MODEL_TESTING_GUIDE.md](MODEL_TESTING_GUIDE.md)** - Comprehensive model selection guide
- **[QUICK_START_EXAMPLES.sh](QUICK_START_EXAMPLES.sh)** - Copy-paste command examples
- **[TEST_FIX_NOTES.md](../TEST_FIX_NOTES.md)** - Academic notes on test fixes

---

## 🎯 Quick Examples

### Test your latest training run
```bash
./tests/integration/run_tests_with_gui.sh
# Select top experiment in GUI → Click "Run Tests"
```

### Test specific archived DDPG model
```bash
pytest --ddpg-model=experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip \
  tests/integration/test_model_evaluation.py -v -s
```

### Performance benchmarks on pre-trained model
```bash
./tests/integration/run_tests_with_gui.sh tests/integration/test_performance_benchmarks.py
# Browse for model → Click "Run Tests"
```

---

## 🔧 How It Works

All model selection methods feed into the same fixture system in [conftest.py](conftest.py).

**Priority Order:**
1. Command-line (`--ppo-model`, `--ddpg-model`)
2. Environment variables (`PPO_MODEL_PATH`, `DDPG_MODEL_PATH`)
3. GUI selection (`.test_model_selection.json`)
4. Experiment (`--experiment`, `--use-latest-experiment`)
5. Defaults (`models/ppo/...`, `models/ddpg/...`)

---

## 💡 Pro Tips

1. **GUI for exploration** - Quickly test different experiments
2. **Environment vars for development** - Set once, test repeatedly
3. **Command-line for CI/CD** - Precise control in scripts
4. **Keep `.test_model_selection.json`** - Documents which models were tested

---

## 🆘 Need Help?

```bash
# List available experiments
pytest --list-experiments

# Run with verbose output
pytest -v -s

# Run specific test
pytest test_model_evaluation.py::TestTrainedModelPerformance::test_trained_vs_random_baseline_pretrained -v -s
```

**Issues?** Check [GUI_TESTING_GUIDE.md](GUI_TESTING_GUIDE.md) → Troubleshooting section
