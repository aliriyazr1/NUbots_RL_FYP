#!/bin/bash
# Quick script to test your specific DDPG model with all benchmarks

cd /home/aliriyazr1/NUbots_RL_FYP
source .venv/bin/activate

MODEL_PATH="experiments/archives/multi_model_training_20250910_213419/models/ddpg/retrained_ddpg_model_20250910_213419.zip"

echo "================================"
echo "Testing DDPG Model Performance"
echo "================================"
echo ""
echo "Model: $MODEL_PATH"
echo ""

# Run all performance benchmarks
python -m pytest tests/integration/test_performance_benchmarks.py \
  --ddpg-model="$MODEL_PATH" \
  -v -s

echo ""
echo "================================"
echo "Done!"
echo "================================"
