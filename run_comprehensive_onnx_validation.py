#!/usr/bin/env python3
"""
Comprehensive ONNX Validation for Thesis Tables
Extracts numerical accuracy and latency metrics
"""

import numpy as np
import torch
import onnxruntime as ort
from stable_baselines3 import DDPG
import time
from scipy import stats

# Model paths
original_model_path = 'experiments/archives/multi_model_training_20250910_213419/models/ddpg/best_model.zip'
onnx_model_path = 'models/best_model.onnx'

print('='*80)
print('COMPREHENSIVE ONNX VALIDATION FOR THESIS')
print('='*80)
print(f'Original model: {original_model_path}')
print(f'ONNX model:     {onnx_model_path}')
print('='*80)

# Load models
print('\n[1/4] Loading models...')
original_model = DDPG.load(original_model_path)
ort_session = ort.InferenceSession(onnx_model_path)
print('✅ Models loaded')

# ===================================================================
# NUMERICAL ACCURACY TESTS
# ===================================================================
print('\n' + '='*80)
print('[2/4] NUMERICAL ACCURACY TESTS')
print('='*80)

# Test 1: Single observation
print('\nTest 1: Single observation accuracy')
test_obs_single = np.random.randn(12).astype(np.float32)
pytorch_output_single, _ = original_model.predict(test_obs_single, deterministic=True)
onnx_input_single = test_obs_single.reshape(1, 12)
onnx_output_single = ort_session.run(None, {'observation': onnx_input_single})[0][0]
diff_single = np.abs(pytorch_output_single - onnx_output_single).max()
print(f'  Max difference: {diff_single:.10f}')

# Test 2: 100 random observations
print('\nTest 2: 100 random observations accuracy')
diffs_100 = []
for i in range(100):
    test_obs = np.random.randn(12).astype(np.float32)
    pytorch_output, _ = original_model.predict(test_obs, deterministic=True)
    onnx_input = test_obs.reshape(1, 12)
    onnx_output = ort_session.run(None, {'observation': onnx_input})[0][0]
    diff = np.abs(pytorch_output - onnx_output).max()
    diffs_100.append(diff)

max_diff_100 = np.max(diffs_100)
mean_diff_100 = np.mean(diffs_100)
print(f'  Maximum difference: {max_diff_100:.10f}')
print(f'  Mean difference:    {mean_diff_100:.10f}')
print(f'  Std difference:     {np.std(diffs_100):.10f}')

# Test 3: Deterministic inference (same input 50 times)
print('\nTest 3: Deterministic inference (50 runs, same input)')
test_obs_fixed = np.random.randn(12).astype(np.float32)
onnx_input_fixed = test_obs_fixed.reshape(1, 12)
outputs_50 = []
for i in range(50):
    onnx_output = ort_session.run(None, {'observation': onnx_input_fixed})[0][0]
    outputs_50.append(onnx_output)

outputs_50 = np.array(outputs_50)
variance_deterministic = np.var(outputs_50, axis=0).max()
print(f'  Maximum variance across 50 runs: {variance_deterministic:.15f}')
print(f'  All outputs identical: {np.allclose(outputs_50, outputs_50[0])}')

# ===================================================================
# LATENCY BENCHMARKS
# ===================================================================
print('\n' + '='*80)
print('[3/4] LATENCY BENCHMARKS (1,000 inferences)')
print('='*80)

# Warm-up
print('\nWarming up (100 iterations)...')
for _ in range(100):
    test_input = np.random.randn(1, 12).astype(np.float32)
    _ = ort_session.run(None, {'observation': test_input})

# Benchmark 1,000 inferences with individual timing
print('\nRunning 1,000 timed inferences...')
latencies_ms = []
for i in range(1000):
    test_input = np.random.randn(1, 12).astype(np.float32)

    start = time.perf_counter()
    _ = ort_session.run(None, {'observation': test_input})
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies_ms.append(latency_ms)

latencies_ms = np.array(latencies_ms)

# Calculate statistics
mean_latency = np.mean(latencies_ms)
median_latency = np.median(latencies_ms)
p95_latency = np.percentile(latencies_ms, 95)
p99_latency = np.percentile(latencies_ms, 99)
max_latency = np.max(latencies_ms)
min_latency = np.min(latencies_ms)
std_latency = np.std(latencies_ms)

# Convert to frequencies (Hz)
mean_freq = 1000.0 / mean_latency
median_freq = 1000.0 / median_latency
p95_freq = 1000.0 / p95_latency
p99_freq = 1000.0 / p99_latency
max_freq = 1000.0 / min_latency  # Note: max freq = 1/min latency
min_freq = 1000.0 / max_latency

# Safety margin
target_latency_ms = 20.0  # 50 Hz requirement
safety_margin = target_latency_ms / mean_latency

print(f'\nLatency statistics:')
print(f'  Mean:     {mean_latency:.6f} ms')
print(f'  Median:   {median_latency:.6f} ms')
print(f'  Std dev:  {std_latency:.6f} ms')
print(f'  Min:      {min_latency:.6f} ms')
print(f'  Max:      {max_latency:.6f} ms')
print(f'  95th percentile: {p95_latency:.6f} ms')
print(f'  99th percentile: {p99_latency:.6f} ms')

print(f'\nFrequency statistics:')
print(f'  Mean:     {mean_freq:.2f} Hz')
print(f'  Median:   {median_freq:.2f} Hz')
print(f'  95th pct: {p95_freq:.2f} Hz')
print(f'  99th pct: {p99_freq:.2f} Hz')
print(f'  Maximum:  {max_freq:.2f} Hz')
print(f'  Minimum:  {min_freq:.2f} Hz')

print(f'\nSafety margin vs 20ms requirement: {safety_margin:.2f}×')

# ===================================================================
# THESIS-READY OUTPUT
# ===================================================================
print('\n' + '='*80)
print('[4/4] THESIS-READY FORMATTED OUTPUT')
print('='*80)

print('\n' + '-'*80)
print('NUMERICAL ACCURACY METRICS')
print('-'*80)
print(f'Single observation max difference:    {diff_single:.10f}')
print(f'100 random observations max diff:     {max_diff_100:.10f}')
print(f'100 random observations mean diff:    {mean_diff_100:.10f}')
print(f'Deterministic inference variance:     {variance_deterministic:.15f}')
print(f'Tolerance threshold:                  1e-4 (0.0001)')
print(f'Accuracy validation:                  {"✅ PASSED" if max_diff_100 < 1e-4 else "❌ FAILED"}')

print('\n' + '-'*80)
print('LATENCY BENCHMARKS (1,000 inferences)')
print('-'*80)
print(f'Mean latency:         {mean_latency:.6f} ms  ({mean_freq:>10.2f} Hz)')
print(f'Median latency:       {median_latency:.6f} ms  ({median_freq:>10.2f} Hz)')
print(f'95th percentile:      {p95_latency:.6f} ms  ({p95_freq:>10.2f} Hz)')
print(f'99th percentile:      {p99_latency:.6f} ms  ({p99_freq:>10.2f} Hz)')
print(f'Best case (fastest):  {min_latency:.6f} ms  ({max_freq:>10.2f} Hz)')
print(f'Worst case (slowest): {max_latency:.6f} ms  ({min_freq:>10.2f} Hz)')
print(f'Standard deviation:   {std_latency:.6f} ms')
print(f'\nTarget requirement:   20.000000 ms  (      50.00 Hz)')
print(f'Safety margin:        {safety_margin:.2f}× faster than requirement')
print(f'Latency validation:   {"✅ MEETS 50Hz" if mean_latency < 20.0 else "❌ FAILS 50Hz"}')

# ===================================================================
# LATEX TABLE FORMAT
# ===================================================================
print('\n' + '='*80)
print('LATEX TABLE FORMAT')
print('='*80)

print('\n% Numerical Accuracy Table')
print('\\begin{table}[h]')
print('\\centering')
print('\\caption{ONNX Model Numerical Accuracy Validation}')
print('\\begin{tabular}{lr}')
print('\\toprule')
print('\\textbf{Metric} & \\textbf{Value} \\\\')
print('\\midrule')
print(f'Single observation max difference & ${diff_single:.2e}$ \\\\')
print(f'100 random observations max diff & ${max_diff_100:.2e}$ \\\\')
print(f'100 random observations mean diff & ${mean_diff_100:.2e}$ \\\\')
print(f'Deterministic inference variance & ${variance_deterministic:.2e}$ \\\\')
print(f'Tolerance threshold & $1 \\times 10^{{-4}}$ \\\\')
print('\\bottomrule')
print('\\end{tabular}')
print('\\end{table}')

print('\n% Latency Benchmark Table')
print('\\begin{table}[h]')
print('\\centering')
print('\\caption{ONNX Model Inference Latency Benchmarks (1,000 iterations)}')
print('\\begin{tabular}{lrr}')
print('\\toprule')
print('\\textbf{Statistic} & \\textbf{Latency (ms)} & \\textbf{Frequency (Hz)} \\\\')
print('\\midrule')
print(f'Mean & ${mean_latency:.3f}$ & ${mean_freq:.2f}$ \\\\')
print(f'Median & ${median_latency:.3f}$ & ${median_freq:.2f}$ \\\\')
print(f'95th percentile & ${p95_latency:.3f}$ & ${p95_freq:.2f}$ \\\\')
print(f'99th percentile & ${p99_latency:.3f}$ & ${p99_freq:.2f}$ \\\\')
print(f'Best case (fastest) & ${min_latency:.3f}$ & ${max_freq:.2f}$ \\\\')
print(f'Worst case (slowest) & ${max_latency:.3f}$ & ${min_freq:.2f}$ \\\\')
print('\\midrule')
print(f'Target (50 Hz) & $20.000$ & $50.00$ \\\\')
print(f'Safety margin & \\multicolumn{{2}}{{c}}{{${safety_margin:.2f}\\times$ faster}} \\\\')
print('\\bottomrule')
print('\\end{tabular}')
print('\\end{table}')

# Save results
print('\n' + '='*80)
print('Saving results to file...')
with open('onnx_validation_thesis_metrics.txt', 'w') as f:
    f.write('='*80 + '\n')
    f.write('ONNX VALIDATION METRICS FOR THESIS\n')
    f.write('='*80 + '\n\n')

    f.write('NUMERICAL ACCURACY:\n')
    f.write(f'- Single observation max diff: {diff_single:.10f}\n')
    f.write(f'- 100 random observations max diff: {max_diff_100:.10f}\n')
    f.write(f'- Deterministic test (50 runs) variance: {variance_deterministic:.15f}\n\n')

    f.write('LATENCY BENCHMARKS (1000 inferences):\n')
    f.write(f'- Mean: {mean_latency:.6f} ms ({mean_freq:.2f} Hz)\n')
    f.write(f'- Median: {median_latency:.6f} ms ({median_freq:.2f} Hz)\n')
    f.write(f'- 95th percentile: {p95_latency:.6f} ms ({p95_freq:.2f} Hz)\n')
    f.write(f'- 99th percentile: {p99_latency:.6f} ms ({p99_freq:.2f} Hz)\n')
    f.write(f'- Best case (fastest): {min_latency:.6f} ms ({max_freq:.2f} Hz)\n')
    f.write(f'- Worst case (slowest): {max_latency:.6f} ms ({min_freq:.2f} Hz)\n')
    f.write(f'- Safety margin vs 20ms requirement: {safety_margin:.2f}×\n')

print('✅ Results saved to: onnx_validation_thesis_metrics.txt')
print('='*80)
