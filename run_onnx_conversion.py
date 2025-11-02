#!/usr/bin/env python3
"""
ONNX Conversion Script for DDPG Model
Converts best DDPG model to ONNX format and validates performance
"""

from src.utils.onnx_converter import convert_sb3_to_onnx, validate_onnx_model, test_onnx_inference_speed
from stable_baselines3 import DDPG
import os

# Best performing DDPG model (best_model from training)
model_path = 'experiments/archives/multi_model_training_20250910_213419/models/ddpg/best_model.zip'
model_type = 'DDPG'
config_path = 'configs/field_config.yaml'

print('='*70)
print('ONNX Conversion for DDPG Best Model')
print('='*70)
print(f'Model: {model_path}')
print('='*70)

# Step 1: Convert to ONNX
print("\n[1/4] Converting to ONNX format...")
onnx_path = convert_sb3_to_onnx(model_path, model_type, config_path=config_path)

if onnx_path:
    print(f'\n✅ Conversion successful: {onnx_path}')

    # Step 2: Validate
    print('\n' + '='*70)
    print('[2/4] Validating ONNX model accuracy...')
    print('='*70)
    original_model = DDPG.load(model_path)
    is_valid = validate_onnx_model(onnx_path, original_model, model_type, config_path)

    # Step 3: Test inference speed
    print('\n' + '='*70)
    print('[3/4] Testing inference speed...')
    print('='*70)
    speed_results = test_onnx_inference_speed(onnx_path, config_path)

    # Step 4: Check file size
    print('\n' + '='*70)
    print('[4/4] Analyzing file sizes...')
    print('='*70)
    print('📊 Model File Sizes:')
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    print(f'  Original (.zip): {original_size:.2f} MB')
    print(f'  ONNX (.onnx):    {onnx_size:.2f} MB')
    print(f'  Size ratio:      {(onnx_size/original_size)*100:.1f}%')

    # Summary
    print('\n' + '='*70)
    print('🎯 ONNX CONVERSION SUMMARY')
    print('='*70)
    print(f'Model:              {model_path}')
    print(f'Output:             {onnx_path}')
    print(f'Validation:         {"✅ PASSED" if is_valid else "❌ FAILED"}')
    if speed_results:
        print(f'Inference speed:    {speed_results["avg_time_ms"]:.3f} ms')
        print(f'Max frequency:      {speed_results["max_frequency"]:.1f} Hz')
        print(f'50Hz requirement:   {"✅ MET" if speed_results["meets_50hz"] else "❌ NOT MET"}')
    print(f'File size:          {onnx_size:.2f} MB')
    print('='*70)

    # Save results to file
    results_file = 'onnx_conversion_results.txt'
    with open(results_file, 'w') as f:
        f.write('='*70 + '\n')
        f.write('ONNX CONVERSION RESULTS\n')
        f.write('='*70 + '\n\n')
        f.write(f'Model Path:         {model_path}\n')
        f.write(f'Model Type:         {model_type}\n')
        f.write(f'ONNX Output:        {onnx_path}\n\n')
        f.write('VALIDATION RESULTS:\n')
        f.write(f'  Accuracy:         {"PASSED" if is_valid else "FAILED"}\n')
        f.write(f'  Tolerance:        1e-4\n\n')
        f.write('PERFORMANCE RESULTS:\n')
        if speed_results:
            f.write(f'  Avg inference:    {speed_results["avg_time_ms"]:.3f} ms\n')
            f.write(f'  Max frequency:    {speed_results["max_frequency"]:.1f} Hz\n')
            f.write(f'  50Hz target:      {"MET" if speed_results["meets_50hz"] else "NOT MET"}\n')
            f.write(f'  Test iterations:  {speed_results["num_tests"]}\n\n')
        f.write('FILE SIZE:\n')
        f.write(f'  Original:         {original_size:.2f} MB\n')
        f.write(f'  ONNX:             {onnx_size:.2f} MB\n')
        f.write(f'  Size ratio:       {(onnx_size/original_size)*100:.1f}%\n')
        f.write('='*70 + '\n')

    print(f'\n📝 Results saved to: {results_file}')
else:
    print('❌ Conversion failed')
