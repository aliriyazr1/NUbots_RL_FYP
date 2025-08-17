"""
ONNX Model Converter for Soccer RL Models
Author: Ali Riyaz (Extended for ONNX conversion)
Convert trained Stable-Baselines3 models to ONNX format for robot deployment
"""

import torch
import torch.onnx
import numpy as np
from stable_baselines3 import PPO, DDPG
import os
import yaml
from SoccerEnv.soccerenv import SoccerEnv
import onnx
import onnxruntime as ort

def get_model_input_shape(config_path="SoccerEnv/field_config.yaml"):
    """Get the correct input shape from the environment"""
    try:
        # Create a temporary environment to get observation space
        env = SoccerEnv(difficulty="easy", config_path=config_path)
        obs_space = env.observation_space
        input_shape = obs_space.shape
        env.close()
        return input_shape
    except Exception as e:
        print(f"⚠️  Could not determine input shape from environment: {e}")
        # Default fallback - you may need to adjust this based on your observation space
        return (12,)  # Adjust this based on your actual observation space size

def convert_sb3_to_onnx(model_path, model_type="PPO", output_path=None, config_path="SoccerEnv/field_config.yaml"):
    """
    Convert Stable-Baselines3 model to ONNX format
    
    Args:
        model_path: Path to the .zip model file (e.g., "soccer_rl_ppo_final.zip")
        model_type: "PPO" or "DDPG"
        output_path: Output ONNX file path (if None, will auto-generate)
        config_path: Path to field configuration
    """
    
    print(f"🔄 Converting {model_type} model: {model_path}")
    
    # Load the trained model
    try:
        if model_type.upper() == "PPO":
            model = PPO.load(model_path)
        elif model_type.upper() == "DDPG":
            model = DDPG.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"✅ Loaded {model_type} model successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get input shape from environment
    input_shape = get_model_input_shape(config_path)
    print(f"📏 Model input shape: {input_shape}")
    
    # Create dummy input tensor
    # Add batch dimension (1, *input_shape)
    dummy_input = torch.randn(1, *input_shape, dtype=torch.float32)
    print(f"🎯 Dummy input tensor shape: {dummy_input.shape}")
    
    # CRITICAL FIX: For PPO, we need to extract only the deterministic action part
    if model_type.upper() == "PPO":
        # Create a wrapper that only returns the mean action (deterministic)
        class DeterministicPPOWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
                
            def forward(self, obs):
                # Extract features
                features = self.policy.extract_features(obs)
                # Get policy features
                policy_features = self.policy.mlp_extractor.policy_net(features)
                # Get mean action (deterministic)
                mean_actions = self.policy.action_net(policy_features)
                return mean_actions
        
        # Use the deterministic wrapper
        policy_net = DeterministicPPOWrapper(model.policy)
        
    else:  # DDPG
        # DDPG is already deterministic, use policy directly
        policy_net = model.policy
    
    # CRITICAL: Set to evaluation mode to disable dropout, batch norm, etc.
    policy_net.eval()  # Set to evaluation mode
    
    # For extra safety, also disable gradients (not needed for inference)
    for param in policy_net.parameters():
        param.requires_grad = False
    
    # Generate output filename if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base_name}.onnx"
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    if not output_path.startswith("models/"):
        output_path = f"models/{output_path}"
    
    try:
        print(f"🚀 Starting ONNX conversion...")
        
        # Convert to ONNX
        torch.onnx.export(
            policy_net,                    # Model to convert
            dummy_input,                   # Example input tensor
            output_path,                   # Output file path
            export_params=True,            # Store trained weights
            opset_version=11,              # ONNX operator set version
            do_constant_folding=True,      # Optimize constant folding
            input_names=['observation'],   # Input tensor name
            output_names=['action'],       # Output tensor name
            dynamic_axes={                 # Variable length axes
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            },
            verbose=True                   # Print conversion details
        )
        
        print(f"✅ ONNX model saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        print("💡 This might be due to unsupported PyTorch operations in the model")
        return None

def validate_onnx_model(onnx_path, original_model, model_type="PPO", config_path="SoccerEnv/field_config.yaml"):
    """
    Validate that ONNX model produces similar outputs to original model
    """
    print(f"\n🔍 Validating ONNX model: {onnx_path}")
    
    try:
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_shape = get_model_input_shape(config_path)
        
        # Create test inputs
        test_inputs = []
        for i in range(5):  # Test with 5 different random inputs
            test_input = np.random.randn(*input_shape).astype(np.float32)
            test_inputs.append(test_input)
        
        validation_passed = True
        max_difference = 0.0
        
        for i, test_input in enumerate(test_inputs):
            print(f"  Test {i+1}/5...\n", end=" ")
            
            # Get original model prediction (deterministic)
            if model_type.upper() == "PPO":
                # For PPO, use deterministic=True to get mean action
                original_action, _ = original_model.predict(test_input, deterministic=True)
            else:
                # For DDPG, it's already deterministic
                original_action, _ = original_model.predict(test_input, deterministic=True)
                
                # Get ONNX model prediction
                onnx_input = test_input.reshape(1, -1)  # Add batch dimension
                onnx_outputs = ort_session.run(['action'], {'observation': onnx_input})
                onnx_action = onnx_outputs[0][0]  # Remove batch dimension
                
                # Calculate difference
                if original_action.shape != onnx_action.shape:
                    print(f"❌ Shape mismatch: {original_action.shape} vs {onnx_action.shape}")
                    validation_passed = False
                    continue
                    
                difference = np.abs(original_action - onnx_action).max()
                max_difference = max(max_difference, difference)
                
                # Tolerance for floating point differences
                tolerance = 1e-4
                if difference > tolerance:
                    print(f"❌ Large difference: {difference:.6f}")
                    validation_passed = False
                else:
                    print(f"✅ Difference: {difference:.6f}")
            
        if validation_passed:
            print(f"🎉 ONNX validation PASSED! Max difference: {max_difference:.6f}")
            return True
        else:
            print(f"⚠️  ONNX validation FAILED! Max difference: {max_difference:.6f}")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def create_onnx_config(onnx_path, input_shape, output_shape):
    """
    Create configuration file for ONNX model deployment
    """
    config = {
        'rl_module': {
            'model': {
                'path': onnx_path,
                'input_shape': list(input_shape),
                'output_shape': list(output_shape),
                'validation_threshold': 0.95
            },
            'runtime': {
                'inference_frequency': 50,  # 50Hz as required
                'max_inference_time': 20,   # 20ms max
                'fallback_enabled': True,
                'performance_monitoring': True
            },
            'preprocessing': {
                'normalise_inputs': True,
                'coordinate_bounds': {
                    'field_x': [-4.5, 4.5],   # RoboCup field dimensions
                    'field_y': [-3.0, 3.0],
                    'velocity': [-2.0, 2.0]
                }
            }
        }
    }
    
    config_path = onnx_path.replace('.onnx', '_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Configuration saved to: {config_path}")
    return config_path

def convert_all_models(config_path="SoccerEnv/field_config.yaml"):
    """
    Convert all trained models to ONNX format
    """
    print("🚀 Converting all trained models to ONNX format...")
    print("="*60)
    
    models_to_convert = [
        ("soccer_rl_ppo_final", "PPO"),
        ("soccer_rl_ddpg_final", "DDPG"),
        ("soccer_rl_ppo_partial", "PPO"),    # In case partial models exist
        ("soccer_rl_ddpg_partial", "DDPG")
    ]
    
    successful_conversions = []
    
    for model_name, model_type in models_to_convert:
        model_path = f"{model_name}.zip"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping {model_name}: File not found")
            continue
        
        print(f"\n{'='*40}")
        print(f"Converting {model_type}: {model_name}")
        print(f"{'='*40}")
        
        # Convert to ONNX
        onnx_path = convert_sb3_to_onnx(
            model_path=model_path,
            model_type=model_type,
            config_path=config_path
        )
        
        if onnx_path:
            # Load original model for validation
            try:
                if model_type == "PPO":
                    original_model = PPO.load(model_path)
                else:
                    original_model = DDPG.load(model_path)
                
                # Validate conversion
                is_valid = validate_onnx_model(onnx_path, original_model, model_type, config_path)
                
                if is_valid:
                    # Create deployment config
                    input_shape = get_model_input_shape(config_path)
                    
                    # Get output shape from model
                    env = SoccerEnv(difficulty="easy", config_path=config_path)
                    output_shape = env.action_space.shape
                    env.close()
                    
                    config_file = create_onnx_config(onnx_path, input_shape, output_shape)
                    
                    successful_conversions.append({
                        'model_type': model_type,
                        'onnx_path': onnx_path,
                        'config_path': config_file,
                        'validated': True
                    })
                    
                    print(f"✅ {model_type} conversion completed successfully!")
                else:
                    print(f"⚠️  {model_type} converted but validation failed")
                    
            except Exception as e:
                print(f"❌ Error during validation: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎯 CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    if successful_conversions:
        print(f"✅ Successfully converted {len(successful_conversions)} models:")
        for conversion in successful_conversions:
            print(f"  • {conversion['model_type']}: {conversion['onnx_path']}")
            print(f"    Config: {conversion['config_path']}")
        
        print(f"\n📋 Next steps for deployment:")
        print(f"1. Copy .onnx files to your robot's models/ directory")
        print(f"2. Copy config files for runtime parameters")
        print(f"3. Integrate with NUbots C++ inference module")
        print(f"4. Test inference at 50Hz frequency")
        
    else:
        print("❌ No models were successfully converted")
        print("💡 Make sure you have trained models available:")
        print("   - soccer_rl_ppo_final.zip")
        print("   - soccer_rl_ddpg_final.zip")
    
    return successful_conversions

def test_onnx_inference_speed(onnx_path, config_path="SoccerEnv/field_config.yaml"):
    """
    Test ONNX model inference speed to ensure it meets 50Hz requirement
    """
    print(f"\n⚡ Testing inference speed: {onnx_path}")
    
    try:
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_shape = get_model_input_shape(config_path)
        
        # Warm up (important for accurate timing)
        print("🔥 Warming up...")
        for _ in range(50):  # More warm-up iterations
            test_input = np.random.randn(1, *input_shape).astype(np.float32)
            _ = ort_session.run(None, {'observation': test_input})
        
        # Time multiple inferences
        import time
        num_tests = 1000
        
        print(f"📊 Running {num_tests} inference tests...")
        start_time = time.time()
        for _ in range(num_tests):
            test_input = np.random.randn(1, *input_shape).astype(np.float32)
            _ = ort_session.run(None, {'observation': test_input})
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_tests) * 1000
        max_frequency = 1000 / avg_time_ms
        
        print(f"📊 Inference performance:")
        print(f"  Average time per inference: {avg_time_ms} ms")
        print(f"  Maximum frequency: {max_frequency} Hz")
        print(f"  Total time for {num_tests} inferences: {total_time}s")
        
        if max_frequency >= 50:
            print(f"✅ MEETS 50Hz requirement! (Can run at {max_frequency} Hz)")
            meets_requirement = True
        else:
            print(f"⚠️  Does not meet 50Hz requirement (only {max_frequency} Hz)")
            meets_requirement = False
        
        return {
            'avg_time_ms': avg_time_ms,
            'max_frequency': max_frequency,
            'meets_50hz': meets_requirement,
            'total_time': total_time,
            'num_tests': num_tests
        }
        
    except Exception as e:
        print(f"❌ Speed test error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🤖 Soccer RL ONNX Model Converter")
    print("="*50)
    
    # Check dependencies
    try:
        import onnx
        import onnxruntime
        print("✅ ONNX dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install onnx onnxruntime")
        exit(1)
    
    print("\nConversion Options:")
    print("1. Convert all models")
    print("2. Convert specific model")
    print("3. Test inference speed")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Convert all models
        conversions = convert_all_models()
        
        # Test speed for all converted models
        for conversion in conversions:
            test_onnx_inference_speed(conversion['onnx_path'])
    
    elif choice == "2":
        # Convert specific model
        model_path = input("Enter model path (e.g., soccer_rl_ppo_final.zip): ").strip()
        model_type = input("Enter model type (PPO/DDPG): ").strip().upper()
        
        if model_type not in ["PPO", "DDPG"]:
            print("❌ Invalid model type. Use PPO or DDPG")
            exit(1)
        
        onnx_path = convert_sb3_to_onnx(model_path, model_type)
        
        if onnx_path:
            # Validate
            if model_type == "PPO":
                original_model = PPO.load(model_path)
            else:
                original_model = DDPG.load(model_path)
            
            validate_onnx_model(onnx_path, original_model, model_type)
            test_onnx_inference_speed(onnx_path)
    
    elif choice == "3":
        # Test existing ONNX model speed
        onnx_path = input("Enter ONNX model path: ").strip()
        test_onnx_inference_speed(onnx_path)
    
    else:
        print("Invalid choice")