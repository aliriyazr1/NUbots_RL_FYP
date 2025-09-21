#!/usr/bin/env python3
"""
Simple World Coordinate Validation Test
=======================================

Basic validation that world coordinates are working correctly.

Author: Claude AI Assistant  
Date: September 20, 2025
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_world_coordinates():
    """Simple test that world coordinates work as expected."""
    print("🧪 Testing World Coordinate Implementation...")
    
    try:
        from src.environments.soccerenv import SoccerEnv
        
        # Create environment with default config
        env = SoccerEnv(render_mode=None, config_path="configs/field_config.yaml")
        env.reset()
        
        print("✅ Environment created successfully")
        
        # Test 1: Pure X movement
        print("\n📍 Test 1: Pure +X movement")
        initial_pos = np.array([200.0, 200.0])
        env.robot_pos = initial_pos.copy()
        env.robot_angle = np.pi/4  # 45 degrees - shouldn't affect movement
        
        # Pure +X movement action
        action = np.array([1.0, 0.0, 0.0])
        
        # Apply action multiple times
        for _ in range(3):
            env._apply_robot_action(action)
        
        final_pos = env.robot_pos
        movement = final_pos - initial_pos
        
        print(f"  Initial position: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
        print(f"  Final position:   ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
        print(f"  Movement vector:  ({movement[0]:.1f}, {movement[1]:.1f})")
        
        # Check that movement is purely in +X direction
        if movement[0] > 0 and abs(movement[1]) < 1e-6:
            print("  ✅ PASSED: Movement is purely in +X direction")
        else:
            print(f"  ❌ FAILED: Expected +X movement, got ({movement[0]:.3f}, {movement[1]:.3f})")
            return False
        
        # Test 2: Pure Y movement
        print("\n📍 Test 2: Pure +Y movement")
        env.robot_pos = np.array([200.0, 200.0])
        env.robot_angle = np.pi/3  # 60 degrees - shouldn't affect movement
        initial_pos = env.robot_pos.copy()
        
        # Pure +Y movement action
        action = np.array([0.0, 1.0, 0.0])
        
        for _ in range(3):
            env._apply_robot_action(action)
        
        final_pos = env.robot_pos
        movement = final_pos - initial_pos
        
        print(f"  Initial position: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
        print(f"  Final position:   ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
        print(f"  Movement vector:  ({movement[0]:.1f}, {movement[1]:.1f})")
        
        # Check that movement is purely in +Y direction
        if abs(movement[0]) < 1e-6 and movement[1] > 0:
            print("  ✅ PASSED: Movement is purely in +Y direction")
        else:
            print(f"  ❌ FAILED: Expected +Y movement, got ({movement[0]:.3f}, {movement[1]:.3f})")
            return False
        
        # Test 3: Angular independence
        print("\n📍 Test 3: Movement independence from robot angle")
        
        test_angles = [0, np.pi/4, np.pi/2, np.pi]
        all_movements = []
        
        for angle in test_angles:
            env.robot_pos = np.array([200.0, 200.0])
            env.robot_angle = angle
            initial_pos = env.robot_pos.copy()
            
            # Same action regardless of angle
            action = np.array([0.5, 0.0, 0.0])  # Half-speed +X
            env._apply_robot_action(action)
            
            movement = env.robot_pos - initial_pos
            all_movements.append(movement)
            print(f"  Angle {np.degrees(angle):3.0f}°: movement ({movement[0]:.3f}, {movement[1]:.3f})")
        
        # Check that all movements are identical (within tolerance)
        reference_movement = all_movements[0]
        movements_consistent = True
        
        for i, movement in enumerate(all_movements[1:], 1):
            diff = np.linalg.norm(movement - reference_movement)
            if diff > 1e-10:
                print(f"  ❌ Movement at angle {i} differs from reference by {diff}")
                movements_consistent = False
        
        if movements_consistent:
            print("  ✅ PASSED: Movement is independent of robot angle")
        else:
            print("  ❌ FAILED: Movement depends on robot angle")
            return False
        
        # Test 4: Discrete dynamics verification
        print("\n📍 Test 4: Discrete dynamics verification")
        env.robot_pos = np.array([250.0, 150.0])
        env.robot_angle = np.pi/6
        initial_pos = env.robot_pos.copy()
        initial_angle = env.robot_angle
        
        action = np.array([0.3, -0.2, 0.1])
        
        # Calculate expected movement based on environment parameters
        max_linear_speed = env.robot_speed
        max_angular_speed = env.robot_rotation_speed
        
        expected_dx = action[0] * max_linear_speed
        expected_dy = action[1] * max_linear_speed
        expected_dtheta = action[2] * max_angular_speed
        
        env._apply_robot_action(action)
        
        actual_dx = env.robot_pos[0] - initial_pos[0]
        actual_dy = env.robot_pos[1] - initial_pos[1]
        actual_dtheta = env.robot_angle - initial_angle
        
        print(f"  Expected change: dx={expected_dx:.3f}, dy={expected_dy:.3f}, dθ={expected_dtheta:.3f}")
        print(f"  Actual change:   dx={actual_dx:.3f}, dy={actual_dy:.3f}, dθ={actual_dtheta:.3f}")
        
        # Check if changes match expectations (within tolerance)
        tolerance = 1e-10
        dx_ok = abs(actual_dx - expected_dx) < tolerance
        dy_ok = abs(actual_dy - expected_dy) < tolerance
        dtheta_ok = abs(actual_dtheta - expected_dtheta) < tolerance
        
        if dx_ok and dy_ok and dtheta_ok:
            print("  ✅ PASSED: Discrete dynamics are mathematically correct")
        else:
            print("  ❌ FAILED: Discrete dynamics don't match expected values")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✨ World coordinate benefits:")
        print("  • Robot moves in intuitive world X/Y directions")
        print("  • Movement independent of robot orientation")
        print("  • Simple discrete dynamics: x[k+1] = x[k] + ẋ*dt")
        print("  • No complex trigonometric calculations")
        print("  • Easier for RL agents to learn directional control")
        
        return True
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Simple World Coordinate Validation")
    print("=" * 50)
    
    success = test_simple_world_coordinates()
    
    if success:
        print("\n🎯 VALIDATION SUCCESSFUL!")
        print("The world coordinate system is working correctly.")
        
        print("\n📋 How to further validate:")
        print("1. Run the environment with render_mode='human'")
        print("2. Apply action [1,0,0] and watch robot move right")
        print("3. Apply action [0,1,0] and watch robot move down")  
        print("4. Apply action [0,0,1] and watch robot rotate")
        print("5. Verify movements don't depend on robot orientation")
        
    else:
        print("\n❌ VALIDATION FAILED!")
        print("Please check the world coordinate implementation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())