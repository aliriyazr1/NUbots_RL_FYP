#!/usr/bin/env python3
"""
World Coordinate Validation Test
================================

This script validates that the world coordinate system is working correctly
in the updated soccerenv.py by testing:

1. Direct movement in world X and Y directions
2. Independence of movement from robot orientation
3. Angular velocity behavior
4. Consistency with 2D vehicle dynamics

Author: Claude AI Assistant
Date: September 20, 2025
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environments.soccerenv import SoccerEnv

def test_world_coordinate_independence():
    """
    Test that movement is independent of robot orientation.
    
    In world coordinates, action [1, 0, 0] should always move robot in +X direction
    regardless of the robot's current angle.
    """
    print("🧪 Testing World Coordinate Independence...")
    
    env = SoccerEnv(render_mode=None, testing_mode=True, config_path="configs/field_config.yaml")
    
    # Test different robot orientations
    test_angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    
    for angle in test_angles:
        print(f"\n📐 Testing at robot angle: {np.degrees(angle):.1f}°")
        
        # Reset and set specific robot angle
        env.reset()
        env.robot_pos = np.array([200.0, 200.0])  # Center position
        env.robot_angle = angle
        initial_pos = env.robot_pos.copy()
        
        # Apply pure +X movement in world coordinates
        action = np.array([1.0, 0.0, 0.0])  # [x_vel, y_vel, angular_vel]
        
        # Take several steps to see clear movement
        for _ in range(5):
            env._apply_robot_action(action)
        
        final_pos = env.robot_pos
        movement = final_pos - initial_pos
        
        print(f"  Initial pos: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
        print(f"  Final pos:   ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
        print(f"  Movement:    ({movement[0]:.1f}, {movement[1]:.1f})")
        
        # Verify movement is purely in +X direction
        assert movement[0] > 0, f"Should move in +X direction, got {movement[0]}"
        assert abs(movement[1]) < 1e-6, f"Should not move in Y direction, got {movement[1]}"
        
        print("  ✅ PASSED: Movement is purely in +X direction regardless of robot angle")

def test_direct_world_movements():
    """
    Test that each action component moves the robot correctly in world coordinates.
    """
    print("\n🧪 Testing Direct World Movements...")
    
    env = SoccerEnv(render_mode=None, testing_mode=True, config_path="configs/field_config.yaml")
    env.reset()
    
    # Test pure X movement
    print("\n📍 Testing +X movement...")
    env.robot_pos = np.array([200.0, 200.0])
    env.robot_angle = np.pi/4  # 45 degrees (should not affect movement)
    initial_pos = env.robot_pos.copy()
    
    action = np.array([1.0, 0.0, 0.0])  # Pure +X
    for _ in range(3):
        env._apply_robot_action(action)
    
    movement = env.robot_pos - initial_pos
    print(f"  Movement: ({movement[0]:.1f}, {movement[1]:.1f})")
    assert movement[0] > 0 and abs(movement[1]) < 1e-6, "Failed +X movement test"
    print("  ✅ PASSED: +X movement working")
    
    # Test pure Y movement
    print("\n📍 Testing +Y movement...")
    env.robot_pos = np.array([200.0, 200.0])
    env.robot_angle = np.pi/3  # 60 degrees (should not affect movement)
    initial_pos = env.robot_pos.copy()
    
    action = np.array([0.0, 1.0, 0.0])  # Pure +Y
    for _ in range(3):
        env._apply_robot_action(action)
    
    movement = env.robot_pos - initial_pos
    print(f"  Movement: ({movement[0]:.1f}, {movement[1]:.1f})")
    assert abs(movement[0]) < 1e-6 and movement[1] > 0, "Failed +Y movement test"
    print("  ✅ PASSED: +Y movement working")
    
    # Test pure rotation
    print("\n📍 Testing pure rotation...")
    env.robot_pos = np.array([200.0, 200.0])
    initial_pos = env.robot_pos.copy()
    initial_angle = env.robot_angle
    
    action = np.array([0.0, 0.0, 1.0])  # Pure rotation
    for _ in range(5):
        env._apply_robot_action(action)
    
    movement = env.robot_pos - initial_pos
    angle_change = env.robot_angle - initial_angle
    print(f"  Position change: ({movement[0]:.1f}, {movement[1]:.1f})")
    print(f"  Angle change: {np.degrees(angle_change):.1f}°")
    assert abs(movement[0]) < 1e-6 and abs(movement[1]) < 1e-6, "Position should not change during pure rotation"
    assert abs(angle_change) > 0, "Angle should change during rotation"
    print("  ✅ PASSED: Pure rotation working")

def test_discrete_dynamics_consistency():
    """
    Test that the discrete dynamics follow the correct mathematical model:
    x[k+1] = x[k] + ẋ * dt
    y[k+1] = y[k] + ẏ * dt  
    θ[k+1] = θ[k] + θ̇ * dt
    """
    print("\n🧪 Testing Discrete Dynamics Consistency...")
    
    env = SoccerEnv(render_mode=None, testing_mode=True, config_path="configs/field_config.yaml")
    env.reset()
    
    # Set known initial state
    initial_x, initial_y = 250.0, 150.0
    initial_angle = np.pi/6  # 30 degrees
    env.robot_pos = np.array([initial_x, initial_y])
    env.robot_angle = initial_angle
    
    # Apply known action
    action = np.array([0.5, -0.3, 0.2])  # Mixed movement
    
    # Calculate expected velocities (from the environment's internal calculations)
    max_linear_speed = env.robot_speed
    max_angular_speed = env.robot_rotation_speed
    
    expected_vx = action[0] * max_linear_speed
    expected_vy = action[1] * max_linear_speed
    expected_angular_vel = action[2] * max_angular_speed
    
    print(f"📊 Expected velocities:")
    print(f"  vx: {expected_vx:.3f} pixels/frame")
    print(f"  vy: {expected_vy:.3f} pixels/frame") 
    print(f"  ω:  {expected_angular_vel:.3f} rad/frame")
    
    # Apply action once
    env._apply_robot_action(action)
    
    # Check results
    actual_x = env.robot_pos[0]
    actual_y = env.robot_pos[1]
    actual_angle = env.robot_angle
    
    # Calculate expected results (dt=1 since velocities are in pixels/frame)
    expected_x = initial_x + expected_vx
    expected_y = initial_y + expected_vy
    expected_angle = (initial_angle + expected_angular_vel) % (2 * np.pi)
    
    print(f"\n📊 Position comparison:")
    print(f"  Expected: ({expected_x:.3f}, {expected_y:.3f})")
    print(f"  Actual:   ({actual_x:.3f}, {actual_y:.3f})")
    print(f"  Diff:     ({abs(actual_x - expected_x):.3f}, {abs(actual_y - expected_y):.3f})")
    
    print(f"\n📊 Angle comparison:")
    print(f"  Expected: {np.degrees(expected_angle):.3f}°")
    print(f"  Actual:   {np.degrees(actual_angle):.3f}°")
    print(f"  Diff:     {np.degrees(abs(actual_angle - expected_angle)):.3f}°")
    
    # Assertions with small tolerance for floating point errors
    tolerance = 1e-10
    assert abs(actual_x - expected_x) < tolerance, f"X position mismatch: {abs(actual_x - expected_x)}"
    assert abs(actual_y - expected_y) < tolerance, f"Y position mismatch: {abs(actual_y - expected_y)}"
    assert abs(actual_angle - expected_angle) < tolerance, f"Angle mismatch: {abs(actual_angle - expected_angle)}"
    
    print("  ✅ PASSED: Discrete dynamics are mathematically correct")

def test_boundary_behavior():
    """
    Test that the robot behaves correctly at field boundaries.
    """
    print("\n🧪 Testing Boundary Behavior...")
    
    env = SoccerEnv(render_mode=None, testing_mode=True, config_path="configs/field_config.yaml")
    env.reset()
    
    # Test right boundary
    env.robot_pos = np.array([env.field_width - env.robot_radius - 1, 200.0])
    initial_pos = env.robot_pos.copy()
    
    action = np.array([1.0, 0.0, 0.0])  # Try to move further right
    env._apply_robot_action(action)
    
    final_pos = env.robot_pos
    print(f"📍 Right boundary test:")
    print(f"  Initial: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
    print(f"  Final:   ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
    print(f"  Field width: {env.field_width}")
    
    # Robot should be clamped to boundary
    assert final_pos[0] <= env.field_width - env.robot_radius, "Robot exceeded right boundary"
    print("  ✅ PASSED: Right boundary clamping works")
    
    # Test left boundary
    env.robot_pos = np.array([env.robot_radius + 1, 200.0])
    initial_pos = env.robot_pos.copy()
    
    action = np.array([-1.0, 0.0, 0.0])  # Try to move further left
    env._apply_robot_action(action)
    
    final_pos = env.robot_pos
    print(f"\n📍 Left boundary test:")
    print(f"  Initial: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
    print(f"  Final:   ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
    
    # Robot should be clamped to boundary
    assert final_pos[0] >= env.robot_radius, "Robot exceeded left boundary"
    print("  ✅ PASSED: Left boundary clamping works")

def test_velocity_storage():
    """
    Test that robot velocities are properly stored for observations.
    """
    print("\n🧪 Testing Velocity Storage...")
    
    env = SoccerEnv(render_mode=None, testing_mode=True, config_path="configs/field_config.yaml")
    env.reset()
    
    # Apply action and check stored velocity
    action = np.array([0.6, -0.4, 0.1])
    env._apply_robot_action(action)
    
    expected_vx = action[0] * env.robot_speed
    expected_vy = action[1] * env.robot_speed
    
    actual_vel = env.robot_vel
    
    print(f"📊 Velocity storage test:")
    print(f"  Expected: ({expected_vx:.3f}, {expected_vy:.3f})")
    print(f"  Stored:   ({actual_vel[0]:.3f}, {actual_vel[1]:.3f})")
    
    assert abs(actual_vel[0] - expected_vx) < 1e-10, "X velocity not stored correctly"
    assert abs(actual_vel[1] - expected_vy) < 1e-10, "Y velocity not stored correctly"
    
    print("  ✅ PASSED: Velocities stored correctly")

def run_all_tests():
    """Run all validation tests."""
    print("🚀 Starting World Coordinate Validation Tests")
    print("=" * 60)
    
    try:
        test_world_coordinate_independence()
        test_direct_world_movements()
        test_discrete_dynamics_consistency()
        test_boundary_behavior()
        test_velocity_storage()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! World coordinate system is working correctly.")
        print("\n✨ Benefits of the new system:")
        print("  • Simple, intuitive movement in world X/Y directions")
        print("  • No complex trigonometric calculations")
        print("  • Robot orientation doesn't affect movement direction")
        print("  • Mathematically correct discrete vehicle dynamics")
        print("  • Easier for RL agents to learn directional movement")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎯 VALIDATION COMPLETE: The world coordinate system is correctly implemented!")
    else:
        print("\n🔧 VALIDATION FAILED: Please check the implementation.")
        
    sys.exit(0 if success else 1)