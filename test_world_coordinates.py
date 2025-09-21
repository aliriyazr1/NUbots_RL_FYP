#!/usr/bin/env python3
"""
Test script to verify world coordinate dynamics work correctly
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments.soccerenv import SoccerEnv

def test_world_coordinate_dynamics():
    """Test the world coordinate implementation"""
    print("🧪 Testing World Coordinate Dynamics")
    print("=" * 50)
    
    env = SoccerEnv(render_mode=None, difficulty="easy")
    print("✅ Environment created successfully")
    
    # Test cases with expected behavior
    test_cases = [
        # [action, description, validation_function]
        ([1.0, 0.0, 0.0], "Move +X (right/toward goal)", lambda dx, dy, da: dx > 5),
        ([-1.0, 0.0, 0.0], "Move -X (left/away from goal)", lambda dx, dy, da: dx < -5),
        ([0.0, 1.0, 0.0], "Move +Y (down)", lambda dx, dy, da: dy > 5),
        ([0.0, -1.0, 0.0], "Move -Y (up)", lambda dx, dy, da: dy < -5),
        ([0.0, 0.0, 1.0], "Rotate +θ (clockwise)", lambda dx, dy, da: da > 0.1),
        ([0.0, 0.0, -1.0], "Rotate -θ (counter-clockwise)", lambda dx, dy, da: da < -0.1),
        ([1.0, 1.0, 0.0], "Move +X +Y (diagonal)", lambda dx, dy, da: dx > 3 and dy > 3),
        ([0.5, 0.0, 0.5], "Move +X and rotate", lambda dx, dy, da: dx > 2 and da > 0.05),
    ]
    
    all_passed = True
    
    for i, (action, description, validation) in enumerate(test_cases):
        print(f"\nTest {i+1}: {description}")
        print(f"Action: {action}")
        
        # Reset and capture initial state
        obs, _ = env.reset()
        initial_pos = env.robot_pos.copy()
        initial_angle = env.robot_angle
        
        # Apply action for multiple steps
        for _ in range(15):  # More steps for clearer results
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Calculate changes
        final_pos = env.robot_pos
        final_angle = env.robot_angle
        
        dx = final_pos[0] - initial_pos[0]
        dy = final_pos[1] - initial_pos[1]
        da = final_angle - initial_angle
        
        # Normalize angle difference
        da = np.arctan2(np.sin(da), np.cos(da))
        
        print(f"Result: Δx={dx:.1f}, Δy={dy:.1f}, Δθ={np.degrees(da):.1f}°")
        
        # Validate
        passed = validation(dx, dy, da)
        if passed:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🎯 World coordinate dynamics are working correctly!")
        print("\nKey improvements:")
        print("- Direct world coordinate control (x_velocity, y_velocity, angular_velocity)")
        print("- No complex trigonometry for body-frame transformations")
        print("- Simpler and more intuitive robot control")
        print("- Proper 2D vehicle model: x[k+1] = x[k] + ẋ*dt")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    return all_passed

if __name__ == "__main__":
    test_world_coordinate_dynamics()