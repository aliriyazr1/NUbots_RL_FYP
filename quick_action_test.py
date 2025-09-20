#!/usr/bin/env python3
"""
COORDINATE SYSTEM VERIFICATION TEST
Run this to verify the implementation is 100% correct
"""

import numpy as np
import time
from SoccerEnv.soccerenv import SoccerEnv

def test_coordinate_system():
    """Test each action type systematically"""
    
    env = SoccerEnv(render_mode="human", difficulty="easy")
    
    # Test cases: [action, expected_behavior, validation_function]
    test_cases = [
        # Basic forward/backward tests
        ([1.0, 0.0, 0.0], "Forward (toward goal/right)", lambda pos_change, angle_change: pos_change[0] > 5),
        ([-1.0, 0.0, 0.0], "Backward (away from goal/left)", lambda pos_change, angle_change: pos_change[0] < -5),
        
        # Basic strafe tests
        ([0.0, 1.0, 0.0], "Strafe Right (down when facing right)", lambda pos_change, angle_change: pos_change[1] > 5),
        ([0.0, -1.0, 0.0], "Strafe Left (up when facing right)", lambda pos_change, angle_change: pos_change[1] < -5),
        
        # Rotation tests
        ([0.0, 0.0, 1.0], "Rotate Clockwise", lambda pos_change, angle_change: angle_change < -0.1),
        ([0.0, 0.0, -1.0], "Rotate Counter-clockwise", lambda pos_change, angle_change: angle_change > 0.1),
        
        # Combined tests
        ([1.0, 1.0, 0.0], "Forward + Right Strafe", lambda pos_change, angle_change: pos_change[0] > 3 and pos_change[1] > 3),
        ([0.5, 0.0, 0.5], "Forward + Rotate", lambda pos_change, angle_change: pos_change[0] > 2 and angle_change < -0.05),
    ]
    
    print("🧪 COORDINATE SYSTEM VERIFICATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    for i, (action, description, validation_func) in enumerate(test_cases):
        print(f"\n🔍 TEST {i+1}: {description}")
        print(f"   Action: {action}")
        
        # Reset and get initial state
        obs, _ = env.reset()
        initial_pos = env.robot_pos.copy()
        initial_angle = env.robot_angle
        
        print(f"   Initial: pos=({initial_pos[0]:.1f}, {initial_pos[1]:.1f}), angle={np.degrees(initial_angle):.1f}°")
        
        # Apply action for 30 steps
        for step in range(30):
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)  # Visual delay
            
            if terminated or truncated:
                break
        
        # Calculate changes
        final_pos = env.robot_pos
        final_angle = env.robot_angle
        
        pos_change = final_pos - initial_pos
        angle_change = final_angle - initial_angle
        
        # Normalize angle change to [-π, π]
        angle_change = np.arctan2(np.sin(angle_change), np.cos(angle_change))
        
        print(f"   Final: pos=({final_pos[0]:.1f}, {final_pos[1]:.1f}), angle={np.degrees(final_angle):.1f}°")
        print(f"   Change: Δpos=({pos_change[0]:.1f}, {pos_change[1]:.1f}), Δangle={np.degrees(angle_change):.1f}°")
        
        # Validate result
        passed = validation_func(pos_change, angle_change)
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
            all_passed = False
        
        time.sleep(0.5)  # Pause between tests
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Coordinate system is correct.")
    else:
        print("❌ SOME TESTS FAILED! Check the implementation.")
    
    env.close()
    return all_passed

def quick_manual_test():
    """Quick test you can control manually"""
    env = SoccerEnv(render_mode="human", difficulty="easy")
    
    print("🎮 MANUAL TEST MODE")
    print("Commands:")
    print("  w = forward    s = backward")
    print("  d = right      a = left") 
    print("  e = rotate CW  q = rotate CCW")
    print("  x = exit")
    
    obs, _ = env.reset()
    
    try:
        import keyboard
        print("✨ Keyboard controls enabled!")
        
        while True:
            action = [0.0, 0.0, 0.0]
            
            if keyboard.is_pressed('w'):
                action[0] = 1.0  # Forward
            elif keyboard.is_pressed('s'):
                action[0] = -1.0  # Backward
                
            if keyboard.is_pressed('d'):
                action[1] = 1.0  # Right
            elif keyboard.is_pressed('a'):
                action[1] = -1.0  # Left
                
            if keyboard.is_pressed('e'):
                action[2] = 1.0  # Clockwise
            elif keyboard.is_pressed('q'):
                action[2] = -1.0  # Counter-clockwise
                
            if keyboard.is_pressed('x'):
                break
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                obs, _ = env.reset()
            
            time.sleep(0.02)
                
    except ImportError:
        print("⚠️  Keyboard library not available, using automatic test")
        return test_coordinate_system()
    
    env.close()

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Automatic verification test")
    print("2. Manual keyboard test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_manual_test()
    else:
        test_coordinate_system()