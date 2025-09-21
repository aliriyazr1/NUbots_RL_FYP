#!/usr/bin/env python3
"""
COORDINATE SYSTEM VERIFICATION TEST
Run this to verify the implementation is 100% correct
"""

import numpy as np
import time
import sys
import os
import pygame

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.soccerenv import SoccerEnv

def test_coordinate_system():
    """Test each action type systematically"""
    
    # Use correct config path relative to project root
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'field_config.yaml')
    env = SoccerEnv(render_mode="human", difficulty="easy", config_path=config_path)
    
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
    """Quick test you can control manually with detailed angle debugging"""
    # Use correct config path relative to project root
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'field_config.yaml')
    env = SoccerEnv(render_mode="human", difficulty="easy", config_path=config_path)
    
    print("🎮 MANUAL TEST MODE WITH ANGLE DEBUGGING")
    print("Commands:")
    print("  W/↑ = forward (+X)     S/↓ = backward (-X)")
    print("  D/→ = right (+Y)       A/← = left (-Y)") 
    print("  E = rotate CW          Q = rotate CCW")
    print("  R = reset robot angle to specific test angles")
    print("  ESC/X = exit")
    print("\n💡 Click on the game window and use keyboard controls!")
    print("\n📐 ANGLE CONVENTION VERIFICATION:")
    print("  θ = 0°      → Robot faces RIGHT (toward goal)")
    print("  θ = 90°     → Robot faces DOWN")  
    print("  θ = 180°    → Robot faces LEFT (away from goal)")
    print("  θ = 270°    → Robot faces UP")
    
    obs, _ = env.reset()
    
    # Initialize pygame for keyboard input
    pygame.init()
    clock = pygame.time.Clock()
    
    print("✨ Pygame controls enabled!")
    print(f"🤖 Robot starting position: ({env.robot_pos[0]:.1f}, {env.robot_pos[1]:.1f})")
    print(f"🧭 Robot starting angle: {np.degrees(env.robot_angle):.1f}° ({env.robot_angle:.3f} radians)")
    
    # Test angle counter for R key
    test_angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # 0°, 90°, 180°, 270°
    test_angle_names = ["0° (RIGHT)", "90° (DOWN)", "180° (LEFT)", "270° (UP)"]
    current_test_angle = 0
    
    running = True
    
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_x:
                    running = False
                elif event.key == pygame.K_r:
                    # Cycle through test angles
                    env.robot_angle = test_angles[current_test_angle]
                    print(f"\n🎯 TEST ANGLE SET: {test_angle_names[current_test_angle]}")
                    print(f"   Radians: {env.robot_angle:.3f}")
                    print(f"   Degrees: {np.degrees(env.robot_angle):.1f}°")
                    print(f"   Expected direction: {test_angle_names[current_test_angle].split()[1]}")
                    current_test_angle = (current_test_angle + 1) % len(test_angles)
        
        # Get current key states
        keys = pygame.key.get_pressed()
        action = [0.0, 0.0, 0.0]
        
        # Movement controls (world coordinates)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action[0] = 1.0  # +X (toward goal/right)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action[0] = -1.0  # -X (away from goal/left)
            
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[1] = 1.0  # +Y (down on screen)
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[1] = -1.0  # -Y (up on screen)
            
        if keys[pygame.K_e]:
            action[2] = 1.0  # Rotate clockwise (+angular velocity)
        elif keys[pygame.K_q]:
            action[2] = -1.0  # Rotate counter-clockwise (-angular velocity)
        
        # Apply action and render
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # Show current state if any movement
        if any(abs(a) > 0 for a in action):
            # Convert angle to 0-360 range for easier reading
            angle_deg = np.degrees(env.robot_angle) % 360
            
            # Determine direction based on angle
            if -22.5 <= angle_deg <= 22.5 or angle_deg >= 337.5:
                direction = "RIGHT (→)"
            elif 22.5 < angle_deg <= 67.5:
                direction = "DOWN-RIGHT (↘)"
            elif 67.5 < angle_deg <= 112.5:
                direction = "DOWN (↓)"
            elif 112.5 < angle_deg <= 157.5:
                direction = "DOWN-LEFT (↙)"
            elif 157.5 < angle_deg <= 202.5:
                direction = "LEFT (←)"
            elif 202.5 < angle_deg <= 247.5:
                direction = "UP-LEFT (↖)"
            elif 247.5 < angle_deg <= 292.5:
                direction = "UP (↑)"
            else:  # 292.5 < angle_deg < 337.5
                direction = "UP-RIGHT (↗)"
            
            print(f"🎯 Action: [{action[0]:+.1f}, {action[1]:+.1f}, {action[2]:+.1f}] | "
                  f"Pos: ({env.robot_pos[0]:.1f}, {env.robot_pos[1]:.1f}) | "
                  f"Angle: {angle_deg:.1f}° ({env.robot_angle:.3f}rad) | "
                  f"Facing: {direction}")
        
        if terminated or truncated:
            print("🔄 Episode ended, resetting...")
            obs, _ = env.reset()
        
        clock.tick(60)  # 60 FPS
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Automatic verification test")
    print("2. Manual keyboard test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_manual_test()
    else:
        test_coordinate_system()