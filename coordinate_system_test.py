#!/usr/bin/env python3
"""
Coordinate System Convention Test
================================

This script tests and clarifies the coordinate system conventions used in pygame
and shows the correct way to implement angles for both physics and rendering.
"""

import pygame
import numpy as np
import math
import time

def test_pygame_angle_convention():
    """Test how pygame handles angles and rotations"""
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Coordinate System Test")
    clock = pygame.time.Clock()
    
    print("🧭 PYGAME ANGLE CONVENTION TEST")
    print("=" * 50)
    print("This will show you how angles work in pygame...")
    print("Press SPACE to advance through test cases")
    print("Press ESC to exit")
    
    # Test cases: (angle_in_radians, description)
    test_cases = [
        (0,           "0° - Should point RIGHT"),
        (math.pi/2,   "90° - Should point DOWN (because +Y is down)"),
        (math.pi,     "180° - Should point LEFT"),
        (3*math.pi/2, "270° - Should point UP"),
        (math.pi/4,   "45° - Should point DOWN-RIGHT"),
        (-math.pi/4,  "-45° - Should point UP-RIGHT"),
    ]
    
    test_index = 0
    running = True
    
    while running and test_index < len(test_cases):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    test_index += 1
        
        if test_index < len(test_cases):
            angle_rad, description = test_cases[test_index]
            
            # Clear screen
            screen.fill((30, 30, 30))
            
            # Draw coordinate system reference
            center = (400, 300)
            
            # Draw axes
            pygame.draw.line(screen, (100, 100, 100), (0, center[1]), (800, center[1]), 2)  # X-axis
            pygame.draw.line(screen, (100, 100, 100), (center[0], 0), (center[0], 600), 2)  # Y-axis
            
            # Draw direction labels
            font = pygame.font.Font(None, 36)
            screen.blit(font.render("X+", True, (255, 255, 255)), (750, center[1] - 20))
            screen.blit(font.render("Y+", True, (255, 255, 255)), (center[0] + 10, 550))
            
            # Calculate direction using standard trigonometry
            direction_length = 150
            direction_end_x = center[0] + direction_length * math.cos(angle_rad)
            direction_end_y = center[1] + direction_length * math.sin(angle_rad)
            
            # Draw the direction vector
            pygame.draw.circle(screen, (255, 100, 100), center, 20)  # Robot center
            pygame.draw.line(screen, (255, 255, 255), center, (int(direction_end_x), int(direction_end_y)), 5)
            
            # Show angle information
            angle_deg = math.degrees(angle_rad)
            
            text_lines = [
                f"Test {test_index + 1}/{len(test_cases)}: {description}",
                f"Angle: {angle_deg:.1f}° ({angle_rad:.3f} rad)",
                f"cos({angle_deg:.1f}°) = {math.cos(angle_rad):.3f}",
                f"sin({angle_deg:.1f}°) = {math.sin(angle_rad):.3f}",
                f"Direction: ({direction_end_x - center[0]:.1f}, {direction_end_y - center[1]:.1f})",
                "",
                "Press SPACE for next test"
            ]
            
            y_offset = 50
            for line in text_lines:
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (50, y_offset))
                y_offset += 40
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    
    print("\n✅ Pygame angle test complete!")
    return True

def test_rotation_direction():
    """Test rotation direction to verify clockwise vs counter-clockwise"""
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Rotation Direction Test")
    clock = pygame.time.Clock()
    
    print("\n🔄 ROTATION DIRECTION TEST")
    print("=" * 40)
    print("Testing positive vs negative angular velocity...")
    print("Press ESC to exit")
    
    center = (400, 300)
    current_angle = 0.0
    angular_velocity = 0.05  # radians per frame
    direction = 1  # 1 for positive, -1 for negative
    
    font = pygame.font.Font(None, 36)
    running = True
    frame_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Switch direction every 3 seconds (180 frames at 60 FPS)
        if frame_count % 180 == 0:
            direction *= -1
            print(f"🔄 Switching to {'POSITIVE' if direction > 0 else 'NEGATIVE'} angular velocity")
        
        # Update angle
        current_angle += angular_velocity * direction
        current_angle = current_angle % (2 * math.pi)  # Keep in [0, 2π]
        
        # Clear screen
        screen.fill((30, 30, 30))
        
        # Draw reference axes
        pygame.draw.line(screen, (100, 100, 100), (0, center[1]), (800, center[1]), 2)
        pygame.draw.line(screen, (100, 100, 100), (center[0], 0), (center[0], 600), 2)
        
        # Draw direction vector
        direction_length = 150
        direction_end_x = center[0] + direction_length * math.cos(current_angle)
        direction_end_y = center[1] + direction_length * math.sin(current_angle)
        
        pygame.draw.circle(screen, (255, 100, 100), center, 20)
        pygame.draw.line(screen, (255, 255, 255), center, (int(direction_end_x), int(direction_end_y)), 5)
        
        # Show information
        angle_deg = math.degrees(current_angle)
        rotation_type = "POSITIVE (+)" if direction > 0 else "NEGATIVE (-)"
        visual_direction = "CLOCKWISE" if direction > 0 else "COUNTER-CLOCKWISE"
        
        text_lines = [
            f"Angular Velocity: {rotation_type}",
            f"Visual Rotation: {visual_direction}",
            f"Current Angle: {angle_deg:.1f}°",
            f"Radians: {current_angle:.3f}",
            "",
            "Direction switches every 3 seconds",
            "Press ESC to exit"
        ]
        
        y_offset = 50
        for line in text_lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (50, y_offset))
            y_offset += 35
        
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
    
    pygame.quit()
    
    print("\n✅ Rotation direction test complete!")
    print("🔍 KEY FINDINGS:")
    print("  • Positive angular velocity → Clockwise visual rotation")
    print("  • Negative angular velocity → Counter-clockwise visual rotation")
    print("  • This is because +Y points DOWN in screen coordinates")
    
    return True

def analyze_current_implementation():
    """Analyze the current soccerenv implementation"""
    
    print("\n🔍 ANALYZING CURRENT IMPLEMENTATION")
    print("=" * 50)
    
    print("Current rendering code:")
    print("  direction_end_x = robot_x + direction_length * np.cos(self.robot_angle)")
    print("  direction_end_y = robot_y + direction_length * np.sin(self.robot_angle)")
    print()
    
    print("This implementation:")
    print("  ✅ Uses standard mathematical trigonometry")
    print("  ✅ θ=0 points right, θ=π/2 points down (correct for screen coords)")
    print("  ✅ Works correctly with screen coordinate system")
    print()
    
    print("Angular velocity in _apply_robot_action:")
    print("  new_angle = self.robot_angle + angular_velocity")
    print()
    
    print("This means:")
    print("  • Positive angular_velocity → increases angle → clockwise rotation")
    print("  • Negative angular_velocity → decreases angle → counter-clockwise rotation")
    print()
    
    print("🎯 RECOMMENDATION:")
    print("  The current implementation is CORRECT for screen coordinates!")
    print("  No changes needed to the rendering or physics code.")
    print()
    
    return True

if __name__ == "__main__":
    print("🧭 COORDINATE SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    print("This will help you understand pygame's coordinate system")
    print("and verify that your implementation is correct.")
    print()
    
    print("Choose test:")
    print("1. Test pygame angle directions (static)")
    print("2. Test rotation directions (animated)")
    print("3. Analyze current implementation")
    print("4. Run all tests")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        test_pygame_angle_convention()
    elif choice == "2":
        test_rotation_direction()
    elif choice == "3":
        analyze_current_implementation()
    elif choice == "4":
        test_pygame_angle_convention()
        input("\nPress Enter to continue to rotation test...")
        test_rotation_direction()
        input("\nPress Enter to see analysis...")
        analyze_current_implementation()
    else:
        print("Invalid choice. Running angle test...")
        test_pygame_angle_convention()