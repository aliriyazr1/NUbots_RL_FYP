"""
Quick test to check if ball physics needs momentum smoothing.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.environments.soccerenv import SoccerEnv

def test_ball_jerkiness():
    """Test if current ball physics causes jerky movement."""

    env = SoccerEnv(render_mode=None)
    env.reset()

    # Test scenario: Robot dribbling the ball
    env.robot_pos = np.array([2.0, 4.5])
    env.ball_pos = np.array([2.3, 4.5])  # Ball slightly ahead
    env.ball_vel = np.array([0.0, 0.0])

    ball_velocities = []
    ball_positions = []
    robot_positions = []

    # Simulate dribbling for 2 seconds
    for i in range(120):  # 2 seconds at 60 FPS
        # Robot moves forward with slight oscillation (typical dribbling pattern)
        env.robot_vel = np.array([1.5 + 0.3 * np.sin(i * 0.2), 0.1 * np.cos(i * 0.3)])
        env.robot_pos += env.robot_vel * env.dt

        # Update physics
        env._update_ball_physics()
        env.ball_pos += env.ball_vel * env.dt

        # Apply friction
        env.ball_vel *= (1 - env.ball_friction * env.dt)

        # Store data
        ball_velocities.append(env.ball_vel.copy())
        ball_positions.append(env.ball_pos.copy())
        robot_positions.append(env.robot_pos.copy())

    ball_velocities = np.array(ball_velocities)
    ball_positions = np.array(ball_positions)
    robot_positions = np.array(robot_positions)

    # Calculate jerkiness metrics
    velocity_changes = np.diff(ball_velocities, axis=0)
    accelerations = velocity_changes / env.dt
    jerk = np.diff(accelerations, axis=0) / env.dt

    max_accel = np.max(np.linalg.norm(accelerations, axis=1))
    mean_accel = np.mean(np.linalg.norm(accelerations, axis=1))
    max_jerk = np.max(np.linalg.norm(jerk, axis=1))
    mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))

    print("\n=== BALL PHYSICS JERKINESS ANALYSIS ===")
    print(f"\nAcceleration metrics:")
    print(f"  Max acceleration:  {max_accel:.2f} pixels/s²")
    print(f"  Mean acceleration: {mean_accel:.2f} pixels/s²")

    print(f"\nJerk metrics (smoothness indicator):")
    print(f"  Max jerk:  {max_jerk:.2f} pixels/s³")
    print(f"  Mean jerk: {mean_jerk:.2f} pixels/s³")

    # Thresholds for smooth gameplay
    SMOOTH_MAX_ACCEL = 500  # pixels/s²
    SMOOTH_MAX_JERK = 10000  # pixels/s³

    print(f"\nSmooth gameplay thresholds:")
    print(f"  Max acceleration should be < {SMOOTH_MAX_ACCEL}")
    print(f"  Max jerk should be < {SMOOTH_MAX_JERK}")

    needs_smoothing = False

    if max_accel > SMOOTH_MAX_ACCEL:
        print("\n⚠️  WARNING: High acceleration detected!")
        print("   Ball movement may appear jerky")
        needs_smoothing = True

    if max_jerk > SMOOTH_MAX_JERK:
        print("\n⚠️  WARNING: High jerk detected!")
        print("   Ball physics are not smooth")
        needs_smoothing = True

    if not needs_smoothing:
        print("\n✅ Ball physics appear smooth enough!")
        print("   Momentum smoothing may not be necessary")
    else:
        print("\n🔧 RECOMMENDATION: Enable momentum smoothing")
        print("   Uncomment lines 1341-1350 in _update_ball_physics")

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Trajectory
    ax = axes[0, 0]
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', alpha=0.5, label='Robot')
    ax.plot(ball_positions[:, 0], ball_positions[:, 1], 'r-', alpha=0.7, label='Ball')
    ax.scatter(robot_positions[0, 0], robot_positions[0, 1], c='b', s=100, marker='o')
    ax.scatter(ball_positions[0, 0], ball_positions[0, 1], c='r', s=100, marker='o')
    ax.set_title('Dribbling Trajectory')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Ball velocity
    ax = axes[0, 1]
    ax.plot(ball_velocities[:, 0], label='X velocity')
    ax.plot(ball_velocities[:, 1], label='Y velocity')
    ax.set_title('Ball Velocity Components')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Velocity (pixels/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ball speed
    ax = axes[0, 2]
    ball_speed = np.linalg.norm(ball_velocities, axis=1)
    ax.plot(ball_speed)
    ax.set_title('Ball Speed')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Speed (pixels/s)')
    ax.grid(True, alpha=0.3)

    # Acceleration magnitude
    ax = axes[1, 0]
    accel_mag = np.linalg.norm(accelerations, axis=1)
    ax.plot(accel_mag)
    ax.axhline(y=SMOOTH_MAX_ACCEL, color='r', linestyle='--', label='Smooth threshold')
    ax.set_title('Ball Acceleration Magnitude')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Acceleration (pixels/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Jerk magnitude
    ax = axes[1, 1]
    jerk_mag = np.linalg.norm(jerk, axis=1)
    ax.plot(jerk_mag)
    ax.axhline(y=SMOOTH_MAX_JERK, color='r', linestyle='--', label='Smooth threshold')
    ax.set_title('Ball Jerk Magnitude (Smoothness)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Jerk (pixels/s³)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of accelerations
    ax = axes[1, 2]
    ax.hist(accel_mag, bins=30, alpha=0.7, color='blue')
    ax.axvline(x=SMOOTH_MAX_ACCEL, color='r', linestyle='--', label='Smooth threshold')
    ax.set_title('Acceleration Distribution')
    ax.set_xlabel('Acceleration (pixels/s²)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Ball Physics Smoothness Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ball_physics_jerkiness_analysis.png', dpi=150)
    plt.show()

    print(f"\n📊 Visualization saved to 'ball_physics_jerkiness_analysis.png'")

    return needs_smoothing


if __name__ == "__main__":
    needs_fix = test_ball_jerkiness()

    if needs_fix:
        print("\n" + "="*60)
        print("HOW TO FIX:")
        print("="*60)
        print("\nIn src/environments/soccerenv.py, uncomment lines 1341-1350:")
        print("""
        # Initialize momentum tracking
        if not hasattr(self, '_ball_push_momentum'):
            self._ball_push_momentum = np.array([0.0, 0.0])

        # Smooth momentum change
        momentum_change_rate = 0.3  # Tune this value
        self._ball_push_momentum += (push_velocity - self._ball_push_momentum) * momentum_change_rate
        self.ball_vel += self._ball_push_momentum

        # Comment out line 1352:
        # self.ball_vel += push_velocity  # Remove this direct addition
        """)
        print("\nTuning momentum_change_rate:")
        print("  - Lower values (0.1-0.2): Smoother but less responsive")
        print("  - Higher values (0.4-0.5): More responsive but less smooth")
        print("  - Recommended: Start with 0.3")