"""
Test script for testing the apply_rotation() function in RubiksCubeEnvironment.
This script creates an environment, applies rotations, and displays them in the MuJoCo viewer.
"""

import os
import sys
import numpy as np
import time
import argparse

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environment.rubiks_cube import RubiksCubeEnvironment


def test_single_rotation(env, rotation_spec, delay=2.0):
    """
    Test a single rotation and display it.
    
    Args:
        env: RubiksCubeEnvironment instance
        rotation_spec: Rotation specification (e.g., 'blue_clock', 'red_anti_clock')
        delay: Time to wait after rotation (seconds)
    """
    print(f"\n{'='*60}")
    print(f"Applying rotation: {rotation_spec}")
    print(f"{'='*60}")
    
    # Get current joint angle before rotation
    face = rotation_spec.split('_')[0] if '_' in rotation_spec else rotation_spec
    if face in env.face_joint_ids:
        joint_id = env.face_joint_ids[face]
        qpos_adr = env.model.jnt_qposadr[joint_id]
        angle_before = env.data.qpos[qpos_adr]
        print(f"Joint angle before: {np.degrees(angle_before):.2f}°")
    
    # Apply rotation
    success = env.apply_rotation(rotation_spec)
    
    if success:
        # Get joint angle after rotation
        if face in env.face_joint_ids:
            angle_after = env.data.qpos[qpos_adr]
            print(f"Joint angle after: {np.degrees(angle_after):.2f}°")
            print(f"Rotation applied successfully!")
        
        # Render and wait
        print(f"Viewing rotation for {delay} seconds...")
        for _ in range(int(delay * 100)):  # Render for delay seconds
            env.render()
            time.sleep(0.01)
    else:
        print(f"Failed to apply rotation: {rotation_spec}")
    
    return success


def test_rotation_sequence(env, rotation_sequence, delay=2.0):
    """
    Test a sequence of rotations.
    
    Args:
        env: RubiksCubeEnvironment instance
        rotation_sequence: List of rotation specifications
        delay: Time to wait after each rotation (seconds)
    """
    print(f"\n{'='*60}")
    print(f"Testing rotation sequence: {rotation_sequence}")
    print(f"{'='*60}")
    
    success = env.apply_rotation_sequence(rotation_sequence)
    
    if success:
        print(f"All rotations applied successfully!")
        print(f"Viewing final state for {delay} seconds...")
        for _ in range(int(delay * 100)):
            env.render()
            time.sleep(0.01)
    else:
        print(f"Some rotations failed!")
    
    return success


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test apply_rotation() function")
    parser.add_argument('--xml', type=str, default='xmls/bidexhands.xml', 
                       help='MuJoCo XML path')
    parser.add_argument('--rotation', type=str, default=None,
                       help='Single rotation to test (e.g., "blue_clock", "red_anti_clock")')
    parser.add_argument('--sequence', type=str, nargs='+', default=None,
                       help='Sequence of rotations to test (e.g., "blue_clock red_anti_clock white")')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Time to wait after each rotation (seconds)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode (wait for user input between rotations)')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all possible rotations')
    args = parser.parse_args()
    
    print("="*60)
    print("Rubik's Cube Rotation Test")
    print("="*60)
    
    # Create environment with viewer enabled
    print("\nCreating environment with viewer...")
    env = RubiksCubeEnvironment(
        xml_path=args.xml,
        enable_viewer=True,
        max_episode_steps=1000,
        gravity_vector=[0.0, 0.0, 0.0]
    )
    
    # Initialize environment
    print("Initializing environment...")
    state = env.initialize()
    print("Environment initialized!")
    
    # Wait a bit to see initial state
    print("\nShowing initial cube state for 2 seconds...")
    for _ in range(200):
        env.render()
        time.sleep(0.01)
    
    # Test single rotation
    if args.rotation:
        test_single_rotation(env, args.rotation, delay=args.delay)
    
    # Test rotation sequence
    elif args.sequence:
        if args.interactive:
            # Interactive mode: apply each rotation one by one
            for i, rotation_spec in enumerate(args.sequence):
                print(f"\nRotation {i+1}/{len(args.sequence)}")
                test_single_rotation(env, rotation_spec, delay=args.delay)
                if i < len(args.sequence) - 1:
                    input("Press Enter to continue to next rotation...")
        else:
            # Apply all rotations at once
            test_rotation_sequence(env, args.sequence, delay=args.delay)
    
    # Test all rotations
    elif args.test_all:
        print("\nTesting all possible rotations...")
        all_faces = ['red', 'orange', 'blue', 'green', 'white', 'yellow']
        all_rotations = []
        for face in all_faces:
            all_rotations.append(f"{face}_clock")
            all_rotations.append(f"{face}_anti_clock")
        
        if args.interactive:
            for i, rotation_spec in enumerate(all_rotations):
                print(f"\nRotation {i+1}/{len(all_rotations)}")
                test_single_rotation(env, rotation_spec, delay=args.delay)
                if i < len(all_rotations) - 1:
                    input("Press Enter to continue to next rotation...")
        else:
            for rotation_spec in all_rotations:
                test_single_rotation(env, rotation_spec, delay=args.delay)
    
    # Default: test a few example rotations
    else:
        print("\nNo specific rotation specified. Running default test sequence...")
        print("You can specify rotations using --rotation or --sequence flags.")
        print("Example: python test_rotation.py --rotation blue_clock")
        print("Example: python test_rotation.py --sequence blue_clock red_anti_clock white")
        
        # Default test sequence
        default_sequence = ['blue_clock', 'red_anti_clock', 'white', 'green_clock']
        print(f"\nRunning default test sequence: {default_sequence}")
        
        if args.interactive:
            for i, rotation_spec in enumerate(default_sequence):
                print(f"\nRotation {i+1}/{len(default_sequence)}")
                test_single_rotation(env, rotation_spec, delay=args.delay)
                if i < len(default_sequence) - 1:
                    input("Press Enter to continue to next rotation...")
        else:
            for rotation_spec in default_sequence:
                test_single_rotation(env, rotation_spec, delay=args.delay)
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    print("\nKeeping viewer open for 5 seconds...")
    for _ in range(500):
        env.render()
        time.sleep(0.01)
    
    # Close environment
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()

