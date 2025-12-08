"""
Evaluation script for trained PPO agent on bidexhands Rubik's cube manipulation.
This script loads a trained model and evaluates it with the MuJoCo viewer.
"""

import os
import sys
import numpy as np
import torch
import argparse
import time
import mujoco as mj
import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitive_controller.ppo import PPOAgent, get_device, print_gpu_info
from environment.rubiks_cube import RubiksCubeEnvironment


def evaluate_agent(env, agent, num_episodes=10, max_steps=1000, render=True, deterministic=True,
                   save_video=False, video_dir="videos", video_fps=30, video_width=640, video_height=480):
    """
    Evaluate the trained agent.
    
    Args:
        env: Environment instance
        agent: PPO agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        deterministic: Whether to use deterministic policy
    """
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    print(f"Deterministic policy: {deterministic}")
    print(f"Rendering: {render}")
    
    episode_rewards = []
    episode_lengths = []
    episode_infos = []
    
    # Track specific reward values
    reward_2_count = 0
    reward_1_count = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        video_writer = None
        renderer = None
        if save_video:
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"evaluation_episode_{episode+1}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, float(video_fps), (int(video_width), int(video_height)))
            renderer = mj.Renderer(env.model, width=int(video_width), height=int(video_height))
        state = env.initialize()
        state_before = state.copy()
        # Extract cube face joint positions and velocities from qpos and qvel
        # Get joint IDs and their addresses in qpos/qvel arrays
        cube_joint_info = {}
        for face_name, joint_name in env.face_to_joint_map.items():
            if face_name in env.face_joint_ids:
                joint_id = env.face_joint_ids[face_name]
                qpos_adr = env.model.jnt_qposadr[joint_id]
                qvel_adr = env.model.jnt_dofadr[joint_id]
                cube_joint_info[face_name] = {
                    'joint_id': joint_id,
                    'joint_name': joint_name,
                    'qpos_adr': qpos_adr,
                    'qvel_adr': qvel_adr
                }
        
        # Extract cube joint positions and velocities from state before rotation
        # State structure: [joint_pos(nq), joint_vel(nv), ...]
        cube_joint_pos_before = {}
        cube_joint_vel_before = {}
        for face_name, info in cube_joint_info.items():
            qpos_adr = info['qpos_adr']
            qvel_adr = info['qvel_adr']
            # Get joint position (for hinge joints, it's a single value)
            cube_joint_pos_before[face_name] = state[qpos_adr]
            # Get joint velocity
            cube_joint_vel_before[face_name] = state[env.model.nq + qvel_adr]
        
        print(f"\n{'='*60}")
        print(f"CUBE JOINT STATE BEFORE apply_rotation('white_anti_clock'):")
        print(f"{'='*60}")
        for face_name in sorted(cube_joint_info.keys()):
            joint_name = cube_joint_info[face_name]['joint_name']
            pos_deg = np.degrees(cube_joint_pos_before[face_name])
            vel_deg = np.degrees(cube_joint_vel_before[face_name])
            print(f"  {face_name:8s} (joint: {joint_name:3s}): Position = {cube_joint_pos_before[face_name]:8.6f} rad ({pos_deg:7.3f}°), "
                  f"Velocity = {cube_joint_vel_before[face_name]:8.6f} rad/s ({vel_deg:7.3f}°/s)")
        
        # Apply rotation
        env.apply_rotation('white_anti_clock')
        env.apply_rotation('blue_clock')
        
        # env._set_neutral_pose()
        # Get state after rotation
        state_after = env.get_state()
        
        # Extract cube joint positions and velocities from state after rotation
        cube_joint_pos_after = {}
        cube_joint_vel_after = {}
        for face_name, info in cube_joint_info.items():
            qpos_adr = info['qpos_adr']
            qvel_adr = info['qvel_adr']
            cube_joint_pos_after[face_name] = state_after[qpos_adr]
            cube_joint_vel_after[face_name] = state_after[env.model.nq + qvel_adr]
        
        print(f"\n{'='*60}")
        print(f"CUBE JOINT STATE AFTER apply_rotation('white_anti_clock'):")
        print(f"{'='*60}")
        for face_name in sorted(cube_joint_info.keys()):
            joint_name = cube_joint_info[face_name]['joint_name']
            pos_deg = np.degrees(cube_joint_pos_after[face_name])
            vel_deg = np.degrees(cube_joint_vel_after[face_name])
            print(f"  {face_name:8s} (joint: {joint_name:3s}): Position = {cube_joint_pos_after[face_name]:8.6f} rad ({pos_deg:7.3f}°), "
                  f"Velocity = {cube_joint_vel_after[face_name]:8.6f} rad/s ({vel_deg:7.3f}°/s)")
        
        print(f"\n{'='*60}")
        print(f"CHANGES:")
        print(f"{'='*60}")
        for face_name in sorted(cube_joint_info.keys()):
            joint_name = cube_joint_info[face_name]['joint_name']
            pos_change = cube_joint_pos_after[face_name] - cube_joint_pos_before[face_name]
            vel_change = cube_joint_vel_after[face_name] - cube_joint_vel_before[face_name]
            pos_change_deg = np.degrees(pos_change)
            vel_change_deg = np.degrees(vel_change)
            print(f"  {face_name:8s} (joint: {joint_name:3s}): Position change = {pos_change:8.6f} rad ({pos_change_deg:7.3f}°), "
                  f"Velocity change = {vel_change:8.6f} rad/s ({vel_change_deg:7.3f}°/s)")
        print(f"{'='*60}\n")
        
        # Calculate initial changes to ALL state space elements (changes from before to after initial rotations)
        # State structure: [joint_pos(nq), joint_vel(nv), cube_pos(3), cube_quat(4), 
        #                    cube_lin_vel(3), cube_ang_vel(3), contact_force(3), prev_actions]
        state_initial_changes = state_after - state_before
        
        # Also keep cube joint-specific changes for debugging/printing
        cube_joint_pos_initial_changes = {}
        cube_joint_vel_initial_changes = {}
        for face_name in cube_joint_info.keys():
            cube_joint_pos_initial_changes[face_name] = cube_joint_pos_after[face_name] - cube_joint_pos_before[face_name]
            cube_joint_vel_initial_changes[face_name] = cube_joint_vel_after[face_name] - cube_joint_vel_before[face_name]
        
        print(f"\n{'='*60}")
        print(f"INITIAL STATE CHANGES SUMMARY:")
        print(f"{'='*60}")
        print(f"  Total state dimension: {len(state_initial_changes)}")
        print(f"  Joint positions (nq={env.model.nq}): changes in range [{np.min(state_initial_changes[:env.model.nq]):.6f}, {np.max(state_initial_changes[:env.model.nq]):.6f}]")
        print(f"  Joint velocities (nv={env.model.nv}): changes in range [{np.min(state_initial_changes[env.model.nq:env.model.nq+env.model.nv]):.6f}, {np.max(state_initial_changes[env.model.nq:env.model.nq+env.model.nv]):.6f}]")
        nq_nv = env.model.nq + env.model.nv
        print(f"  Cube position (3): {state_initial_changes[nq_nv:nq_nv+3]}")
        print(f"  Cube quaternion (4): {state_initial_changes[nq_nv+3:nq_nv+7]}")
        print(f"  Cube linear velocity (3): {state_initial_changes[nq_nv+7:nq_nv+10]}")
        print(f"  Cube angular velocity (3): {state_initial_changes[nq_nv+10:nq_nv+13]}")
        print(f"  Contact force (3): {state_initial_changes[nq_nv+13:nq_nv+16]}")
        print(f"  Previous actions ({len(env.hand_actuators)}): changes in range [{np.min(state_initial_changes[nq_nv+16:]):.6f}, {np.max(state_initial_changes[nq_nv+16:]):.6f}]")
        print(f"{'='*60}\n")
        
        # Update state to the state after rotation
        state = state_after
        
        episode_reward = 0
        episode_length = 0
        
        # Track rewards for this episode
        episode_reward_2_count = 0
        episode_reward_1_count = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        for step in range(max_steps):
            # At the beginning of each step, subtract initial changes from ALL state elements
            # This makes the agent see the state as if the initial changes never happened
            # while keeping the actual simulation state intact
            state_for_agent = state.copy()
            
            # Subtract initial changes from all state elements
            # state_for_agent = current_state - (state_after - state_before)
            # This effectively makes the agent see state relative to initial state_before
            state_for_agent = state_for_agent - state_initial_changes
            
            # Print the cube joint positions and velocities after subtracting initial changes (for debugging)
            if step == 0:
                print(f"\nCube joint positions and velocities after subtracting initial changes (step 0):")
                for face_name, info in cube_joint_info.items():
                    qpos_adr = info['qpos_adr']
                    qvel_adr = info['qvel_adr']
                    pos_deg = np.degrees(state_for_agent[qpos_adr])
                    vel_deg = np.degrees(state_for_agent[env.model.nq + qvel_adr])
                    print(f"  {face_name:8s} (joint: {info['joint_name']:3s}): Position = {state_for_agent[qpos_adr]:8.6f} rad ({pos_deg:7.3f}°), "
                          f"Velocity = {state_for_agent[env.model.nq + qvel_adr]:8.6f} rad/s ({vel_deg:7.3f}°/s)")
                print()
            
            # Use deterministic policy for evaluation with modified state
            action, _, _ = agent.select_action(state_for_agent, deterministic=deterministic)
            
            # Take step (this uses the actual simulation state, not the modified one)
            next_state, reward, done, info = env.take_step(action)
            
            # Track specific reward values
            if abs(reward - 2.0) < 1e-6:  # Check for exactly 2.0
                reward_2_count += 1
                episode_reward_2_count += 1
            elif abs(reward - 1.0) < 1e-6:  # Check for exactly 1.0
                reward_1_count += 1
                episode_reward_1_count += 1
            
            total_steps += 1
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Print step info every 50 steps
            if step % 50 == 0:
                cube_pos = info['cube_position']
                print(f"  Step {step:4d}: Reward={reward:7.3f}, Cube pos=[{cube_pos[0]:6.3f}, {cube_pos[1]:6.3f}, {cube_pos[2]:6.3f}]")
            
            # Render if enabled and/or record video
            if render:
                env.render()
                time.sleep(0.01)
            if save_video and renderer is not None and video_writer is not None:
                renderer.update_scene(env.data)
                frame_rgb = renderer.render()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            if done:
                print(f"\n  Episode ended: {info.get('termination_reason', 'unknown')}")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_infos.append(info)

        if save_video and video_writer is not None:
            video_writer.release()
            video_writer = None
        if renderer is not None:
            renderer.close()
            renderer = None
        
        print(f"\n  Episode Summary:")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Episode Length: {episode_length}")
        print(f"    Final Cube Position: {info['cube_position']}")
        print(f"    Contacts: {info['contact_count']}")
        print(f"    Reward 2.0 count: {episode_reward_2_count} (this episode)")
        print(f"    Reward 1.0 count: {episode_reward_1_count} (this episode)")
    
    # Overall statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Length: {avg_length:.2f} ± {std_length:.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    print(f"")
    print(f"REWARD BREAKDOWN:")
    print(f"  Total Steps: {total_steps}")
    print(f"  Reward 2.0 count: {reward_2_count} ({reward_2_count/total_steps*100:.1f}% of steps)")
    print(f"  Reward 1.0 count: {reward_1_count} ({reward_1_count/total_steps*100:.1f}% of steps)")
    print(f"  Other rewards: {total_steps - reward_2_count - reward_1_count} ({(total_steps - reward_2_count - reward_1_count)/total_steps*100:.1f}% of steps)")
    print(f"{'='*60}\n")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'std_length': std_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_infos': episode_infos,
        'reward_2_count': reward_2_count,
        'reward_1_count': reward_1_count,
        'total_steps': total_steps
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent on Rubik\'s Cube environment')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='saved_models/ppo_model_best.pth',
                       help='Path to trained model')
    
    # Environment arguments
    parser.add_argument('--xml', type=str, default='xmls/bidexhands.xml',
                       help='MuJoCo XML path')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Max steps per episode')
    parser.add_argument('--enable_viewer', action='store_true', default=True,
                       help='Enable MuJoCo viewer')
    parser.add_argument('--no_viewer', action='store_true',
                       help='Disable MuJoCo viewer')
    parser.add_argument('--visualize_collision_boxes', action='store_true',
                       help='Visualize collision boxes')
    parser.add_argument('--rotation_sequence', type=str, default=None,
                       help='Comma-separated list of face names to rotate (e.g., "red,blue,white")')
    
    # Evaluation arguments
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy')

    # Video recording arguments
    parser.add_argument('--save_video', action='store_true',
                       help='Save MP4 video(s) of evaluation episodes')
    parser.add_argument('--video_dir', type=str, default='videos',
                       help='Directory to save evaluation videos')
    parser.add_argument('--video_fps', type=int, default=30,
                       help='Frames per second for saved videos')
    parser.add_argument('--video_width', type=int, default=640,
                       help='Video width in pixels')
    parser.add_argument('--video_height', type=int, default=480,
                       help='Video height in pixels')
    
    # Agent arguments (must match training configuration)
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (must match training)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (must match training)')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                       help='PPO clip epsilon (must match training)')
    parser.add_argument('--k_epochs', type=int, default=10,
                       help='PPO update epochs (must match training)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy coefficient (must match training)')
    parser.add_argument('--value_coef', type=float, default=0.5,
                       help='Value loss coefficient (must match training)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Max grad norm (must match training)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (must match training)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Override viewer setting if --no_viewer is specified
    if args.no_viewer:
        args.enable_viewer = False
    
    # Override deterministic setting if --stochastic is specified
    if args.stochastic:
        args.deterministic = False
    
    # Parse rotation_sequence if provided
    rotation_sequence = None
    if args.rotation_sequence:
        rotation_sequence = [face.strip() for face in args.rotation_sequence.split(',')]
        print(f"Rotation sequence: {rotation_sequence}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print(f"Please check the path and try again.")
        return
    
    print(f"Loading model from: {args.model_path}")
    
    # Determine device
    device = get_device(args.device, args.gpu_id)
    print_gpu_info(device)
    
    # Create environment
    print("\nCreating environment...")
    env = RubiksCubeEnvironment(
        xml_path=args.xml,
        max_episode_steps=args.max_steps,
        enable_viewer=args.enable_viewer,
        visualize_collision_boxes=args.visualize_collision_boxes,
        rotation_sequence=rotation_sequence
    )
    
    # Get state and action dimensions
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
        batch_size=args.batch_size
    )
    
    # Load trained model
    print(f"\nLoading trained model from {args.model_path}...")
    agent.load_model(args.model_path)
    
    # Set model to evaluation mode
    agent.policy.eval()
    
    # Evaluate
    try:
        results = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.enable_viewer,
            deterministic=args.deterministic,
            save_video=args.save_video,
            video_dir=args.video_dir,
            video_fps=args.video_fps,
            video_width=args.video_width,
            video_height=args.video_height
        )
        
        # Save results
        results_file = f"evaluation_results_{os.path.basename(args.model_path).replace('.pth', '')}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Episodes: {args.num_episodes}\n")
            f.write(f"Deterministic: {args.deterministic}\n")
            f.write(f"\nResults:\n")
            f.write(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"Average Length: {results['avg_length']:.2f} ± {results['std_length']:.2f}\n")
            f.write(f"\nEpisode Details:\n")
            for i, (reward, length) in enumerate(zip(results['episode_rewards'], results['episode_lengths'])):
                f.write(f"Episode {i+1}: Reward={reward:.2f}, Length={length}\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment closed")


if __name__ == "__main__":
    main()

