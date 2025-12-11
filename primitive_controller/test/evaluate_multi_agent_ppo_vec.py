"""
Evaluation script for trained Multi-Agent PPO Vec agent on bidexhands Rubik's cube manipulation.
This script loads a trained multi_agent_ppo_vec.py model and evaluates it with the MuJoCo viewer.
"""

import os
import sys
import numpy as np
import torch
import argparse
import time
import mujoco as mj
import cv2

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from primitive_controller.multi_agent_ppo_vec import MultiAgentPPOVec
from primitive_controller.ppo import get_device, print_gpu_info
from environment.rubiks_cube import RubiksCubeEnvironment


def evaluate_agent(env, multi_agent, num_episodes=10, max_steps=1000, render=True, deterministic=True,
                   save_video=False, video_dir="videos", video_fps=30, video_width=640, video_height=480):
    """
    Evaluate the trained multi-agent.
    
    Args:
        env: Environment instance
        multi_agent: MultiAgentPPOVec instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        deterministic: Whether to use deterministic policy
    """
    print(f"\nEvaluating multi-agent for {num_episodes} episodes...")
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
        episode_reward = 0
        episode_length = 0
        
        # Track rewards for this episode
        episode_reward_2_count = 0
        episode_reward_1_count = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        for step in range(max_steps):
            # Select actions for both hands using deterministic policy
            left_action, right_action, _, _, _, _ = multi_agent.select_actions(state, deterministic=deterministic)
            
            # Combine actions (left hand actions + right hand actions)
            combined_action = np.concatenate([left_action, right_action])
            
            # Take step
            next_state, reward, done, info = env.take_step(combined_action)
            
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
    parser = argparse.ArgumentParser(description='Evaluate trained Multi-Agent PPO Vec on Rubik\'s Cube environment')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='saved_models/multi_agent_ppo_vec_model_best.pth',
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
    parser.add_argument('--high_reward_threshold', type=float, default=1.5,
                       help='Reward threshold for high-reward prioritization (must match training)')
    parser.add_argument('--clip_vloss', action='store_true', default=True,
                       help='Use clipped value loss (must match training)')
    parser.add_argument('--no_clip_vloss', dest='clip_vloss', action='store_false',
                       help='Disable clipped value loss')
    parser.add_argument('--value_clip_coef', type=float, default=0.2,
                       help='Value clipping coefficient (must match training)')
    parser.add_argument('--value_loss_mode', type=str, default='min', choices=['max', 'min', 'mean'],
                       help='Value loss mode (must match training)')
    parser.add_argument('--target_kl', type=float, default=0.01,
                       help='Target KL divergence for early stopping (must match training)')
    parser.add_argument('--early_stop_kl', action='store_true', default=False,
                       help='Enable early stopping based on KL divergence (must match training)')
    parser.add_argument('--use_torch_compile', action='store_true', default=False,
                       help='Use torch.compile() (must match training)')
    parser.add_argument('--num_envs', type=int, default=8,
                       help='Number of parallel environments (must match training)')
    
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
    left_action_dim = len(env.left_hand_actuators)
    right_action_dim = len(env.right_hand_actuators)
    
    print(f"State dimension: {state_dim}")
    print(f"Left hand action dimension: {left_action_dim}")
    print(f"Right hand action dimension: {right_action_dim}")
    
    # Create Multi-Agent PPO Vec
    print("\nCreating Multi-Agent PPO Vec...")
    multi_agent = MultiAgentPPOVec(
        state_dim=state_dim,
        left_action_dim=left_action_dim,
        right_action_dim=right_action_dim,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
        batch_size=args.batch_size,
        high_reward_threshold=args.high_reward_threshold,
        clip_vloss=args.clip_vloss,
        value_clip_coef=args.value_clip_coef,
        value_loss_mode=args.value_loss_mode,
        target_kl=args.target_kl,
        early_stop_kl=args.early_stop_kl,
        use_torch_compile=args.use_torch_compile,
        num_envs=args.num_envs,
        max_steps=args.max_steps
    )
    
    # Load trained model
    print(f"\nLoading trained model from {args.model_path}...")
    multi_agent.load_model(args.model_path)
    
    # Set models to evaluation mode
    # Handle torch.compile() wrapped models
    if hasattr(multi_agent.left_agent.policy, '_orig_mod'):
        multi_agent.left_agent.policy._orig_mod.eval()
    else:
        multi_agent.left_agent.policy.eval()
    
    if hasattr(multi_agent.right_agent.policy, '_orig_mod'):
        multi_agent.right_agent.policy._orig_mod.eval()
    else:
        multi_agent.right_agent.policy.eval()
    
    # Evaluate
    try:
        results = evaluate_agent(
            env=env,
            multi_agent=multi_agent,
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

