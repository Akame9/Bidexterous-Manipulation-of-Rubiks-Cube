"""
Script to evaluate a trained PPO model with visual feedback.
This allows you to see the learned cube manipulation behavior.
"""

import os
import sys
import numpy as np
import torch
import argparse
from datetime import datetime
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitive_controller.ppo import PPOAgent
from environment.rubiks_cube import RubiksCubeEnvironment


def evaluate_with_viewer(model_path, num_episodes=5, max_steps=1000):
    """
    Evaluate trained model with visual feedback.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print(f"Loading model from: {model_path}")
    
    # Create environment with viewer enabled
    env = RubiksCubeEnvironment(
        xml_path="xmls/bidexhands.xml",
        max_episode_steps=max_steps,
        enable_viewer=True,  # Enable visual feedback
        visualize_collision_boxes=True  # Show collision boxes
    )
    
    # Create agent
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load trained model
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("Close the viewer window or press Ctrl+C to stop evaluation")
    
    episode_rewards = []
    episode_lengths = []
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(range(num_episodes), desc="Evaluating with Viewer", unit="episode")
    
    try:
        for episode in eval_pbar:
            print(f"\n=== Episode {episode + 1} ===")
            
            state = env.initialize()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Use deterministic policy for evaluation
                action, _, _ = agent.select_action(state, deterministic=True)
                
                next_state, reward, done, info = env.take_step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Print step info every 100 steps
                if step % 100 == 0:
                    print(f"Step {step}: Reward = {reward:.3f}, Total = {episode_reward:.2f}")
                    print(f"  Cube pos: {info.get('cube_position', 'N/A')}")
                    print(f"  Contact count: {info.get('contact_count', 'N/A')}")
                
                if done:
                    print(f"Episode ended early at step {step}")
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Update progress bar
            eval_pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Length': f'{episode_length}',
                'Avg Reward': f'{np.mean(episode_rewards):.2f}'
            })
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Final Cube Position: {info.get('cube_position', 'N/A')}")
            
            # Wait for user input before next episode
            if episode < num_episodes - 1:
                input("Press Enter to continue to next episode...")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        eval_pbar.close()
        env.close()
    
    # Print final statistics
    if episode_rewards:
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\n=== Final Evaluation Results ===")
        print(f"Episodes evaluated: {len(episode_rewards)}")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Length: {avg_length:.2f}")
        print(f"Best Episode Reward: {max(episode_rewards):.2f}")
        print(f"Worst Episode Reward: {min(episode_rewards):.2f}")


def analyze_learned_behavior(model_path, num_episodes=10):
    """
    Analyze the learned behavior without viewer for detailed statistics.
    """
    print(f"Analyzing learned behavior from: {model_path}")
    
    # Create environment without viewer for faster analysis
    env = RubiksCubeEnvironment(
        xml_path="xmls/bidexhands.xml",
        max_episode_steps=1000,
        enable_viewer=False
    )
    
    # Create agent
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load trained model
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"\nAnalyzing behavior for {num_episodes} episodes...")
    
    # Detailed statistics
    all_rewards = []
    all_lengths = []
    contact_forces = []
    cube_rotations = []
    cube_positions = []
    
    # Create progress bar for analysis
    analyze_pbar = tqdm(range(num_episodes), desc="Analyzing Behavior", unit="episode")
    
    for episode in analyze_pbar:
        state = env.initialize()
        episode_reward = 0
        episode_length = 0
        episode_contacts = []
        episode_rotations = []
        episode_positions = []
        
        for step in range(1000):
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.take_step(action)
            
            # Collect detailed statistics
            episode_contacts.append(info.get('contact_count', 0))
            episode_positions.append(info.get('cube_position', [0, 0, 0]))
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        contact_forces.extend(episode_contacts)
        cube_positions.extend(episode_positions)
        
        # Update progress bar
        analyze_pbar.set_postfix({
            'Reward': f'{episode_reward:.2f}',
            'Avg Reward': f'{np.mean(all_rewards):.2f}',
            'Avg Contact': f'{np.mean(contact_forces):.1f}' if contact_forces else '0.0'
        })
    
    analyze_pbar.close()
    
    # Analyze results
    print(f"\n=== Detailed Behavior Analysis ===")
    print(f"Episodes analyzed: {num_episodes}")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(all_lengths):.2f}")
    print(f"Average Contact Count: {np.mean(contact_forces):.2f}")
    
    # Cube position analysis
    if cube_positions:
        cube_positions = np.array(cube_positions)
        print(f"Cube Position Statistics:")
        print(f"  X: {np.mean(cube_positions[:, 0]):.3f} ± {np.std(cube_positions[:, 0]):.3f}")
        print(f"  Y: {np.mean(cube_positions[:, 1]):.3f} ± {np.std(cube_positions[:, 1]):.3f}")
        print(f"  Z: {np.mean(cube_positions[:, 2]):.3f} ± {np.std(cube_positions[:, 2]):.3f}")
    
    # Success metrics
    successful_episodes = sum(1 for r in all_rewards if r > 10)  # Define success threshold
    print(f"Successful Episodes (reward > 10): {successful_episodes}/{num_episodes} ({100*successful_episodes/num_episodes:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    parser.add_argument('--model_path', type=str, default='saved_models/ppo_bidexhands_final.pth',
                       help='Path to the trained model file')
    parser.add_argument('--mode', type=str, choices=['viewer', 'analyze'], default='viewer',
                       help='Evaluation mode: viewer (visual) or analyze (detailed stats)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if args.mode == 'viewer':
        evaluate_with_viewer(args.model_path, args.episodes, args.max_steps)
    else:
        analyze_learned_behavior(args.model_path, args.episodes)


if __name__ == "__main__":
    main()
