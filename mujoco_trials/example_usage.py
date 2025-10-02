"""
Example usage script for PPO training on bidexhands manipulation.
This script demonstrates how to use the PPO agent and environment.
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitive_controller.ppo import PPOAgent
from environment.rubiks_cube import RubiksCubeEnvironment


def example_basic_training():
    """Example of basic training setup."""
    print("=== Basic Training Example ===")
    
    # Create environment
    env = RubiksCubeEnvironment(
        xml_path="bidexhands.xml",
        max_episode_steps=500,
        enable_viewer=False  # Set to True to see visualization
    )
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        device='cpu'  # Use 'cuda' if GPU is available
    )
    
    print(f"Environment state dimension: {env.state_dim}")
    print(f"Environment action dimension: {env.action_dim}")
    
    # Run a few episodes
    for episode in range(5):
        state = env.initialize()
        episode_reward = 0
        
        for step in range(100):  # Short episodes for demo
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.take_step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update agent
        agent.update()
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    print("Basic training example completed!")


def example_environment_usage():
    """Example of environment usage."""
    print("\n=== Environment Usage Example ===")
    
    # Create environment
    env = RubiksCubeEnvironment(
        xml_path="bidexhands.xml",
        max_episode_steps=200,
        enable_viewer=False
    )
    
    # Initialize environment
    state = env.initialize()
    print(f"Initial state shape: {state.shape}")
    
    # Get action breakdown
    actions = env.get_action(np.random.uniform(-0.1, 0.1, env.action_dim))
    print(f"Left hand actions shape: {actions['left_hand'].shape}")
    print(f"Right hand actions shape: {actions['right_hand'].shape}")
    print(f"Cube actions shape: {actions['cube'].shape}")
    
    # Run a few steps
    for step in range(10):
        # Random action
        action = np.random.uniform(-0.1, 0.1, env.action_dim)
        
        # Take step
        next_state, reward, done, info = env.take_step(action)
        
        print(f"Step {step + 1}: Reward = {reward:.3f}, Done = {done}")
        print(f"  Cube position: {info['cube_position']}")
        print(f"  Contact count: {info['contact_count']}")
        
        if done:
            print("Episode ended!")
            break
    
    env.close()
    print("Environment usage example completed!")


def example_agent_usage():
    """Example of PPO agent usage."""
    print("\n=== PPO Agent Usage Example ===")
    
    # Create agent
    state_dim = 100  # Example state dimension
    action_dim = 50  # Example action dimension
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        device='cpu'
    )
    
    # Generate some random data
    states = np.random.randn(10, state_dim)
    actions = np.random.randn(10, action_dim)
    
    print(f"Agent state dimension: {agent.state_dim}")
    print(f"Agent action dimension: {agent.action_dim}")
    
    # Test action selection
    for i in range(3):
        state = states[i]
        action, log_prob, value = agent.select_action(state)
        
        print(f"State {i + 1}:")
        print(f"  Action shape: {action.shape}")
        print(f"  Log prob shape: {log_prob.shape}")
        print(f"  Value shape: {value.shape}")
    
    # Test model saving/loading
    model_path = "example_model.pth"
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Create new agent and load model
    new_agent = PPOAgent(state_dim, action_dim, device='cpu')
    new_agent.load_model(model_path)
    print("Model loaded successfully!")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Example model file cleaned up")
    
    print("PPO agent usage example completed!")


def example_training_loop():
    """Example of a complete training loop."""
    print("\n=== Complete Training Loop Example ===")
    
    # Create environment
    env = RubiksCubeEnvironment(
        xml_path="bidexhands.xml",
        max_episode_steps=100,
        enable_viewer=False
    )
    
    # Create agent
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=5,
        device='cpu'
    )
    
    # Training loop
    num_episodes = 10
    max_steps = 50
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.initialize()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.take_step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Update agent
        agent.update()
        
        # Log progress
        if episode % 2 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Get training statistics
    stats = agent.get_training_stats()
    print(f"\nTraining completed!")
    print(f"Total episodes: {len(stats['episode_rewards'])}")
    print(f"Average reward: {np.mean(stats['episode_rewards']):.2f}")
    print(f"Average length: {np.mean(stats['episode_lengths']):.2f}")
    
    env.close()
    print("Complete training loop example completed!")


def main():
    """Run all examples."""
    print("PPO Bidexhands Manipulation - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_environment_usage()
        example_agent_usage()
        example_basic_training()
        example_training_loop()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo run full training, use:")
        print("python train_bidexhands_ppo.py --num_episodes 1000 --enable_viewer")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

