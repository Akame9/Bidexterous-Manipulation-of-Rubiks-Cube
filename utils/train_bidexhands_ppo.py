"""
Main training script for PPO agent on bidexhands Rubik's cube manipulation.
This script connects the PPO agent with the Rubik's cube environment and handles training.
"""

import os
import sys
import numpy as np
import torch
import argparse
from datetime import datetime
import time
from tqdm import tqdm

# Weights & Biases for experiment tracking
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitive_controller.ppo import PPOAgent, train_ppo_agent
from environment.rubiks_cube import RubiksCubeEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO agent on bidexhands manipulation')
    
    # Environment arguments
    parser.add_argument('--xml_path', type=str, default='xmls/bidexhands.xml',
                       help='Path to MuJoCo XML model file')
    parser.add_argument('--max_episode_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--enable_viewer', action='store_true',
                       help='Enable visual rendering during training')
    
    # Training arguments
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--k_epochs', type=int, default=10,
                       help='Number of epochs for policy update')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                       help='Value function loss coefficient')
    
    # Model saving arguments
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Interval for saving model')
    parser.add_argument('--model_path', type=str, default='saved_models/ppo_bidexhands',
                       help='Path to save model')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load pre-trained model')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for PPO updates (reduce if out of memory)')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Interval for logging training progress')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='rubiks-cube-ppo',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                       help='Tags for the wandb run')
    
    return parser.parse_args()


def setup_device(device_arg, force_cpu=False):
    """Setup computation device."""
    if force_cpu:
        device = 'cpu'
        print("Forcing CPU usage")
    elif device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    
    # Print GPU memory info if using CUDA
    if device == 'cuda' and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print("If you encounter CUDA out of memory errors, try:")
        print("  1. Use --force_cpu flag")
        print("  2. Reduce --batch_size (default: 256)")
        print("  3. Reduce --max_episode_steps")
    
    return device


def create_directories(model_path):
    """Create necessary directories."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)


def setup_wandb(args, env, agent):
    """Initialize Weights & Biases logging."""
    if not args.use_wandb or not _WANDB_AVAILABLE:
        return False
    
    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_run_name = f"ppo_rubiks_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config={
            # Environment config
            'env_name': 'RubiksCubeEnvironment',
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'max_episode_steps': args.max_episode_steps,
            
            # Training config
            'num_episodes': args.num_episodes,
            'learning_rate': args.lr,
            'gamma': args.gamma,
            'eps_clip': args.eps_clip,
            'k_epochs': args.k_epochs,
            'entropy_coef': args.entropy_coef,
            'value_coef': args.value_coef,
            'batch_size': args.batch_size,
            
            # System config
            'device': args.device,
            'force_cpu': args.force_cpu,
            'save_interval': args.save_interval,
            'log_interval': args.log_interval,
        }
    )
    
    # Watch the model
    wandb.watch(agent.policy, log='all', log_freq=100)
    
    print(f"Weights & Biases initialized: {wandb.run.url}")
    
    # Save wandb link to logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_log_path = f"logs/wandb_run_{timestamp}.txt"
    
    with open(wandb_log_path, 'w') as f:
        f.write(f"Weights & Biases Run Information\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Run Name: {wandb.run.name}\n")
        f.write(f"Run ID: {wandb.run.id}\n")
        f.write(f"Project: {wandb.run.project}\n")
        f.write(f"URL: {wandb.run.url}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"-" * 20 + "\n")
        for key, value in wandb.config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTo view this run later, visit: {wandb.run.url}\n")
    
    print(f"Wandb run info saved to: {wandb_log_path}")
    return True


def evaluate_agent(env, agent, num_episodes=10, max_steps=500):
    """Evaluate the trained agent."""
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(range(num_episodes), desc="Evaluating", unit="episode")
    
    for episode in eval_pbar:
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
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update progress bar
        eval_pbar.set_postfix({
            'Reward': f'{episode_reward:.2f}',
            'Length': f'{episode_length}',
            'Avg Reward': f'{np.mean(episode_rewards):.2f}'
        })
        
        # Print detailed info every few episodes
        if (episode + 1) % 5 == 0 or episode == 0:
            tqdm.write(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    eval_pbar.close()
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Length: {avg_length:.2f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device, args.force_cpu)
    
    # Create directories
    create_directories(args.model_path)
    
    # Create environment
    print("Creating environment...")
    env = RubiksCubeEnvironment(
        xml_path=args.xml_path,
        max_episode_steps=args.max_episode_steps,
        enable_viewer=args.enable_viewer
    )
    
    # Get state and action dimensions
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create PPO agent
    print("Creating PPO agent...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        device=device
    )
    
    # Setup Weights & Biases
    use_wandb = setup_wandb(args, env, agent)
    
    # Load pre-trained model if specified
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}")
        agent.load_model(args.load_model)
    
    # Training
    print(f"\nStarting training for {args.num_episodes} episodes...")
    print(f"Model will be saved to {args.model_path}")
    
    start_time = datetime.now()
    
    try:
        # Create progress bar for episodes
        episode_pbar = tqdm(range(args.num_episodes), desc="Training Episodes", unit="episode")
        
        # Train the agent
        for episode in episode_pbar:
            episode_start_time = time.time()
            state = env.initialize()
            episode_reward = 0
            episode_length = 0
            
            for step in range(args.max_episode_steps):
                # Select action
                action, log_prob, value = agent.select_action(state)
                
                # Take step in environment
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
            
            # Store episode statistics
            agent.training_stats['episode_rewards'].append(episode_reward)
            agent.training_stats['episode_lengths'].append(episode_length)
            
            # Log to Weights & Biases
            if use_wandb:
                log_dict = {
                    'episode': episode,
                    'episode/reward': episode_reward,
                    'episode/length': episode_length,
                    'episode/steps_per_second': episode_length / max(time.time() - episode_start_time, 1e-6),
                }
                
                # Add environment-specific metrics from info
                if 'cube_position' in info:
                    cube_pos = info['cube_position']
                    log_dict.update({
                        'cube/position_x': cube_pos[0],
                        'cube/position_y': cube_pos[1],
                        'cube/position_z': cube_pos[2],
                        'cube/distance_from_center': np.linalg.norm(cube_pos[:2])
                    })
                
                if 'contact_count' in info:
                    log_dict['cube/contact_count'] = info['contact_count']
                
                # Add training losses if available
                if agent.training_stats['policy_loss']:
                    log_dict.update({
                        'loss/policy': agent.training_stats['policy_loss'][-1],
                        'loss/value': agent.training_stats['value_loss'][-1],
                        'loss/entropy': agent.training_stats['entropy_loss'][-1],
                        'loss/total': agent.training_stats['total_loss'][-1],
                    })
                
                # Add rolling averages
                if len(agent.training_stats['episode_rewards']) >= 10:
                    log_dict.update({
                        'episode/avg_reward_10': np.mean(agent.training_stats['episode_rewards'][-10:]),
                        'episode/avg_length_10': np.mean(agent.training_stats['episode_lengths'][-10:]),
                    })
                
                if len(agent.training_stats['episode_rewards']) >= 100:
                    log_dict.update({
                        'episode/avg_reward_100': np.mean(agent.training_stats['episode_rewards'][-100:]),
                        'episode/avg_length_100': np.mean(agent.training_stats['episode_lengths'][-100:]),
                    })
                
                wandb.log(log_dict)
            
            # Update progress bar with current statistics
            if len(agent.training_stats['episode_rewards']) >= 10:
                avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                avg_length = np.mean(agent.training_stats['episode_lengths'][-10:])
                episode_pbar.set_postfix({
                    'Avg Reward': f'{avg_reward:.2f}',
                    'Avg Length': f'{avg_length:.1f}',
                    'Current Reward': f'{episode_reward:.2f}'
                })
            else:
                episode_pbar.set_postfix({
                    'Current Reward': f'{episode_reward:.2f}',
                    'Length': f'{episode_length}'
                })
            
            # Log progress (less frequent with progress bar)
            if episode % args.log_interval == 0 and episode > 0:
                avg_reward = np.mean(agent.training_stats['episode_rewards'][-args.log_interval:])
                avg_length = np.mean(agent.training_stats['episode_lengths'][-args.log_interval:])
                tqdm.write(f"Episode {episode:4d} | Avg Reward: {avg_reward:8.2f} | Avg Length: {avg_length:6.1f}")
            
            # Save model
            if episode % args.save_interval == 0 and episode > 0:
                model_path = f"{args.model_path}_episode_{episode}.pth"
                agent.save_model(model_path)
                tqdm.write(f"Model saved: {model_path}")
                
                # Log model checkpoint to wandb
                if use_wandb:
                    wandb.save(model_path)
        
        # Close progress bar
        episode_pbar.close()
        
        # Save final model
        final_model_path = f"{args.model_path}_final.pth"
        agent.save_model(final_model_path)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\nTraining completed in {training_time}")
        print(f"Final model saved to {final_model_path}")
        
        # Log final model to wandb
        if use_wandb:
            wandb.save(final_model_path)
            wandb.log({
                'training/total_time_seconds': training_time.total_seconds(),
                'training/episodes_completed': args.num_episodes,
                'training/final_avg_reward': np.mean(agent.training_stats['episode_rewards'][-10:]) if agent.training_stats['episode_rewards'] else 0
            })
        
        # Evaluate the trained agent
        print("\nEvaluating trained agent...")
        eval_results = evaluate_agent(env, agent, num_episodes=10)
        
        # Log evaluation results to wandb
        if use_wandb:
            wandb.log({
                'evaluation/avg_reward': eval_results['avg_reward'],
                'evaluation/std_reward': eval_results['std_reward'],
                'evaluation/avg_length': eval_results['avg_length'],
                'evaluation/best_reward': max(eval_results['episode_rewards']),
                'evaluation/worst_reward': min(eval_results['episode_rewards'])
            })
        
        # Save evaluation results
        eval_path = f"logs/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(eval_path, 'w') as f:
            f.write(f"Training completed at: {end_time}\n")
            f.write(f"Training time: {training_time}\n")
            f.write(f"Final model: {final_model_path}\n")
            f.write(f"Evaluation results:\n")
            f.write(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}\n")
            f.write(f"Average Length: {eval_results['avg_length']:.2f}\n")
            
            # Add wandb link if available
            if use_wandb:
                f.write(f"\nWeights & Biases Dashboard:\n")
                f.write(f"Run Name: {wandb.run.name}\n")
                f.write(f"Project: {wandb.run.project}\n")
                f.write(f"URL: {wandb.run.url}\n")
        
        print(f"Evaluation results saved to {eval_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current model
        interrupted_model_path = f"{args.model_path}_interrupted.pth"
        agent.save_model(interrupted_model_path)
        print(f"Model saved to {interrupted_model_path}")
        
        # Save interruption info with wandb link
        if use_wandb:
            wandb.save(interrupted_model_path)
            wandb.log({'training/interrupted': True})
            
            # Save interruption log with wandb link
            interrupt_log_path = f"logs/interrupted_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(interrupt_log_path, 'w') as f:
                f.write(f"Training Interrupted\n")
                f.write(f"=" * 30 + "\n")
                f.write(f"Interrupted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model saved to: {interrupted_model_path}\n")
                f.write(f"\nWeights & Biases Dashboard:\n")
                f.write(f"Run Name: {wandb.run.name}\n")
                f.write(f"Project: {wandb.run.project}\n")
                f.write(f"URL: {wandb.run.url}\n")
            
            print(f"Interruption log saved to: {interrupt_log_path}")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error to wandb
        if use_wandb:
            wandb.log({'training/error': str(e)})
    
    finally:
        # Clean up
        env.close()
        print("Environment closed")
        
        # Finish wandb run
        if use_wandb:
            wandb.finish()
            print("Weights & Biases run finished")


if __name__ == "__main__":
    main()

