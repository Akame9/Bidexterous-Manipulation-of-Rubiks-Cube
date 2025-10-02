"""
Proximal Policy Optimization (PPO) implementation for bidexhands manipulation.
This module implements the PPO algorithm for training agents to manipulate objects
using the bidexhands simulation environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
from collections import deque
import random
import time

# Optional Weights & Biases logging
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.
    The actor network outputs action means and standard deviations.
    The critic network outputs state values.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network."""
        shared_features = self.shared_layers(state)
        
        # Actor outputs
        action_mean = self.actor_mean(shared_features)
        action_std = F.softplus(self.actor_std(shared_features)) + 1e-5  # Ensure positive std
        
        # Critic output
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from the policy."""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, value
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate(self, state, action):
        """Evaluate action given state for PPO updates."""
        action_mean, action_std, value = self.forward(state)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class PPOAgent:
    """
    Proximal Policy Optimization agent for bidexhands manipulation.
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=10, entropy_coef=0.01, 
                 value_coef=0.5, max_grad_norm=0.5, device='cpu'):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs for policy update
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory for storing experiences
        self.memory = PPOMemory()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
    
    def select_action(self, state, deterministic=False):
        """
        Select action given current state.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(state, deterministic=True)
                return action.cpu().numpy().flatten(), None, value.cpu().numpy().flatten()
            else:
                action, log_prob, value = self.policy.get_action(state, deterministic=False)
                return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory."""
        self.memory.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.memory.states) < 32:  # Reduced minimum batch size for memory efficiency
            return
        
        # Process data in smaller batches to avoid memory issues
        batch_size = min(256, len(self.memory.states))  # Limit batch size
        
        # Get all stored data
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.FloatTensor(self.memory.actions).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        next_states = torch.FloatTensor(self.memory.next_states).to(self.device)
        dones = torch.BoolTensor(self.memory.dones).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.memory.values).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update with mini-batches
        total_samples = len(states)
        indices = torch.randperm(total_samples)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.k_epochs):
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses for statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Clear cache to free memory
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Store training statistics (averaged over all mini-batches)
        if policy_losses:
            self.training_stats['policy_loss'].append(np.mean(policy_losses))
            self.training_stats['value_loss'].append(np.mean(value_losses))
            self.training_stats['entropy_loss'].append(np.mean(entropy_losses))
            total_loss_avg = np.mean(policy_losses) + self.value_coef * np.mean(value_losses) + self.entropy_coef * np.mean(entropy_losses)
            self.training_stats['total_loss'].append(total_loss_avg)
        
        # Clear memory
        self.memory.clear()
    
    def compute_gae(self, rewards, values, dones, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor
            values: Value estimates tensor
            dones: Done flags tensor
            lam: GAE lambda parameter
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = []
        returns = []
        
        # Compute next values (append 0 for terminal states)
        next_values = torch.cat([values[1:], torch.zeros(1, 1).to(self.device)])
        
        # Compute TD errors
        td_errors = rewards + self.gamma * next_values * (~dones).float() - values
        
        # Compute GAE advantages
        advantage = 0
        for i in reversed(range(len(td_errors))):
            advantage = td_errors[i] + self.gamma * lam * (~dones[i]).float() * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.cat(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return self.training_stats


class PPOMemory:
    """Memory buffer for storing PPO experiences."""
    
    def __init__(self, max_size=2000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.max_size = max_size
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # Limit memory size to prevent memory issues
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
    
    def clear(self):
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


def train_ppo_agent(env, agent, num_episodes=1000, max_steps=500, 
                   save_interval=100, model_path="ppo_model.pth"):
    """
    Train PPO agent on the environment.
    
    Args:
        env: Environment instance
        agent: PPO agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_interval: Interval for saving model
        model_path: Path to save model
    """
    print("Starting PPO training...")
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.config.update({
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'save_interval': save_interval,
            'model_path': model_path,
            'state_dim': agent.state_dim,
            'action_dim': agent.action_dim,
            'lr': agent.lr,
            'gamma': agent.gamma,
            'eps_clip': agent.eps_clip,
            'k_epochs': agent.k_epochs,
            'entropy_coef': agent.entropy_coef,
            'value_coef': agent.value_coef,
            'max_grad_norm': agent.max_grad_norm,
        }, allow_val_change=True)
    
    for episode in range(num_episodes):
        state = env.initialize()
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        for step in range(max_steps):
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

        # Log to wandb
        if _WANDB_AVAILABLE and wandb.run is not None:
            log_payload = {
                'episode': episode,
                'episode/reward': episode_reward,
                'episode/length': episode_length,
                'time/episode_s': max(time.time() - start_time, 1e-9),
            }
            for key in ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss']:
                if agent.training_stats[key]:
                    log_payload[f'loss/{key}'] = agent.training_stats[key][-1]
            if len(agent.training_stats['episode_rewards']) >= 10:
                log_payload['episode/avg_reward_10'] = float(np.mean(agent.training_stats['episode_rewards'][-10:]))
                log_payload['episode/avg_length_10'] = float(np.mean(agent.training_stats['episode_lengths'][-10:]))
            wandb.log(log_payload)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
            avg_length = np.mean(agent.training_stats['episode_lengths'][-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"{model_path}_{episode}")
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.save(f"{model_path}_{episode}")
    
    # Save final model
    agent.save_model(model_path)
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.save(model_path)
    print("Training completed!")


if __name__ == "__main__":
    import argparse
    from environment.rubiks_cube import RubiksCubeEnvironment

    parser = argparse.ArgumentParser(description="Train PPO on Rubik's Cube environment")
    parser.add_argument('--xml', type=str, default='xmls/bidexhands.xml', help='MuJoCo XML path')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--k_epochs', type=int, default=10, help='PPO update epochs')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max grad norm')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between checkpoints')
    parser.add_argument('--model_path', type=str, default='saved_models/ppo_model.pth', help='Model save path')
    parser.add_argument('--enable_viewer', action='store_true', help='Enable MuJoCo viewer')
    parser.add_argument('--visualize_collision_boxes', action='store_true', help='Visualize collision boxes')
    parser.add_argument('--project', type=str, default='mujoco-rubiks-ppo', help='wandb project name')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize wandb
    if _WANDB_AVAILABLE:
        wandb.init(project=args.project, name=args.run_name, config={
            'algorithm': 'PPO',
            'env': 'RubiksCubeEnvironment',
            'xml_path': args.xml,
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'lr': args.lr,
            'gamma': args.gamma,
            'eps_clip': args.eps_clip,
            'k_epochs': args.k_epochs,
            'entropy_coef': args.entropy_coef,
            'value_coef': args.value_coef,
            'max_grad_norm': args.max_grad_norm,
            'device': args.device,
            'seed': args.seed,
        })

    # Create environment
    env = RubiksCubeEnvironment(
        xml_path=args.xml,
        enable_viewer=args.enable_viewer,
        visualize_collision_boxes=args.visualize_collision_boxes,
    )

    # Instantiate agent
    state_dim = env.state_dim
    action_dim = env.action_dim
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
        device=args.device,
    )

    # Train
    try:
        train_ppo_agent(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            save_interval=args.save_interval,
            model_path=args.model_path,
        )
    finally:
        env.close()
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
