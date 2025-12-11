"""
Multi-Agent Proximal Policy Optimization (PPO) implementation with vectorized environments.
This module implements independent PPO agents for each hand, where each hand learns
its own policy while sharing the same state and reward signal, optimized for parallel rollouts.
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
import argparse
import sys
import math
from tqdm import tqdm
from environment.rubiks_cube import RubiksCubeEnvironment
from environment.vectorized_env import VectorizedEnv

# Import from single-agent PPO for shared components
# Try relative import first, then absolute
try:
    from .ppo import get_device, print_gpu_info, ActorCritic
    from .ppo_vec import OptimizedPPOMemory
except ImportError:
    from ppo import get_device, print_gpu_info, ActorCritic
    from ppo_vec import OptimizedPPOMemory

# Optional Weights & Biases logging
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


class MultiAgentPPOVec:
    """
    Multi-Agent PPO with vectorized environments where each hand (left and right) is an independent agent.
    Each agent has its own policy network, optimizer, and memory buffer.
    They share the same state and reward signal from the environment.
    Optimized for parallel rollouts using vectorized environments.
    """
    
    def __init__(self, state_dim, left_action_dim, right_action_dim, 
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10, 
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, 
                 device='cpu', use_mixed_precision=False, batch_size=64,
                 high_reward_threshold=1.5, clip_vloss=True, 
                 value_clip_coef=0.2, value_loss_mode='min',
                 target_kl=0.01, early_stop_kl=True,
                 use_torch_compile=True, num_envs=1, max_steps=500):
        """
        Initialize Multi-Agent PPO with vectorized environment support.
        
        Args:
            state_dim: Dimension of state space (shared by both agents)
            left_action_dim: Dimension of left hand action space
            right_action_dim: Dimension of right hand action space
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs for policy update
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ('cpu' or 'cuda')
            use_mixed_precision: Whether to use mixed precision training (GPU only)
            batch_size: Batch size for training
            high_reward_threshold: Reward threshold for high-reward experience prioritization
            clip_vloss: Whether to use clipped value loss
            value_clip_coef: Clipping coefficient for value loss
            value_loss_mode: How to combine clipped/unclipped losses ('max', 'min', 'mean')
            target_kl: Target KL divergence for early stopping
            early_stop_kl: Whether to use early stopping based on KL divergence
            use_torch_compile: Whether to use torch.compile() for faster execution
            num_envs: Number of parallel environments (for calculating memory size)
            max_steps: Maximum steps per episode (for calculating memory size)
        """
        self.state_dim = state_dim
        self.left_action_dim = left_action_dim
        self.right_action_dim = right_action_dim
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.batch_size = batch_size
        self.high_reward_threshold = high_reward_threshold
        self.clip_vloss = clip_vloss
        self.value_clip_coef = value_clip_coef
        self.value_loss_mode = value_loss_mode
        self.target_kl = target_kl
        self.early_stop_kl = early_stop_kl
        
        # Create independent agents for left and right hands
        self.left_agent = self._create_agent('left', state_dim, left_action_dim, lr, 
                                             use_torch_compile, num_envs, max_steps)
        self.right_agent = self._create_agent('right', state_dim, right_action_dim, lr,
                                              use_torch_compile, num_envs, max_steps)
        
        # Training statistics (combined and per-agent)
        self.training_stats = {
            'left_policy_loss': [],
            'left_value_loss': [],
            'left_entropy_loss': [],
            'right_policy_loss': [],
            'right_value_loss': [],
            'right_entropy_loss': [],
            'total_policy_loss': [],
            'total_value_loss': [],
            'total_entropy_loss': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'left_policy_shift': [],
            'right_policy_shift': [],
            'left_learning_rate': [],
            'right_learning_rate': [],
            'left_action_std': [],
            'right_action_std': [],
            'left_entropy': [],
            'right_entropy': [],
        }
    
    def _create_agent(self, agent_name, state_dim, action_dim, lr, 
                     use_torch_compile, num_envs, max_steps):
        """Create a single PPO agent with all the features."""
        agent = SingleHandPPOAgentVec(
            agent_name=agent_name,
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=self.gamma,
            eps_clip=self.eps_clip,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            device=self.device,
            use_mixed_precision=self.use_mixed_precision,
            batch_size=self.batch_size,
            high_reward_threshold=self.high_reward_threshold,
            clip_vloss=self.clip_vloss,
            value_clip_coef=self.value_clip_coef,
            value_loss_mode=self.value_loss_mode,
            target_kl=self.target_kl,
            early_stop_kl=self.early_stop_kl,
            use_torch_compile=use_torch_compile,
            num_envs=num_envs,
            max_steps=max_steps,
        )
        return agent
    
    def select_actions(self, state, deterministic=False):
        """
        Select actions for both hands (single state).
        
        Args:
            state: Current state (shared by both agents)
            deterministic: Whether to use deterministic policy
            
        Returns:
            left_action: Action for left hand
            right_action: Action for right hand
            left_log_prob: Log probability of left action
            right_log_prob: Log probability of right action
            left_value: Value estimate for left agent
            right_value: Value estimate for right agent
        """
        left_action, left_log_prob, left_value = self.left_agent.select_action(state, deterministic)
        right_action, right_log_prob, right_value = self.right_agent.select_action(state, deterministic)
        
        return left_action, right_action, left_log_prob, right_log_prob, left_value, right_value
    
    def select_actions_batch(self, states, deterministic=False):
        """
        Select actions for both hands (batch of states for vectorized environments).
        
        Args:
            states: Array of states with shape (batch_size, state_dim)
            deterministic: Whether to use deterministic policy
            
        Returns:
            left_actions: Array of left hand actions (batch_size, left_action_dim)
            right_actions: Array of right hand actions (batch_size, right_action_dim)
            left_log_probs: Array of left log probabilities (batch_size,)
            right_log_probs: Array of right log probabilities (batch_size,)
            left_values: Array of left values (batch_size,)
            right_values: Array of right values (batch_size,)
        """
        left_actions, left_log_probs, left_values = self.left_agent.select_actions_batch(states, deterministic)
        right_actions, right_log_probs, right_values = self.right_agent.select_actions_batch(states, deterministic)
        
        return left_actions, right_actions, left_log_probs, right_log_probs, left_values, right_values
    
    def store_transition(self, state, left_action, right_action, reward, 
                        next_state, done, left_log_prob, right_log_prob, 
                        left_value, right_value, env_id=0):
        """Store transition for both agents (they share the same reward)."""
        self.left_agent.store_transition(state, left_action, reward, next_state, done, left_log_prob, left_value, env_id)
        self.right_agent.store_transition(state, right_action, reward, next_state, done, right_log_prob, right_value, env_id)
    
    def update(self):
        """Update both agents independently."""
        # Update left hand agent
        left_stats = self.left_agent.update()
        
        # Update right hand agent
        right_stats = self.right_agent.update()
        
        # Store combined statistics
        if left_stats and right_stats:
            self.training_stats['left_policy_loss'].append(left_stats.get('policy_loss', 0.0))
            self.training_stats['left_value_loss'].append(left_stats.get('value_loss', 0.0))
            self.training_stats['left_entropy_loss'].append(left_stats.get('entropy_loss', 0.0))
            self.training_stats['right_policy_loss'].append(right_stats.get('policy_loss', 0.0))
            self.training_stats['right_value_loss'].append(right_stats.get('value_loss', 0.0))
            self.training_stats['right_entropy_loss'].append(right_stats.get('entropy_loss', 0.0))
            
            # Combined losses
            self.training_stats['total_policy_loss'].append(
                (left_stats.get('policy_loss', 0.0) + right_stats.get('policy_loss', 0.0)) / 2.0
            )
            self.training_stats['total_value_loss'].append(
                (left_stats.get('value_loss', 0.0) + right_stats.get('value_loss', 0.0)) / 2.0
            )
            self.training_stats['total_entropy_loss'].append(
                (left_stats.get('entropy_loss', 0.0) + right_stats.get('entropy_loss', 0.0)) / 2.0
            )
            
            # Policy shifts and learning rates
            if 'policy_shift' in left_stats:
                self.training_stats['left_policy_shift'].append(left_stats['policy_shift'])
            if 'policy_shift' in right_stats:
                self.training_stats['right_policy_shift'].append(right_stats['policy_shift'])
            if 'learning_rate' in left_stats:
                self.training_stats['left_learning_rate'].append(left_stats['learning_rate'])
            if 'learning_rate' in right_stats:
                self.training_stats['right_learning_rate'].append(right_stats['learning_rate'])
            if 'action_std' in left_stats:
                self.training_stats['left_action_std'].append(left_stats['action_std'])
            if 'action_std' in right_stats:
                self.training_stats['right_action_std'].append(right_stats['action_std'])
            if 'entropy' in left_stats:
                self.training_stats['left_entropy'].append(left_stats['entropy'])
            if 'entropy' in right_stats:
                self.training_stats['right_entropy'].append(right_stats['entropy'])
    
    def save_model(self, filepath):
        """Save both agent models."""
        torch.save({
            'left_policy_state_dict': self.left_agent.policy.state_dict(),
            'left_optimizer_state_dict': self.left_agent.optimizer.state_dict(),
            'right_policy_state_dict': self.right_agent.policy.state_dict(),
            'right_optimizer_state_dict': self.right_agent.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        print(f"Multi-agent model saved to {filepath}")
    
    def save_best_model(self, reward, filepath):
        """Save model only if it achieves a new best reward."""
        if reward > self.left_agent.best_reward:
            self.left_agent.best_reward = reward
            self.right_agent.best_reward = reward
            torch.save({
                'left_policy_state_dict': self.left_agent.policy.state_dict(),
                'left_optimizer_state_dict': self.left_agent.optimizer.state_dict(),
                'right_policy_state_dict': self.right_agent.policy.state_dict(),
                'right_optimizer_state_dict': self.right_agent.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'best_reward': reward
            }, filepath)
            print(f"New best multi-agent model saved! Reward: {reward:.2f}")
            return True
        return False
    
    def load_model(self, filepath):
        """Load both agent models."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.left_agent.policy.load_state_dict(checkpoint['left_policy_state_dict'])
            self.left_agent.optimizer.load_state_dict(checkpoint['left_optimizer_state_dict'])
            self.right_agent.policy.load_state_dict(checkpoint['right_policy_state_dict'])
            self.right_agent.optimizer.load_state_dict(checkpoint['right_optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"Multi-agent model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return self.training_stats


class SingleHandPPOAgentVec:
    """
    Single hand PPO agent optimized for vectorized environments (used internally by MultiAgentPPOVec).
    This is essentially the same as PPOAgentVec but simplified for multi-agent use.
    """
    
    def __init__(self, agent_name, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=10, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, device='cpu', use_mixed_precision=False,
                 batch_size=64, high_reward_threshold=1.5, clip_vloss=True,
                 value_clip_coef=0.2, value_loss_mode='min', target_kl=0.01,
                 early_stop_kl=True, use_torch_compile=True, num_envs=1, max_steps=500):
        """Initialize single hand PPO agent with vectorized environment support."""
        self.agent_name = agent_name
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
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.batch_size = batch_size
        self.high_reward_threshold = high_reward_threshold
        self.clip_vloss = clip_vloss
        self.value_clip_coef = value_clip_coef
        self.value_loss_mode = value_loss_mode
        self.target_kl = target_kl
        self.early_stop_kl = early_stop_kl
        
        # Initialize network
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        
        # Apply torch.compile() if available and requested
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.policy = torch.compile(self.policy, mode='reduce-overhead')
                print(f"[{agent_name}] Using torch.compile() for faster execution")
            except Exception as e:
                print(f"[{agent_name}] Warning: torch.compile() failed: {e}, continuing without it")
        
        safe_lr = min(lr, 1e-4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=safe_lr)
        if safe_lr != lr:
            print(f"[{agent_name}] Warning: Learning rate reduced from {lr} to {safe_lr}")
        
        # Dynamic learning rate configuration
        self.initial_lr = safe_lr
        self.current_lr = safe_lr
        self.max_lr = max(lr, safe_lr)
        self.min_lr = 1e-6
        self.target_policy_shift = 0.01
        self.policy_shift_tolerance = 0.005
        self.lr_increase_factor = 1.1
        self.lr_decrease_factor = 1.1
        self.policy_shift_history = deque(maxlen=50)
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Track best reward
        self.best_reward = float('-inf')
        
        # Calculate memory size based on num_envs * max_steps
        memory_max_size = num_envs * max_steps
        
        # Memory for storing experiences (with pre-allocated buffers)
        self.memory = OptimizedPPOMemory(
            max_size=memory_max_size,
            state_dim=state_dim,
            action_dim=action_dim,
            high_reward_threshold=high_reward_threshold
        )
    
    def select_action(self, state, deterministic=False):
        """Select action given current state (single state)."""
        if np.isnan(state).any() or np.isinf(state).any():
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(state, deterministic=True)
                return action.cpu().numpy().flatten(), None, value.cpu().numpy().item()
            else:
                action, log_prob, value, _ = self.policy.get_action(state, deterministic=False)
                return action.cpu().numpy().flatten(), log_prob.cpu().numpy().item(), value.cpu().numpy().item()
    
    def select_actions_batch(self, states, deterministic=False):
        """
        Select actions for a batch of states (for vectorized environments).
        
        Args:
            states: Array of states with shape (batch_size, state_dim)
            deterministic: Whether to use deterministic policy
            
        Returns:
            actions: Array of actions (batch_size, action_dim)
            log_probs: Array of log probabilities (batch_size,)
            values: Array of values (batch_size,)
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                actions, values = self.policy.get_action(states_tensor, deterministic=True)
                return actions.cpu().numpy(), None, values.cpu().numpy().flatten()
            else:
                actions, log_probs, values, _ = self.policy.get_action(states_tensor, deterministic=False)
                return (
                    actions.cpu().numpy(),
                    log_probs.cpu().numpy().flatten(),
                    values.cpu().numpy().flatten()
                )
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value, env_id=0):
        """Store transition in memory."""
        self.memory.store(state, action, reward, next_state, done, log_prob, value, env_id)
    
    def compute_gae_vectorized(self, rewards, values, dones, next_states=None, env_ids=None, lam=0.95):
        """
        Vectorized GAE computation that handles per-episode trajectories correctly.
        
        This correctly computes GAE for vectorized environments where transitions from
        different environments are interleaved. It groups transitions by environment
        and episode, then computes GAE separately for each episode.
        """
        # Ensure tensors are the right shape
        rewards = rewards.view(-1)
        values = values.view(-1)
        dones = dones.view(-1)
        
        if isinstance(env_ids, np.ndarray):
            env_ids = torch.from_numpy(env_ids).to(values.device)
        env_ids = env_ids.view(-1)
        
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Group transitions by environment and compute GAE per episode
        unique_env_ids = torch.unique(env_ids)
        for env_id in unique_env_ids:
            env_mask = (env_ids == env_id)
            env_indices = torch.where(env_mask)[0]
            
            if len(env_indices) == 0:
                continue
            
            env_rewards = rewards[env_indices]
            env_values = values[env_indices]
            env_dones = dones[env_indices]
            
            done_mask = env_dones
            done_positions = torch.where(done_mask)[0]
            
            # Process each episode separately
            episode_start = 0
            for done_pos in done_positions:
                # Get episode data (inclusive of done step)
                episode_local_indices = torch.arange(episode_start, done_pos + 1, device=env_indices.device)
                episode_global_indices = env_indices[episode_local_indices]
                
                episode_rewards = rewards[episode_global_indices]
                episode_values = values[episode_global_indices]
                episode_dones = dones[episode_global_indices]
                
                # Compute next values for this episode
                if len(episode_global_indices) > 0:
                    # Get next state for last transition in episode
                    episode_next_states = None
                    if next_states is not None and len(next_states) > 0:
                        last_idx = episode_global_indices[-1].item()
                        if last_idx < len(next_states):
                            episode_next_states = next_states[last_idx:last_idx+1]
                    
                    if episode_next_states is not None:
                        with torch.no_grad():
                            _, _, last_next_value = self.policy.forward(episode_next_states)
                            last_next_value = last_next_value.squeeze().unsqueeze(0).to(values.device)
                    else:
                        last_next_value = torch.zeros(1, device=values.device)
                    
                    # Compute TD errors for this episode
                    if len(episode_values) > 1:
                        episode_next_values = torch.cat([episode_values[1:], last_next_value])
                    else:
                        episode_next_values = last_next_value
                    
                    episode_td_errors = episode_rewards + self.gamma * episode_next_values * (~episode_dones).float() - episode_values
                    
                    # Compute GAE for this episode
                    episode_advantages = []
                    advantage = torch.tensor(0.0, device=episode_td_errors.device)
                    gamma_lambda = self.gamma * lam
                    for i in reversed(range(len(episode_td_errors))):
                        advantage = episode_td_errors[i] + gamma_lambda * (~episode_dones[i]).float() * advantage
                        episode_advantages.insert(0, advantage)
                    
                    episode_advantages = torch.stack(episode_advantages)
                    episode_returns = episode_advantages + episode_values
                    
                    # Store results at original indices
                    advantages[episode_global_indices] = episode_advantages
                    returns[episode_global_indices] = episode_returns
                
                episode_start = done_pos + 1
            
        returns = returns.view(-1, 1)
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.memory) < 32:
            return None
        
        batch_size = min(self.batch_size, len(self.memory))
        
        # Get all stored data
        data = self.memory.get_data()
        
        # Convert to tensors
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        rewards = torch.FloatTensor(data['rewards']).to(self.device)
        next_states = torch.FloatTensor(data['next_states']).to(self.device)
        dones = torch.BoolTensor(data['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        old_values = torch.FloatTensor(data['values']).to(self.device).view(-1, 1)
        env_ids = data.get('env_ids', None)
        
        # Compute advantages and returns (vectorized, with proper per-episode handling)
        advantages, returns = self.compute_gae_vectorized(rewards, old_values, dones, next_states, env_ids)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_samples = len(states)
        kl_values = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        action_stds = []
        entropy_values = []
        
        epochs_completed = 0
        for epoch in range(self.k_epochs):
            indices = torch.randperm(total_samples)
            epoch_kl_values = []
            
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                if self.use_mixed_precision:
                    from torch.cuda.amp import autocast
                    with autocast():
                        _, log_probs, values, entropy = self.policy.get_action(batch_states, batch_actions, deterministic=False)
                        log_probs = log_probs.squeeze(1) if log_probs.dim() > 1 and log_probs.shape[1] == 1 else log_probs
                        batch_old_log_probs = batch_old_log_probs.squeeze(1) if batch_old_log_probs.dim() > 1 and batch_old_log_probs.shape[1] == 1 else batch_old_log_probs
                        
                        log_ratio = log_probs - batch_old_log_probs
                        ratios = torch.exp(log_ratio)
                        
                        kl_div = ((ratios - 1) - log_ratio).mean()
                        approx_kl = kl_div.detach().cpu().item()
                        kl_values.append(approx_kl)
                        epoch_kl_values.append(approx_kl)
                        
                        with torch.no_grad():
                            _, action_std, _ = self.policy.forward(batch_states[0:1])
                            action_stds.append(action_std.mean().item())
                            entropy_values.append(entropy.mean().item())
                        
                        surr1 = ratios * batch_advantages
                        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        newvalue = values.squeeze(1) if values.dim() > 1 and values.shape[1] == 1 else values
                        b_returns = batch_returns.squeeze(1) if batch_returns.dim() > 1 and batch_returns.shape[1] == 1 else batch_returns
                        b_values = old_values[batch_indices].squeeze(1) if old_values[batch_indices].dim() > 1 and old_values[batch_indices].shape[1] == 1 else old_values[batch_indices]
                        
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns) ** 2
                            v_clipped = b_values + torch.clamp(
                                newvalue - b_values,
                                -self.value_clip_coef,
                                self.value_clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns) ** 2
                            
                            if self.value_loss_mode == 'max':
                                v_loss_combined = torch.max(v_loss_unclipped, v_loss_clipped)
                            elif self.value_loss_mode == 'min':
                                v_loss_combined = torch.min(v_loss_unclipped, v_loss_clipped)
                            elif self.value_loss_mode == 'mean':
                                v_loss_combined = (v_loss_unclipped + v_loss_clipped) / 2.0
                            else:
                                raise ValueError(f"Unknown value_loss_mode: {self.value_loss_mode}")
                            
                            value_loss = 0.5 * v_loss_combined.mean()
                        else:
                            value_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
                        
                        entropy_loss = -entropy.mean()
                        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                        self.optimizer.zero_grad()
                        continue
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _, log_probs, values, entropy = self.policy.get_action(batch_states, batch_actions, deterministic=False)
                    log_probs = log_probs.squeeze(1) if log_probs.dim() > 1 and log_probs.shape[1] == 1 else log_probs
                    batch_old_log_probs = batch_old_log_probs.squeeze(1) if batch_old_log_probs.dim() > 1 and batch_old_log_probs.shape[1] == 1 else batch_old_log_probs
                    
                    log_ratio = log_probs - batch_old_log_probs
                    ratios = torch.exp(log_ratio)
                    
                    kl_div = ((ratios - 1) - log_ratio).mean()
                    approx_kl = kl_div.detach().cpu().item()
                    kl_values.append(approx_kl)
                    epoch_kl_values.append(approx_kl)
                    
                    with torch.no_grad():
                        _, action_std, _ = self.policy.forward(batch_states[0:1])
                        action_stds.append(action_std.mean().item())
                        entropy_values.append(entropy.mean().item())
                    
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    newvalue = values.squeeze(1) if values.dim() > 1 and values.shape[1] == 1 else values
                    b_returns = batch_returns.squeeze(1) if batch_returns.dim() > 1 and batch_returns.shape[1] == 1 else batch_returns
                    b_values = old_values[batch_indices].squeeze(1) if old_values[batch_indices].dim() > 1 and old_values[batch_indices].shape[1] == 1 else old_values[batch_indices]
                    
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns) ** 2
                        v_clipped = b_values + torch.clamp(
                            newvalue - b_values,
                            -self.value_clip_coef,
                            self.value_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns) ** 2
                        
                        if self.value_loss_mode == 'max':
                            v_loss_combined = torch.max(v_loss_unclipped, v_loss_clipped)
                        elif self.value_loss_mode == 'min':
                            v_loss_combined = torch.min(v_loss_unclipped, v_loss_clipped)
                        elif self.value_loss_mode == 'mean':
                            v_loss_combined = (v_loss_unclipped + v_loss_clipped) / 2.0
                        else:
                            raise ValueError(f"Unknown value_loss_mode: {self.value_loss_mode}")
                        
                        value_loss = 0.5 * v_loss_combined.mean()
                    else:
                        value_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
                    
                    entropy_loss = -entropy.mean()
                    total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                        self.optimizer.zero_grad()
                        continue
                    
                    self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Early stopping
            if self.early_stop_kl and epoch_kl_values:
                mean_kl = np.mean(epoch_kl_values)
                if mean_kl > 1.5 * self.target_kl:
                    epochs_completed = epoch + 1
                    print(f"[{self.agent_name}] Early stopping at epoch {epochs_completed}/{self.k_epochs} due to high KL: {mean_kl:.6f}")
                    break
            epochs_completed = epoch + 1
        
        # Store statistics
        stats = {}
        if policy_losses:
            stats['policy_loss'] = np.mean(policy_losses)
            stats['value_loss'] = np.mean(value_losses)
            stats['entropy_loss'] = np.mean(entropy_losses)
        
        if kl_values:
            mean_policy_shift = float(np.mean(kl_values))
            stats['policy_shift'] = mean_policy_shift
            self._adjust_learning_rate(mean_policy_shift)
            stats['learning_rate'] = self.current_lr
        
        if action_stds:
            stats['action_std'] = np.mean(action_stds)
        if entropy_values:
            stats['entropy'] = np.mean(entropy_values)
        
        # Clear memory
        self.memory.clear()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return stats
    
    def _adjust_learning_rate(self, mean_policy_shift):
        """Dynamically adjust learning rate based on policy shift."""
        if np.isnan(mean_policy_shift) or np.isinf(mean_policy_shift):
            return
        
        self.policy_shift_history.append(mean_policy_shift)
        smoothed_shift = float(np.mean(self.policy_shift_history))
        
        upper_bound = self.target_policy_shift + self.policy_shift_tolerance
        lower_bound = max(0.0, self.target_policy_shift - self.policy_shift_tolerance)
        
        previous_lr = self.current_lr
        if smoothed_shift > upper_bound and self.current_lr > self.min_lr:
            self.current_lr = max(self.min_lr, self.current_lr / self.lr_decrease_factor)
        elif smoothed_shift < lower_bound and self.current_lr < self.max_lr:
            self.current_lr = min(self.max_lr, self.current_lr * self.lr_increase_factor)
        
        if self.current_lr != previous_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            print(
                f"[{self.agent_name}] Adjusted learning rate from {previous_lr:.6e} to {self.current_lr:.6e} "
                f"(smoothed policy shift: {smoothed_shift:.6f})"
            )
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr


def train_multi_agent_ppo_vec(env_fn, multi_agent, num_episodes=1000, max_steps=500,
                              save_interval=100, model_path="multi_agent_ppo_model.pth",
                              save_best_only=False, episodes_per_update=4, min_buffer_size=500,
                              num_envs=8, use_vectorized=True, save_best_after_episode=0,
                              save_final_model=True):
    """
    Train multi-agent PPO with vectorized environments for faster data collection.
    
    Args:
        env_fn: Function that creates a single environment instance
        multi_agent: MultiAgentPPOVec instance
        num_episodes: Number of training episodes (total across all envs)
        max_steps: Maximum steps per episode
        save_interval: Interval for saving model
        model_path: Path to save model
        save_best_only: If True, only save the best model
        episodes_per_update: Number of episodes to accumulate before updating
        min_buffer_size: Minimum number of transitions before updating
        num_envs: Number of parallel environments (only used if use_vectorized=True)
        use_vectorized: Whether to use vectorized environments
        save_best_after_episode: Start saving best model after this episode
        save_final_model: Whether to save final model
    """
    print("Starting Multi-Agent PPO training with vectorized environments...")
    if use_vectorized:
        print(f"Using {num_envs} parallel environments for {num_envs}x faster data collection")
    
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.config.update({
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'episodes_per_update': episodes_per_update,
            'min_buffer_size': min_buffer_size,
            'num_envs': num_envs if use_vectorized else 1,
            'use_vectorized': use_vectorized,
        }, allow_val_change=True)
    
    if use_vectorized:
        # Use vectorized environments
        vec_env = VectorizedEnv(env_fn, num_envs=num_envs)
        states = vec_env.reset()
        
        episodes_since_update = 0
        total_episodes = 0
        
        for episode in tqdm(range(num_episodes // num_envs), desc="Training Episodes"):
            # Reset all environments at the start of each epoch
            states = vec_env.reset()
            # Track which environments have already been counted for episode statistics in this epoch
            counted_for_stats = set()
            
            for step in range(max_steps):
                # Save done state BEFORE step (to know which envs were already done)
                dones_before_step = vec_env.dones.copy()
                
                # Only select actions for environments that are not done
                active_envs = ~dones_before_step
                
                # Initialize arrays for all environments
                left_actions_full = np.zeros((num_envs, multi_agent.left_action_dim), dtype=np.float32)
                right_actions_full = np.zeros((num_envs, multi_agent.right_action_dim), dtype=np.float32)
                left_log_probs_full = np.zeros(num_envs, dtype=np.float32)
                right_log_probs_full = np.zeros(num_envs, dtype=np.float32)
                left_values_full = np.zeros(num_envs, dtype=np.float32)
                right_values_full = np.zeros(num_envs, dtype=np.float32)
                
                if active_envs.any():
                    # Select actions only for active (non-done) environments
                    active_states = states[active_envs]
                    left_actions, right_actions, left_log_probs, right_log_probs, left_values, right_values = \
                        multi_agent.select_actions_batch(active_states)
                    
                    # Fill in actions, log_probs, and values for active environments
                    left_actions_full[active_envs] = left_actions
                    right_actions_full[active_envs] = right_actions
                    left_log_probs_full[active_envs] = left_log_probs
                    right_log_probs_full[active_envs] = right_log_probs
                    left_values_full[active_envs] = left_values
                    right_values_full[active_envs] = right_values
                
                # Combine actions for environment
                combined_actions = np.concatenate([left_actions_full, right_actions_full], axis=1)
                
                # Step all environments in parallel (vectorized_env will skip done ones)
                next_states, rewards, dones, infos = vec_env.step(combined_actions)
                
                # Store transitions for all environments
                # Only store if environment was not already done before this step
                for i in range(num_envs):
                    # Skip if environment was already done before this step
                    if not dones_before_step[i]:
                        multi_agent.store_transition(
                            states[i],
                            left_actions_full[i],
                            right_actions_full[i],
                            rewards[i],
                            next_states[i],
                            dones[i],
                            left_log_probs_full[i],
                            right_log_probs_full[i],
                            left_values_full[i],
                            right_values_full[i],
                            env_id=i  # Track which environment this transition belongs to
                        )
                
                states = next_states
                
                # Track episode statistics when environments finish (but don't reset yet)
                if dones.any():
                    done_indices = np.where(dones)[0].tolist()
                    # Filter to only newly done environments (not already counted)
                    newly_done_indices = [i for i in done_indices if i not in counted_for_stats]
                    
                    if newly_done_indices:
                        # Store episode statistics only for newly done environments
                        stats = vec_env.get_episode_stats()
                        for i in newly_done_indices:
                            multi_agent.training_stats['episode_rewards'].append(stats['rewards'][i])
                            multi_agent.training_stats['episode_lengths'].append(stats['lengths'][i])
                            counted_for_stats.add(i)  # Mark as counted
                        episodes_since_update += len(newly_done_indices)
                        total_episodes += len(newly_done_indices)
                
                # Early break if all done (will reset all at start of next epoch)
                if dones.all():
                    break
            
            # Update policy when enough data collected
            left_buffer_size = len(multi_agent.left_agent.memory)
            right_buffer_size = len(multi_agent.right_agent.memory)
            buffer_size = min(left_buffer_size, right_buffer_size)
            
            should_update = (
                episodes_since_update >= episodes_per_update and
                left_buffer_size >= min_buffer_size and
                right_buffer_size >= min_buffer_size
            )
            
            if should_update and left_buffer_size >= 32 and right_buffer_size >= 32:
                multi_agent.update()
                episodes_since_update = 0
            elif left_buffer_size >= multi_agent.left_agent.memory.max_size or \
                 right_buffer_size >= multi_agent.right_agent.memory.max_size:
                multi_agent.update()
                episodes_since_update = 0
            
            # Logging - log the last num_envs episodes' rewards, lengths, and losses
            if _WANDB_AVAILABLE and wandb.run is not None:
                # Log the last num_envs episodes
                num_episodes_to_log = min(num_envs, len(multi_agent.training_stats['episode_rewards']))
                if num_episodes_to_log > 0:
                    # Get the last num_envs episodes
                    start_idx = len(multi_agent.training_stats['episode_rewards']) - num_episodes_to_log
                    
                    for i in range(num_episodes_to_log):
                        idx = start_idx + i
                        episode_num = total_episodes - num_episodes_to_log + i + 1  # Episode number (1-indexed)
                        log_payload = {
                            'episode': episode_num,
                            'episode/reward': multi_agent.training_stats['episode_rewards'][idx],
                            'episode/length': multi_agent.training_stats['episode_lengths'][idx],
                        }
                        
                        # Add loss metrics - log the last num_envs loss values
                        for key in ['left_policy_loss', 'left_value_loss', 'left_entropy_loss',
                                   'right_policy_loss', 'right_value_loss', 'right_entropy_loss',
                                   'total_policy_loss', 'total_value_loss', 'total_entropy_loss']:
                            if multi_agent.training_stats[key]:
                                num_losses = len(multi_agent.training_stats[key])
                                if num_losses > 0:
                                    loss_start_idx = max(0, num_losses - num_episodes_to_log)
                                    loss_idx = min(loss_start_idx + i, num_losses - 1)
                                    log_payload[f'loss/{key}'] = multi_agent.training_stats[key][loss_idx]
                        
                        # Add learning rates
                        for key in ['left_learning_rate', 'right_learning_rate']:
                            if multi_agent.training_stats[key]:
                                num_lrs = len(multi_agent.training_stats[key])
                                if num_lrs > 0:
                                    lr_start_idx = max(0, num_lrs - num_episodes_to_log)
                                    lr_idx = min(lr_start_idx + i, num_lrs - 1)
                                    log_payload[f'train/{key}'] = multi_agent.training_stats[key][lr_idx]
                        
                        # Add exploration metrics
                        for key in ['left_action_std', 'right_action_std', 'left_entropy', 'right_entropy']:
                            if multi_agent.training_stats[key]:
                                num_vals = len(multi_agent.training_stats[key])
                                if num_vals > 0:
                                    val_start_idx = max(0, num_vals - num_episodes_to_log)
                                    val_idx = min(val_start_idx + i, num_vals - 1)
                                    log_payload[f'exploration/{key}'] = multi_agent.training_stats[key][val_idx]
                        
                        # Add running averages
                        if idx >= 0:
                            avg_start_idx = max(0, idx - 9)
                            avg_end_idx = idx + 1
                            if avg_end_idx > avg_start_idx:
                                log_payload['episode/avg_reward_10'] = float(np.mean(multi_agent.training_stats['episode_rewards'][avg_start_idx:avg_end_idx]))
                                log_payload['episode/avg_length_10'] = float(np.mean(multi_agent.training_stats['episode_lengths'][avg_start_idx:avg_end_idx]))
                        
                        wandb.log(log_payload)
            
            # Print progress
            if total_episodes % 10 == 0 and multi_agent.training_stats['episode_rewards']:
                avg_reward = np.mean(multi_agent.training_stats['episode_rewards'][-10:])
                avg_length = np.mean(multi_agent.training_stats['episode_lengths'][-10:]) if len(multi_agent.training_stats['episode_lengths']) >= 10 else 0.0
                print(f"Episode {total_episodes}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, "
                      f"Left Buffer: {left_buffer_size}, Right Buffer: {right_buffer_size}")
            
            # Save model
            if save_best_only and total_episodes >= save_best_after_episode:
                if multi_agent.training_stats['episode_rewards']:
                    # Use average reward (last 10 episodes) for best model saving
                    if len(multi_agent.training_stats['episode_rewards']) >= 10:
                        avg_reward = np.mean(multi_agent.training_stats['episode_rewards'][-10:])
                        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
                        best_model_path = f"{model_name}_best.pth"
                        multi_agent.save_best_model(avg_reward, best_model_path)
                    else:
                        # If less than 10 episodes, use current episode reward
                        episode_reward = multi_agent.training_stats['episode_rewards'][-1]
                        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
                        best_model_path = f"{model_name}_best.pth"
                        multi_agent.save_best_model(episode_reward, best_model_path)
        
        vec_env.close()
    else:
        # Fallback to single environment (original behavior)
        env = env_fn()
        episodes_since_update = 0
        
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            state = env.initialize()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select actions for both hands
                left_action, right_action, left_log_prob, right_log_prob, left_value, right_value = \
                    multi_agent.select_actions(state)
                
                # Combine actions (left hand actions + right hand actions)
                combined_action = np.concatenate([left_action, right_action])
                
                # Take step in environment
                next_state, reward, done, info = env.take_step(combined_action)
                
                # Store transition for both agents (they share the same reward)
                multi_agent.store_transition(
                    state, left_action, right_action, reward, next_state, done,
                    left_log_prob, right_log_prob, left_value, right_value
                )
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episodes_since_update += 1
            
            # Store episode statistics
            multi_agent.training_stats['episode_rewards'].append(episode_reward)
            multi_agent.training_stats['episode_lengths'].append(episode_length)
            
            # Update policy only when we have accumulated enough data
            should_update = (
                episodes_since_update >= episodes_per_update and
                len(multi_agent.left_agent.memory.states) >= min_buffer_size and
                len(multi_agent.right_agent.memory.states) >= min_buffer_size
            )
            
            if should_update and (len(multi_agent.left_agent.memory.states) >= 32 and
                                  len(multi_agent.right_agent.memory.states) >= 32):
                multi_agent.update()
                episodes_since_update = 0
            elif (len(multi_agent.left_agent.memory.states) >= multi_agent.left_agent.memory.max_size or
                  len(multi_agent.right_agent.memory.states) >= multi_agent.right_agent.memory.max_size):
                multi_agent.update()
                episodes_since_update = 0
        
        env.close()
    
    # Final update on remaining data
    left_buffer_size = len(multi_agent.left_agent.memory)
    right_buffer_size = len(multi_agent.right_agent.memory)
    if left_buffer_size >= 32 and right_buffer_size >= 32:
        print(f"Final update on remaining data (Left: {left_buffer_size}, "
              f"Right: {right_buffer_size} transitions)")
        multi_agent.update()
    
    # Save final model
    if save_final_model:
        final_models_dir = "saved_final_models"
        os.makedirs(final_models_dir, exist_ok=True)
        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
        model_basename = os.path.basename(model_name)
        final_model_path = os.path.join(final_models_dir, f"{model_basename}_final.pth")
        multi_agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    print("Multi-agent training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Agent PPO with vectorized environments on Rubik's Cube")
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
    parser.add_argument('--no-clip-vloss', '--no_clip_vloss', action='store_false', dest='clip_vloss', help='Disable clipped value loss')
    parser.add_argument('--value_clip_coef', type=float, default=0.2, help='Value clipping coefficient')
    parser.add_argument('--value_loss_mode', type=str, default='min', choices=['max', 'min', 'mean'], help='Value loss mode')
    parser.add_argument('--target_kl', type=float, default=0.01, help='Target KL divergence for early stopping')
    parser.add_argument('--early_stop_kl', action='store_true', default=False, help='Enable early stopping based on KL divergence')
    parser.add_argument('--no_early_stop_kl', dest='early_stop_kl', action='store_false', help='Disable early stopping')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda, etc.)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between checkpoints')
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--use_vectorized', action='store_true', default=True, help='Use vectorized environments')
    parser.add_argument('--use_torch_compile', action='store_true', default=True, help='Use torch.compile()')
    parser.add_argument('--model_path', type=str, default='saved_models/multi_agent_ppo_vec_model.pth', help='Model save path')
    parser.add_argument('--save_best_only', action='store_true', help='Save only the best model')
    parser.add_argument('--enable_viewer', action='store_true', help='Enable MuJoCo viewer')
    parser.add_argument('--visualize_collision_boxes', action='store_true', help='Visualize collision boxes')
    parser.add_argument('--rotation_sequence', type=str, nargs='*', default=None,
                        help='Rotation sequence of face names (e.g., --rotation_sequence blue)')
    parser.add_argument('--enable_gravity', dest='enable_gravity', action='store_true', help='Override environment gravity')
    parser.add_argument('--disable_gravity', dest='enable_gravity', action='store_false', help='Disable gravity override')
    parser.set_defaults(enable_gravity=False)
    parser.add_argument('--high_reward_threshold', type=float, default=1.5, help='Reward threshold for high-reward prioritization')
    parser.add_argument('--episodes_per_update', type=int, default=4, help='Number of episodes to accumulate before updating')
    parser.add_argument('--min_buffer_size', type=int, default=500, help='Minimum number of transitions before updating')
    parser.add_argument('--save_best_after_episode', type=int, default=0, help='Start saving best model after this episode')
    parser.add_argument('--save_final_model', action='store_true', default=True, help='Save final model after training')
    parser.add_argument('--project', type=str, default='mujoco-rubiks-multi-agent-ppo-vec', help='wandb project name')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_model', type=str, default=None, help='Path to model checkpoint to load before training')
    args = parser.parse_args()
    
    if not hasattr(args, 'clip_vloss') or args.clip_vloss is None:
        args.clip_vloss = True
    
    # Determine device
    device = get_device(args.device, args.gpu_id)
    print_gpu_info(device)
    
    # Seeding
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(device)
    
    # Initialize wandb
    if _WANDB_AVAILABLE:
        wandb.init(project=args.project, name=args.run_name, config={
            'algorithm': 'Multi-Agent PPO-Vec',
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
            'device': str(device),
            'seed': args.seed,
            'gravity_override': args.enable_gravity,
            'num_envs': args.num_envs if args.use_vectorized else 1,
            'use_vectorized': args.use_vectorized,
        })
    
    # Create environment factory
    rotation_sequence = args.rotation_sequence if args.rotation_sequence else None
    if rotation_sequence:
        print(f"Rotation sequence: {rotation_sequence}")

    # Create environment factory
    def make_env():
        return RubiksCubeEnvironment(
            xml_path=args.xml,
            enable_viewer=args.enable_viewer,
            visualize_collision_boxes=args.visualize_collision_boxes,
            enable_gravity=args.enable_gravity,
            rotation_sequence=rotation_sequence,
            max_episode_steps=args.max_steps
        )
    
    # Get dimensions from a temporary environment
    temp_env = make_env()
    state_dim = temp_env.state_dim
    left_action_dim = len(temp_env.left_hand_actuators)
    right_action_dim = len(temp_env.right_hand_actuators)
    temp_env.close()
    
    print(f"State dimension: {state_dim}")
    print(f"Left hand action dimension: {left_action_dim}")
    print(f"Right hand action dimension: {right_action_dim}")
    
    # Create multi-agent PPO
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
        use_mixed_precision=args.use_mixed_precision,
        batch_size=args.batch_size,
        high_reward_threshold=args.high_reward_threshold,
        clip_vloss=args.clip_vloss,
        value_clip_coef=args.value_clip_coef,
        value_loss_mode=args.value_loss_mode,
        target_kl=args.target_kl,
        early_stop_kl=args.early_stop_kl,
        use_torch_compile=args.use_torch_compile,
        num_envs=args.num_envs if args.use_vectorized else 1,
        max_steps=args.max_steps
    )
    
    # Load model if specified
    if args.load_model:
        multi_agent.load_model(args.load_model)
    
    # Train
    try:
        train_multi_agent_ppo_vec(
            env_fn=make_env,
            multi_agent=multi_agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            save_interval=args.save_interval,
            model_path=args.model_path,
            save_best_only=args.save_best_only,
            episodes_per_update=args.episodes_per_update,
            min_buffer_size=args.min_buffer_size,
            num_envs=args.num_envs,
            use_vectorized=args.use_vectorized,
            save_best_after_episode=args.save_best_after_episode,
            save_final_model=args.save_final_model,
        )
    finally:
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

