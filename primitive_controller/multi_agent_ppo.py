"""
Multi-Agent Proximal Policy Optimization (PPO) implementation for bidexhands manipulation.
This module implements independent PPO agents for each hand, where each hand learns
its own policy while sharing the same state and reward signal.
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

# Import from single-agent PPO for shared components
# Try relative import first, then absolute
try:
    from .ppo import get_device, print_gpu_info, ActorCritic, PPOMemory
except ImportError:
    from ppo import get_device, print_gpu_info, ActorCritic, PPOMemory

# Optional Weights & Biases logging
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


class MultiAgentPPO:
    """
    Multi-Agent PPO where each hand (left and right) is an independent agent.
    Each agent has its own policy network, optimizer, and memory buffer.
    They share the same state and reward signal from the environment.
    """
    
    def __init__(self, state_dim, left_action_dim, right_action_dim, 
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10, 
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, 
                 device='cpu', use_mixed_precision=False, batch_size=64,
                 high_reward_threshold=1.5, clip_vloss=True, 
                 value_clip_coef=0.2, value_loss_mode='min',
                 target_kl=0.01, early_stop_kl=True):
        """
        Initialize Multi-Agent PPO.
        
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
        self.left_agent = self._create_agent('left', state_dim, left_action_dim, lr)
        self.right_agent = self._create_agent('right', state_dim, right_action_dim, lr)
        
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
    
    def _create_agent(self, agent_name, state_dim, action_dim, lr):
        """Create a single PPO agent with all the features."""
        agent = SingleHandPPOAgent(
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
        )
        return agent
    
    def select_actions(self, state, deterministic=False):
        """
        Select actions for both hands.
        
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
    
    def store_transition(self, state, left_action, right_action, reward, 
                        next_state, done, left_log_prob, right_log_prob, 
                        left_value, right_value):
        """Store transition for both agents (they share the same reward)."""
        self.left_agent.store_transition(state, left_action, reward, next_state, done, left_log_prob, left_value)
        self.right_agent.store_transition(state, right_action, reward, next_state, done, right_log_prob, right_value)
    
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
            checkpoint = torch.load(filepath, map_location=self.device)
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


class SingleHandPPOAgent:
    """
    Single hand PPO agent (used internally by MultiAgentPPO).
    This is essentially the same as PPOAgent but simplified for multi-agent use.
    """
    
    def __init__(self, agent_name, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=10, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, device='cpu', use_mixed_precision=False,
                 batch_size=64, high_reward_threshold=1.5, clip_vloss=True,
                 value_clip_coef=0.2, value_loss_mode='min', target_kl=0.01,
                 early_stop_kl=True):
        """Initialize single hand PPO agent."""
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
        
        # Memory
        self.memory = PPOMemory(high_reward_threshold=self.high_reward_threshold)
    
    def select_action(self, state, deterministic=False):
        """Select action given current state."""
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
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory."""
        self.memory.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.memory.states) < 32:
            return None
        
        batch_size = min(self.batch_size, len(self.memory.states))
        
        # Get all stored data
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.FloatTensor(self.memory.actions).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        next_states = torch.FloatTensor(self.memory.next_states).to(self.device)
        dones = torch.BoolTensor(self.memory.dones).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.memory.values).to(self.device).view(-1, 1)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_states)
        
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
                    
                    # AATHIRA : ratios? Why is it used here?
                    kl_div = ((ratios - 1) - log_ratio).mean()
                    approx_kl = kl_div.detach().cpu().item()
                    kl_values.append(approx_kl)
                    epoch_kl_values.append(approx_kl)
                    
                    # AATHIRA : Whaat is this?
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
    
    def compute_gae(self, rewards, values, dones, next_states=None, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        rewards = rewards.view(-1)
        values = values.view(-1)
        dones = dones.view(-1)
        
        if next_states is not None and len(next_states) > 0:
            with torch.no_grad():
                _, _, last_next_value = self.policy.forward(next_states[-1:])
                last_next_value = last_next_value.squeeze().unsqueeze(0).to(values.device)
        else:
            last_next_value = torch.zeros(1, device=values.device)
        
        next_values = torch.cat([values[1:], last_next_value])
        td_errors = rewards + self.gamma * next_values * (~dones).float() - values
        
        advantages = []
        advantage = 0
        for i in reversed(range(len(td_errors))):
            advantage = td_errors[i] + self.gamma * lam * (~dones[i]).float() * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.stack(advantages)
        returns = advantages + values
        returns = returns.view(-1, 1)
        
        return advantages, returns
    
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


def train_multi_agent_ppo(env, multi_agent, num_episodes=1000, max_steps=500,
                          save_interval=100, model_path="multi_agent_ppo_model.pth",
                          save_best_only=False, episodes_per_update=4, min_buffer_size=500):
    """
    Train multi-agent PPO on the environment.
    
    Args:
        env: Environment instance
        multi_agent: MultiAgentPPO instance
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_interval: Interval for saving model
        model_path: Path to save model
        save_best_only: If True, only save the best model
        episodes_per_update: Number of episodes to accumulate before updating
        min_buffer_size: Minimum number of transitions before updating
    """
    print("Starting Multi-Agent PPO training...")
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.config.update({
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'episodes_per_update': episodes_per_update,
            'min_buffer_size': min_buffer_size,
        }, allow_val_change=True)
    
    episodes_since_update = 0
    
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.initialize()
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
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
        
        # Log to wandb
        if _WANDB_AVAILABLE and wandb.run is not None:
            log_payload = {
                'episode': episode,
                'episode/reward': episode_reward,
                'episode/length': episode_length,
                'time/episode_s': max(time.time() - start_time, 1e-9),
            }
            
            # Log per-agent losses
            if multi_agent.training_stats['left_policy_loss']:
                log_payload['loss/left_policy_loss'] = multi_agent.training_stats['left_policy_loss'][-1]
                log_payload['loss/left_value_loss'] = multi_agent.training_stats['left_value_loss'][-1]
                log_payload['loss/left_entropy_loss'] = multi_agent.training_stats['left_entropy_loss'][-1]
            if multi_agent.training_stats['right_policy_loss']:
                log_payload['loss/right_policy_loss'] = multi_agent.training_stats['right_policy_loss'][-1]
                log_payload['loss/right_value_loss'] = multi_agent.training_stats['right_value_loss'][-1]
                log_payload['loss/right_entropy_loss'] = multi_agent.training_stats['right_entropy_loss'][-1]
            if multi_agent.training_stats['total_policy_loss']:
                log_payload['loss/total_policy_loss'] = multi_agent.training_stats['total_policy_loss'][-1]
                log_payload['loss/total_value_loss'] = multi_agent.training_stats['total_value_loss'][-1]
                log_payload['loss/total_entropy_loss'] = multi_agent.training_stats['total_entropy_loss'][-1]
            
            # Log exploration metrics
            if multi_agent.training_stats['left_action_std']:
                log_payload['exploration/left_action_std'] = multi_agent.training_stats['left_action_std'][-1]
                log_payload['exploration/left_entropy'] = multi_agent.training_stats['left_entropy'][-1]
            if multi_agent.training_stats['right_action_std']:
                log_payload['exploration/right_action_std'] = multi_agent.training_stats['right_action_std'][-1]
                log_payload['exploration/right_entropy'] = multi_agent.training_stats['right_entropy'][-1]
            
            # Log learning rates
            if multi_agent.training_stats['left_learning_rate']:
                log_payload['training/left_learning_rate'] = multi_agent.training_stats['left_learning_rate'][-1]
            if multi_agent.training_stats['right_learning_rate']:
                log_payload['training/right_learning_rate'] = multi_agent.training_stats['right_learning_rate'][-1]
            
            if len(multi_agent.training_stats['episode_rewards']) >= 10:
                log_payload['episode/avg_reward_10'] = float(np.mean(multi_agent.training_stats['episode_rewards'][-10:]))
                log_payload['episode/avg_length_10'] = float(np.mean(multi_agent.training_stats['episode_lengths'][-10:]))
            
            wandb.log(log_payload)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(multi_agent.training_stats['episode_rewards'][-10:])
            avg_length = np.mean(multi_agent.training_stats['episode_lengths'][-10:])
            left_buffer = len(multi_agent.left_agent.memory.states)
            right_buffer = len(multi_agent.right_agent.memory.states)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, "
                  f"Left Buffer: {left_buffer}, Right Buffer: {right_buffer}, "
                  f"Episodes Since Update: {episodes_since_update}")
        
        if save_best_only:
            model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
            best_model_path = f"{model_name}_best.pth"
            best_model_saved = multi_agent.save_best_model(avg_reward, best_model_path)
            if best_model_saved and _WANDB_AVAILABLE and wandb.run is not None:
                wandb.save(best_model_path)
        else:
            if episode % save_interval == 0:
                multi_agent.save_model(f"{model_path}_{episode}")
                if _WANDB_AVAILABLE and wandb.run is not None:
                    wandb.save(f"{model_path}_{episode}")
    
    # Final update on remaining data
    if (len(multi_agent.left_agent.memory.states) >= 32 and
        len(multi_agent.right_agent.memory.states) >= 32):
        print(f"Final update on remaining data (Left: {len(multi_agent.left_agent.memory.states)}, "
              f"Right: {len(multi_agent.right_agent.memory.states)} transitions)")
        multi_agent.update()
    
    print("Multi-agent training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Agent PPO on Rubik's Cube environment")
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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between checkpoints')
    parser.add_argument('--model_path', type=str, default='saved_models/multi_agent_ppo_model.pth', help='Model save path')
    parser.add_argument('--save_best_only', action='store_true', help='Save only the best model')
    parser.add_argument('--enable_viewer', action='store_true', help='Enable MuJoCo viewer')
    parser.add_argument('--visualize_collision_boxes', action='store_true', help='Visualize collision boxes')
    parser.add_argument('--enable_gravity', dest='enable_gravity', action='store_true', help='Override environment gravity')
    parser.add_argument('--disable_gravity', dest='enable_gravity', action='store_false', help='Disable gravity override')
    parser.set_defaults(enable_gravity=False)
    parser.add_argument('--high_reward_threshold', type=float, default=1.5, help='Reward threshold for high-reward prioritization')
    parser.add_argument('--episodes_per_update', type=int, default=4, help='Number of episodes to accumulate before updating')
    parser.add_argument('--min_buffer_size', type=int, default=500, help='Minimum number of transitions before updating')
    parser.add_argument('--project', type=str, default='mujoco-rubiks-multi-agent-ppo', help='wandb project name')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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
            'algorithm': 'Multi-Agent PPO',
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
        })
    
    # Create environment
    env = RubiksCubeEnvironment(
        xml_path=args.xml,
        enable_viewer=args.enable_viewer,
        visualize_collision_boxes=args.visualize_collision_boxes,
        enable_gravity=args.enable_gravity,
    )
    
    # Get state and action dimensions
    state_dim = env.state_dim
    left_action_dim = len(env.left_hand_actuators)
    right_action_dim = len(env.right_hand_actuators)
    
    print(f"State dimension: {state_dim}")
    print(f"Left hand action dimension: {left_action_dim}")
    print(f"Right hand action dimension: {right_action_dim}")
    
    # Create multi-agent PPO
    multi_agent = MultiAgentPPO(
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
    )
    
    # Train
    try:
        train_multi_agent_ppo(
            env=env,
            multi_agent=multi_agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            save_interval=args.save_interval,
            model_path=args.model_path,
            save_best_only=args.save_best_only,
            episodes_per_update=args.episodes_per_update,
            min_buffer_size=args.min_buffer_size,
        )
    finally:
        env.close()
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

