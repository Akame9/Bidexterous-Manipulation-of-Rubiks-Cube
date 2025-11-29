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
import argparse
import sys
from tqdm import tqdm
from environment.rubiks_cube import RubiksCubeEnvironment
# Optional Weights & Biases logging
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


def get_device(device_arg='auto', gpu_id=0):
    """
    Automatically determine the best device for training.
    
    Args:
        device_arg: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        gpu_id: GPU ID to use if multiple GPUs available
        
    Returns:
        torch.device: The device to use for training
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            # Check if specific GPU is available
            if gpu_id < torch.cuda.device_count():
                device = f'cuda:{gpu_id}'
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                device = 'cuda:0'
                print(f"GPU {gpu_id} not available, using GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    else:
        device = device_arg
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                device = 'cpu'
            else:
                print(f"Using specified device: {device}")
        else:
            print(f"Using specified device: {device}")
    
    return torch.device(device)


def print_gpu_info(device):
    """Print GPU information if using CUDA."""
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("Running on CPU")


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

        # DONE AATHIRA : Make std a global learnable parameter. The std should be same for all actions.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim)) #nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization with smaller scale."""
        if isinstance(module, nn.Linear):
            # Use smaller initialization scale to prevent gradient explosion
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network."""
        shared_features = self.shared_layers(state)
        
        # Actor outputs
        action_mean = self.actor_mean(shared_features)
        # action_std = F.softplus(self.actor_std(shared_features)) + 1e-5  # Ensure positive std
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # Critic output
        value = self.critic(shared_features)
        
        # AATHIRA : The network should not produce Nan values.
        # Check for NaN values and replace with zeros if found
        if torch.isnan(action_mean).any():
            print("Warning: NaN detected in action_mean, replacing with zeros")
            action_mean = torch.zeros_like(action_mean)
        
        if torch.isnan(action_std).any():
            print("Warning: NaN detected in action_std, replacing with small positive values")
            action_std = torch.ones_like(action_std) * 1e-5
        
        if torch.isnan(value).any():
            print("Warning: NaN detected in value, replacing with zeros")
            value = torch.zeros_like(value)
        
        return action_mean, action_std, value
    
    def get_action(self, state, action=None, deterministic=False):
        """Sample action from the policy."""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, value
        
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, entropy
    
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
                 value_coef=0.5, max_grad_norm=0.5, device='cpu', 
                 use_mixed_precision=False, batch_size=64, 
                 high_reward_threshold=1.5, clip_vloss=True, value_clip_coef=0.2, value_loss_mode='min'):
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
            use_mixed_precision: Whether to use mixed precision training (GPU only)
            batch_size: Batch size for training
            high_reward_threshold: Reward threshold for high-reward experience prioritization
            clip_vloss: Whether to use clipped value loss (prevents value function from updating too drastically)
            value_clip_coef: Clipping coefficient for value loss (similar to eps_clip for policy)
            value_loss_mode: How to combine clipped/unclipped losses ('max'=conservative, 'min'=prefer clipping, 'mean'=average)
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
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.batch_size = batch_size
        self.high_reward_threshold = high_reward_threshold
        self.clip_vloss = clip_vloss
        self.value_clip_coef = value_clip_coef
        self.value_loss_mode = value_loss_mode  # 'max', 'min', or 'mean'
        
        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        safe_lr = min(lr, 1e-4)  # Cap learning rate at 1e-4
        # AATHIRA : Use eps=1e-5 for Adam optimizer to reduce gradient variance..
        self.optimizer = optim.Adam(self.policy.parameters(), lr=safe_lr) #eps=1e-5, weight_decay=1e-5
        if safe_lr != lr:
            print(f"Warning: Learning rate reduced from {lr} to {safe_lr} to prevent instability")
        
        # Dynamic learning rate configuration
        self.initial_lr = safe_lr
        self.current_lr = safe_lr
        self.max_lr = max(lr, safe_lr)
        # Set minimum learning rate to prevent learning from stopping
        # 1e-6 is about the lowest practical learning rate for neural networks
        # Going lower (e.g., 1e-8) causes updates to be too small, making entropy constant
        # When LR is too small, gradient updates become negligible, so policy (and entropy) don't change
        self.min_lr = 1e-5
        self.target_policy_shift = 0.01  # Desired average policy shift (approximate KL)
        self.policy_shift_tolerance = 0.005  # Allowable deviation from target before adjustment
        self.lr_increase_factor = 1.1
        self.lr_decrease_factor = 1.1
        self.policy_shift_history = deque(maxlen=50)
        
        # Initialize mixed precision scaler if using GPU
        if self.use_mixed_precision:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("Using mixed precision training")
        else:
            self.scaler = None
            
        # Data-driven entropy cap (EMA over batches)
        self.entropy_ema_beta = 0.95
        self.entropy_ema_mean = 0.0
        self.entropy_ema_sq = 1.0
        self.entropy_cap_k = 2.0  # cap at mean + k*std

        # Track best reward for saving best model
        self.best_reward = float('-inf')
        
        # Track high-reward models (reward > 310) - keep last 10
        self.high_reward_models = []  # List of (episode, reward, filepath) tuples
        self.high_reward_threshold = 310.0
        self.max_high_reward_models = 10
        
        # Track consecutive NaN occurrences
        self.nan_count = 0
        self.max_nan_count = 5  # Reset model after 5 consecutive NaN occurrences
        
        # Memory for storing experiences
        self.memory = PPOMemory(high_reward_threshold=self.high_reward_threshold)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_shift': [],
            'learning_rate': []
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
        # AATHIRA : The network should not produce Nan values.
        # Validate input state
        if np.isnan(state).any() or np.isinf(state).any():
            print("Warning: NaN/Inf detected in input state, replacing with zeros")
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
        """Update policy using PPO algorithm with high-reward experience prioritization."""
        if len(self.memory.states) < 32:  # Reduced minimum batch size for memory efficiency
            return
        
        # Use configured batch size or limit based on memory
        batch_size = min(self.batch_size, len(self.memory.states))
        
        # Get all stored data
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.FloatTensor(self.memory.actions).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        next_states = torch.FloatTensor(self.memory.next_states).to(self.device)
        dones = torch.BoolTensor(self.memory.dones).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.memory.values).to(self.device).view(-1, 1)
        
        # Get high-reward experiences for additional training
        high_reward_data = self.memory.get_high_reward_experiences(num_samples=min(16, len(self.memory.high_reward_indices)))
        
        # Can you print the shapes of the tensors?
        # print(f"states shape: {states.shape}")
        # print(f"actions shape: {actions.shape}")
        # print(f"rewards shape: {rewards.shape}")
        # print(f"next_states shape: {next_states.shape}")
        # print(f"dones shape: {dones.shape}")
        # print(f"old_log_probs shape: {old_log_probs.shape}")
        # print(f"old_values shape: {old_values.shape}")
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_states)
        
        # Analyze data ranges for better clipping
        print(f"Data Analysis - Rewards: [{rewards.min().item():.3f}, {rewards.max().item():.3f}] (mean: {rewards.mean().item():.3f})")
        print(f"Data Analysis - Returns: [{returns.min().item():.3f}, {returns.max().item():.3f}] (mean: {returns.mean().item():.3f})")
        print(f"Data Analysis - Old Values: [{old_values.min().item():.3f}, {old_values.max().item():.3f}] (mean: {old_values.mean().item():.3f})")
        
        # Analyze reward distribution
        reward_2_count = (rewards == 2.0).sum().item()
        reward_1_count = (rewards == 1.0).sum().item()
        print(f"Data Analysis - Reward 2.0: {reward_2_count} times, Reward 1.0: {reward_1_count} times")
        
        # DONE AATHIRA : Don't clip returns. It should be free to learn.
        # Dynamic clipping based on data percentiles
        # returns_std = returns.std().item()
        # returns_mean = returns.mean().item()
        # returns_clip = max(abs(returns_mean) + 3 * returns_std, 10.0)  # 3-sigma rule, min 10
        # returns = torch.clamp(returns, min=-returns_clip, max=returns_clip)
        # print(f"Data Analysis - Returns clipped to: [{-returns_clip:.3f}, {returns_clip:.3f}]")
        
        # Debug: Print shapes after GAE computation
        # print(f"After GAE computation:")
        # print(f"  advantages shape: {advantages.shape}")
        # print(f"  returns shape: {returns.shape}")
        # print(f"  old_values shape: {old_values.shape}")
        
        # Normalize advantages with dynamic clipping
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # DONE AATHIRA : Don't clip advantages. It should be free to learn.
        # Dynamic advantage clipping based on normalized distribution
        # adv_std = advantages.std().item()
        # adv_clip = max(3.0, adv_std * 2)  # 2-sigma rule, min 3
        # advantages = torch.clamp(advantages, min=-adv_clip, max=adv_clip)
        # print(f"Data Analysis - Advantages clipped to: [{-adv_clip:.3f}, {adv_clip:.3f}]")
        
        # Process high-reward experiences if available
        high_reward_states = None
        high_reward_actions = None
        high_reward_old_log_probs = None
        high_reward_advantages = None
        high_reward_returns = None
        
        # if high_reward_data is not None and len(high_reward_data['states']) > 0:
        #     # Convert high-reward data to tensors
        #     high_reward_states = torch.FloatTensor(high_reward_data['states']).to(self.device)
        #     high_reward_actions = torch.FloatTensor(high_reward_data['actions']).to(self.device)
        #     high_reward_rewards = torch.FloatTensor(high_reward_data['rewards']).to(self.device)
        #     high_reward_dones = torch.BoolTensor(high_reward_data['dones']).to(self.device)
        #     high_reward_old_log_probs = torch.FloatTensor(high_reward_data['log_probs']).to(self.device)
        #     high_reward_old_values = torch.FloatTensor(high_reward_data['values']).to(self.device).view(-1, 1)
            
        #     # Compute advantages and returns for high-reward experiences
        #     high_reward_advantages, high_reward_returns = self.compute_gae(
        #         high_reward_rewards, high_reward_old_values, high_reward_dones
        #     )
            
        #     # Normalize high-reward advantages using the same normalization as regular data
        #     if len(high_reward_advantages) > 1:
        #         high_reward_advantages = (high_reward_advantages - high_reward_advantages.mean()) / (high_reward_advantages.std() + 1e-8)
        #     else:
        #         high_reward_advantages = high_reward_advantages - high_reward_advantages.mean()
        
        # PPO update with mini-batches
        total_samples = len(states)
        kl_values = []
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.k_epochs):
            # DONE AATHIRA : Shuffle indices under each k_epochs.
            indices = torch.randperm(total_samples)
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Use mixed precision if enabled
                if self.use_mixed_precision:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # Get current policy outputs
                        #log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                        _, log_probs, values, entropy = self.policy.get_action(batch_states, batch_actions,deterministic=False)
                        log_probs = log_probs.squeeze(1) if log_probs.dim() > 1 and log_probs.shape[1] == 1 else log_probs
                        batch_old_log_probs = batch_old_log_probs.squeeze(1) if batch_old_log_probs.dim() > 1 and batch_old_log_probs.shape[1] == 1 else batch_old_log_probs
                        # Compute ratios with PPO-appropriate clipping
                        log_ratio = log_probs - batch_old_log_probs
                        
                        # PPO theory: ratios should be close to 1.0, so log_ratio should be close to 0
                        # Clip to reasonable range: exp(±2) ≈ [0.135, 7.39], which is reasonable for PPO
                        # log_ratio = torch.clamp(log_ratio, min=-2.0, max=2.0)
                        ratios = torch.exp(log_ratio)
                        
                        approx_kl = torch.abs(batch_old_log_probs - log_probs).mean()
                        kl_values.append(approx_kl.detach().cpu().item())
                        
                        # Additional safety: clip ratios themselves
                        # ratios = torch.clamp(ratios, min=0.1, max=10.0)
                        
                        # Compute surrogate losses
                        surr1 = ratios * batch_advantages
                        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Debug: Log extreme values (commented out due to variable scope)
                        # if len(policy_losses) % 100 == 0:
                        #     print(f"Debug - Log ratio range: [{log_ratio.min().item():.4f}, {log_ratio.max().item():.4f}]")
                        #     print(f"Debug - Ratios range: [{ratios.min().item():.4f}, {ratios.max().item():.4f}]")
                        #     print(f"Debug - Advantages range: [{batch_advantages.min().item():.4f}, {batch_advantages.max().item():.4f}]")
                        #     print(f"Debug - Policy loss: {policy_loss.item():.4f}")
                        
                        # Value loss with optional clipping (PPO-style)
                        # Ensure shapes are compatible
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
                            
                            # Combine clipped and unclipped losses based on mode:
                            # 'max': Conservative - use larger loss (prevents aggressive updates)
                            # 'min': Prefer clipping - use smaller loss (benefits from clipping)
                            # 'mean': Average both (balanced approach)
                            if self.value_loss_mode == 'max':
                                v_loss_combined = torch.max(v_loss_unclipped, v_loss_clipped)
                            elif self.value_loss_mode == 'min':
                                v_loss_combined = torch.min(v_loss_unclipped, v_loss_clipped)
                            elif self.value_loss_mode == 'mean':
                                v_loss_combined = (v_loss_unclipped + v_loss_clipped) / 2.0
                            else:
                                raise ValueError(f"Unknown value_loss_mode: {self.value_loss_mode}. Use 'max', 'min', or 'mean'")
                            
                            value_loss = 0.5 * v_loss_combined.mean()
                        else:
                            value_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
                        
                        # DONE AATHIRA : Don't cap entropy. It should be free to learn.
                        # batch_entropy_mean = entropy.mean().detach().item()
                        # Update EMA of mean and squared mean
                        # self.entropy_ema_mean = self.entropy_ema_beta * self.entropy_ema_mean + (1 - self.entropy_ema_beta) * batch_entropy_mean
                        # self.entropy_ema_sq = self.entropy_ema_beta * self.entropy_ema_sq + (1 - self.entropy_ema_beta) * (batch_entropy_mean ** 2)
                        # ema_var = max(0.0, self.entropy_ema_sq - self.entropy_ema_mean * self.entropy_ema_mean)
                        # ema_std = ema_var ** 0.5
                        # cap_value = max(0.0, self.entropy_ema_mean + self.entropy_cap_k * ema_std)
                        # entropy_capped_value = min(batch_entropy_mean, cap_value)
                        entropy_loss = -entropy.mean()  # Direct entropy loss without capping
                        
                        # Debug: Log all loss components (commented out due to variable scope)
                        # if len(policy_losses) % 100 == 0:
                        #     print(f"Debug - Values range: [{values.min().item():.4f}, {values.max().item():.4f}]")
                        #     print(f"Debug - Returns range: [{batch_returns.min().item():.4f}, {batch_returns.max().item():.4f}]")
                        #     print(f"Debug - Value loss: {value_loss.item():.4f}")
                        #     print(f"Debug - Entropy: {entropy.mean().item():.4f}, Entropy Loss: {entropy_loss.item():.4f}")
                        #     print(f"Debug - Action std mean: {action_std.mean().item():.4f}, Action std max: {action_std.max().item():.4f}")
                        
                        # Total loss
                        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                        
                        # Check for loss explosion
                        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                            print(f"Warning: Loss explosion detected! Policy: {policy_loss.item():.2e}, Value: {value_loss.item():.2e}, Total: {total_loss.item():.2e}")
                            self.nan_count += 1
                            if self.nan_count >= self.max_nan_count:
                                print("Too many loss explosions, resetting model weights")
                                self._reset_model_weights()
                                self.nan_count = 0
                            continue
                        elif torch.isnan(total_loss):
                            print("Warning: NaN detected in total_loss, skipping update")
                            self.nan_count += 1
                            if self.nan_count >= self.max_nan_count:
                                print("Too many NaN occurrences, resetting model weights")
                                self._reset_model_weights()
                                self.nan_count = 0
                            continue
                        else:
                            self.nan_count = 0  # Reset counter on successful update
                    
                    # Update with mixed precision
                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    # Check for NaN gradients
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                        print("Warning: NaN/Inf detected in gradients, skipping update")
                        self.optimizer.zero_grad()
                        continue
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Get current policy outputs
                    # log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                    _, log_probs, values, entropy = self.policy.get_action(batch_states, batch_actions,deterministic=False)
                    log_probs = log_probs.squeeze(1) if log_probs.dim() > 1 and log_probs.shape[1] == 1 else log_probs
                    batch_old_log_probs = batch_old_log_probs.squeeze(1) if batch_old_log_probs.dim() > 1 and batch_old_log_probs.shape[1] == 1 else batch_old_log_probs
                    # Compute ratios with PPO-appropriate clipping
                    log_ratio = log_probs - batch_old_log_probs
                    
                    # Diagnostic prints for first update
                    # if len(policy_losses) == 0 and start_idx == 0:
                    #     print(f"\n=== FIRST UPDATE DIAGNOSTICS ===")
                    #     print(f"log_probs shape: {log_probs.shape}, old_log_probs shape: {batch_old_log_probs.shape}")
                    #     print(f"log_probs range: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]")
                    #     print(f"old_log_probs range: [{batch_old_log_probs.min().item():.4f}, {batch_old_log_probs.max().item():.4f}]")
                    #     print(f"log_ratio range: [{log_ratio.min().item():.4f}, {log_ratio.max().item():.4f}]")
                    #     print(f"================================\n")
                    
                    # PPO theory: ratios should be close to 1.0, so log_ratio should be close to 0
                    # Clip to reasonable range: exp(±2) ≈ [0.135, 7.39], which is reasonable for PPO
                    # DONE AATHIRA : Don't clip log_ratio. It should be free to learn.
                    # log_ratio = torch.clamp(log_ratio, min=-2.0, max=2.0)
                    ratios = torch.exp(log_ratio)
                    # print(f"ratios: {ratios.min().item():.4f}, {ratios.max().item():.4f}")
                    # Additional safety: clip ratios themselves
                    # DONE AATHIRA : Don't clip ratios. It should be free to learn.
                    # ratios = torch.clamp(ratios, min=0.1, max=10.0)
                    
                    approx_kl = torch.abs(batch_old_log_probs - log_probs).mean()
                    kl_values.append(approx_kl.detach().cpu().item())
                    
                    # Compute surrogate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    # Value loss with optional clipping (PPO-style)
                    # Ensure shapes are compatible
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
                        
                        # Combine clipped and unclipped losses based on mode:
                        # 'max': Conservative - use larger loss (prevents aggressive updates)
                        # 'min': Prefer clipping - use smaller loss (benefits from clipping)
                        # 'mean': Average both (balanced approach)
                        if self.value_loss_mode == 'max':
                            v_loss_combined = torch.max(v_loss_unclipped, v_loss_clipped)
                        elif self.value_loss_mode == 'min':
                            v_loss_combined = torch.min(v_loss_unclipped, v_loss_clipped)
                        elif self.value_loss_mode == 'mean':
                            v_loss_combined = (v_loss_unclipped + v_loss_clipped) / 2.0
                        else:
                            raise ValueError(f"Unknown value_loss_mode: {self.value_loss_mode}. Use 'max', 'min', or 'mean'")
                        
                        value_loss = 0.5 * v_loss_combined.mean()
                    else:
                        value_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
                    
                    # DONE AATHIRA : Don't cap entropy. It should be free to learn.
                    # batch_entropy_mean = entropy.mean().detach().item()
                    # Update EMA of mean and squared mean
                    # self.entropy_ema_mean = self.entropy_ema_beta * self.entropy_ema_mean + (1 - self.entropy_ema_beta) * batch_entropy_mean
                    # self.entropy_ema_sq = self.entropy_ema_beta * self.entropy_ema_sq + (1 - self.entropy_ema_beta) * (batch_entropy_mean ** 2)
                    # ema_var = max(0.0, self.entropy_ema_sq - self.entropy_ema_mean * self.entropy_ema_mean)
                    # ema_std = ema_var ** 0.5
                    # cap_value = max(0.0, self.entropy_ema_mean + self.entropy_cap_k * ema_std)
                    # entropy_capped_value = min(batch_entropy_mean, cap_value)
                    entropy_loss = -entropy.mean()  # Direct entropy loss without capping
                    
                    # Total loss
                    total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    # Check for loss explosion
                    # if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                    #     print(f"Warning: Loss explosion detected! Policy: {policy_loss.item():.2e}, Value: {value_loss.item():.2e}, Total: {total_loss.item():.2e}")
                    #     self.nan_count += 1
                    #     if self.nan_count >= self.max_nan_count:
                    #         print("Too many loss explosions, resetting model weights")
                    #         self._reset_model_weights()
                    #         self.nan_count = 0
                    #     continue
                    # elif torch.isnan(total_loss):
                    #     print("Warning: NaN detected in total_loss, skipping update")
                    #     self.nan_count += 1
                    #     if self.nan_count >= self.max_nan_count:
                    #         print("Too many NaN occurrences, resetting model weights")
                    #         self._reset_model_weights()
                    #         self.nan_count = 0
                    #     continue
                    # else:
                    #     self.nan_count = 0  # Reset counter on successful update
                    
                    # Update
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Check for NaN gradients
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                        print("Warning: NaN/Inf detected in gradients, skipping update")
                        self.optimizer.zero_grad()
                        continue
                    
                    self.optimizer.step()
                
                # Store losses for statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Clear cache to free memory
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # # Additional training on high-reward experiences if available
        # if high_reward_states is not None and len(high_reward_states) > 0:
        #     print(f"Training on {len(high_reward_states)} high-reward experiences")
            
        #     # Train on high-reward experiences for additional epochs
        #     for _ in range(min(2, self.k_epochs)):  # Use fewer epochs for high-reward data
        #         # Get current policy outputs for high-reward experiences
        #         if self.use_mixed_precision:
        #             from torch.cuda.amp import autocast
        #             with autocast():
        #                 hr_log_probs, hr_values, hr_entropy = self.policy.evaluate(high_reward_states, high_reward_actions)
                        
        #                 # Compute ratios
        #                 hr_ratios = torch.exp(hr_log_probs - high_reward_old_log_probs)
                        
        #                 # Compute surrogate losses
        #                 hr_surr1 = hr_ratios * high_reward_advantages
        #                 hr_surr2 = torch.clamp(hr_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * high_reward_advantages
        #                 hr_policy_loss = -torch.min(hr_surr1, hr_surr2).mean()
                        
        #                 # Value loss
        #                 hr_value_loss = F.mse_loss(hr_values, high_reward_returns)
                        
        #                 # Entropy loss
        #                 hr_entropy_loss = -hr_entropy.mean()
                        
        #                 # Total loss with higher weight for high-reward experiences
        #                 hr_total_loss = 2.0 * hr_policy_loss + self.value_coef * hr_value_loss + self.entropy_coef * hr_entropy_loss
                        
        #                 # Check for NaN in losses
        #                 if torch.isnan(hr_total_loss):
        #                     print("Warning: NaN detected in high-reward total_loss, skipping update")
        #                     continue
                    
        #             # Update with mixed precision
        #             self.optimizer.zero_grad()
        #             self.scaler.scale(hr_total_loss).backward()
        #             self.scaler.unscale_(self.optimizer)
                    
        #             # Check for NaN gradients
        #             hr_total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        #             if torch.isnan(hr_total_grad_norm) or torch.isinf(hr_total_grad_norm):
        #                 print("Warning: NaN/Inf detected in high-reward gradients, skipping update")
        #                 self.optimizer.zero_grad()
        #                 continue
                    
        #             self.scaler.step(self.optimizer)
        #             self.scaler.update()
        #         else:
        #             # Get current policy outputs for high-reward experiences
        #             hr_log_probs, hr_values, hr_entropy = self.policy.evaluate(high_reward_states, high_reward_actions)
                    
        #             # Compute ratios
        #             hr_ratios = torch.exp(hr_log_probs - high_reward_old_log_probs)
                    
        #             # Compute surrogate losses
        #             hr_surr1 = hr_ratios * high_reward_advantages
        #             hr_surr2 = torch.clamp(hr_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * high_reward_advantages
        #             hr_policy_loss = -torch.min(hr_surr1, hr_surr2).mean()
                    
        #             # Value loss
        #             hr_value_loss = F.mse_loss(hr_values, high_reward_returns)
                    
        #             # Entropy loss
        #             hr_entropy_loss = -hr_entropy.mean()
                    
        #             # Total loss with higher weight for high-reward experiences
        #             hr_total_loss = 2.0 * hr_policy_loss + self.value_coef * hr_value_loss + self.entropy_coef * hr_entropy_loss
                    
        #             # Check for NaN in losses
        #             if torch.isnan(hr_total_loss):
        #                 print("Warning: NaN detected in high-reward total_loss, skipping update")
        #                 continue
                    
        #             # Update
        #             self.optimizer.zero_grad()
        #             hr_total_loss.backward()
                    
        #             # Check for NaN gradients
        #             hr_total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        #             if torch.isnan(hr_total_grad_norm) or torch.isinf(hr_total_grad_norm):
        #                 print("Warning: NaN/Inf detected in high-reward gradients, skipping update")
        #                 self.optimizer.zero_grad()
        #                 continue
                    
        #             self.optimizer.step()
                
        #         # Store high-reward losses for statistics
        #         policy_losses.append(hr_policy_loss.item())
        #         value_losses.append(hr_value_loss.item())
        #         entropy_losses.append(hr_entropy_loss.item())
                
        #         # Clear cache to free memory
        #         if self.device == 'cuda':
        #             torch.cuda.empty_cache()
        
        # Store training statistics (averaged over all mini-batches)
        if policy_losses:
            self.training_stats['policy_loss'].append(np.mean(policy_losses))
            self.training_stats['value_loss'].append(np.mean(value_losses))
            self.training_stats['entropy_loss'].append(np.mean(entropy_losses))
            total_loss_avg = np.mean(policy_losses) + self.value_coef * np.mean(value_losses) + self.entropy_coef * np.mean(entropy_losses)
            self.training_stats['total_loss'].append(total_loss_avg)
        
        if kl_values:
            mean_policy_shift = float(np.mean(kl_values))
            self.training_stats['policy_shift'].append(mean_policy_shift)
            self._adjust_learning_rate(mean_policy_shift)
        else:
            mean_policy_shift = None
            self.training_stats['policy_shift'].append(float('nan'))
        
        if mean_policy_shift is not None:
            self.training_stats['learning_rate'].append(self.current_lr)
        elif self.training_stats['learning_rate']:
            # Repeat last learning rate to keep stats aligned
            self.training_stats['learning_rate'].append(self.training_stats['learning_rate'][-1])
        else:
            self.training_stats['learning_rate'].append(self.current_lr)
        
        # Clear memory
        self.memory.clear()
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def compute_gae(self, rewards, values, dones, next_states=None, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor
            values: Value estimates tensor
            dones: Done flags tensor
            next_states: Next states tensor (optional, used to get value for last next_state)
            lam: GAE lambda parameter
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        # Ensure tensors are the right shape
        rewards = rewards.view(-1)
        values = values.view(-1)  # Convert from [batch_size, 1] to [batch_size]
        dones = dones.view(-1)
        
        # Debug: Print shapes if they seem wrong
        if len(rewards) != len(values) or len(rewards) != len(dones):
            print(f"Warning: Shape mismatch in GAE computation:")
            print(f"  rewards shape: {rewards.shape}")
            print(f"  values shape: {values.shape}")
            print(f"  dones shape: {dones.shape}")
        
        # AATHIRA : Calculate the next value for the last next_state.
        # Compute next values - get value from policy network for last next_state
        if next_states is not None and len(next_states) > 0:
            # Get value for the last next_state from the policy network
            with torch.no_grad():
                _, _, last_next_value = self.policy.forward(next_states[-1:])
                # Convert from [1, 1] to scalar, then to 1D tensor [1] on same device as values
                last_next_value = last_next_value.squeeze().unsqueeze(0).to(values.device)
        else:
            # Fallback to 0 if next_states not provided
            last_next_value = torch.zeros(1, device=values.device)
        
        next_values = torch.cat([values[1:], last_next_value])
        
        # Compute TD errors
        td_errors = rewards + self.gamma * next_values * (~dones).float() - values
        
        # Compute GAE advantages
        advantages = []
        advantage = 0
        for i in reversed(range(len(td_errors))):
            advantage = td_errors[i] + self.gamma * lam * (~dones[i]).float() * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.stack(advantages)
        returns = advantages + values
        
        # Ensure returns has the same shape as policy network output [batch_size, 1]
        returns = returns.view(-1, 1)
        
        return advantages, returns
    
    def _adjust_learning_rate(self, mean_policy_shift):
        """
        Dynamically adjust the learning rate based on how much the policy
        has shifted from the behaviour policy that generated the data.
        
        Args:
            mean_policy_shift (float): Average absolute difference between
                old and new log probabilities (approximate policy shift).
        """
        if np.isnan(mean_policy_shift) or np.isinf(mean_policy_shift):
            return
        
        self.policy_shift_history.append(mean_policy_shift)
        smoothed_shift = float(np.mean(self.policy_shift_history))
        
        upper_bound = self.target_policy_shift + self.policy_shift_tolerance  #0.015
        lower_bound = max(0.0, self.target_policy_shift - self.policy_shift_tolerance) #0.005
        
        previous_lr = self.current_lr
        if smoothed_shift > upper_bound and self.current_lr > self.min_lr:
            self.current_lr = max(self.min_lr, self.current_lr / self.lr_decrease_factor)
        elif smoothed_shift < lower_bound and self.current_lr < self.max_lr:
            self.current_lr = min(self.max_lr, self.current_lr * self.lr_increase_factor)
        
        if self.current_lr != previous_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            print(
                f"Adjusted learning rate from {previous_lr:.6e} to {self.current_lr:.6e} "
                f"(smoothed policy shift: {smoothed_shift:.6f})"
            )
        else:
            # Ensure optimizer param groups stay in sync even if unchanged
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
    
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def save_best_model(self, reward, filepath, episode=None):
        """Save model only if it achieves a new best reward."""
        if reward > self.best_reward:
            previous_best = self.best_reward
            self.best_reward = reward
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'best_reward': self.best_reward
            }, filepath)
            episode_str = f" at Episode {episode}" if episode is not None else ""
            print(f"New best model saved{episode_str}! Reward: {reward:.2f} (Previous best: {previous_best:.2f})")
            return True
        return False
    
    def save_high_reward_model(self, reward, episode, base_model_path, save_dir="saved_last_10"):
        """
        Save model if reward > 310, keeping only the last 10 models.
        
        Args:
            reward: Current episode reward (or average reward)
            episode: Current episode number
            base_model_path: Base path for model (used to generate filename)
            save_dir: Directory to save high-reward models (default: "saved_last_10")
        """
        if reward > self.high_reward_threshold:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename
            model_name = os.path.basename(base_model_path).rsplit('.', 1)[0] if '.' in base_model_path else os.path.basename(base_model_path)
            filename = f"{model_name}_ep{episode}_reward{reward:.2f}.pth"
            filepath = os.path.join(save_dir, filename)
            
            # Save the model
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'episode': episode,
                'reward': reward
            }, filepath)
            
            # Add to list
            self.high_reward_models.append((episode, reward, filepath))
            
            # Keep only the last 10 models
            if len(self.high_reward_models) > self.max_high_reward_models:
                # Remove the oldest model (first in list)
                old_episode, old_reward, old_filepath = self.high_reward_models.pop(0)
                # Delete the old file
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
                    print(f"Removed old high-reward model: Episode {old_episode}, Reward {old_reward:.2f}")
            
            print(f"High-reward model saved! Episode {episode}, Reward: {reward:.2f} (Saved to {filepath}, Total saved: {len(self.high_reward_models)})")
            return True
        return False
    
    def load_model(self, filepath):
        """Load a trained model."""
        if os.path.exists(filepath):
            # weights_only=False is needed for PyTorch 2.6+ to load checkpoints with numpy objects
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return self.training_stats
    
    def _reset_model_weights(self):
        """Reset model weights to prevent persistent NaN issues."""
        print("Resetting model weights...")
        # Reinitialize the policy network
        self.policy.apply(self.policy._init_weights)
        # Reset optimizer state
        reset_lr = min(self.lr, 1e-4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=reset_lr)
        self.max_lr = max(self.lr, reset_lr)
        # Set minimum learning rate to prevent learning from stopping
        # 1e-6 is about the lowest practical learning rate for neural networks
        self.min_lr = 1e-5
        self.current_lr = reset_lr
        self.initial_lr = reset_lr
        self.policy_shift_history.clear()
        # Clear memory to start fresh
        self.memory.clear()
        print("Model weights reset successfully")


class PPOMemory:
    """Memory buffer for storing PPO experiences with high-reward prioritization."""
    
    def __init__(self, max_size=2000, high_reward_threshold=1.5):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.max_size = max_size
        self.high_reward_threshold = high_reward_threshold
        self.high_reward_indices = []  # Track indices of high-reward experiences
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition with high-reward prioritization."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # Track high-reward experiences
        if reward >= self.high_reward_threshold:
            self.high_reward_indices.append(len(self.states) - 1)
        
        # Limit memory size to prevent memory issues
        if len(self.states) > self.max_size:
            # Remove oldest experience
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
            
            # Update high-reward indices
            self.high_reward_indices = [idx - 1 for idx in self.high_reward_indices if idx > 0]
    
    def clear(self):
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.high_reward_indices.clear()
    
    def get_high_reward_experiences(self, num_samples=None):
        """Get high-reward experiences for prioritized replay."""
        if not self.high_reward_indices:
            return None
        
        if num_samples is None:
            num_samples = min(len(self.high_reward_indices), 32)  # Default batch size
        
        # Sample from high-reward experiences
        selected_indices = np.random.choice(
            self.high_reward_indices, 
            size=min(num_samples, len(self.high_reward_indices)), 
            replace=False
        )
        
        return {
            'states': [self.states[i] for i in selected_indices],
            'actions': [self.actions[i] for i in selected_indices],
            'rewards': [self.rewards[i] for i in selected_indices],
            'next_states': [self.next_states[i] for i in selected_indices],
            'dones': [self.dones[i] for i in selected_indices],
            'log_probs': [self.log_probs[i] for i in selected_indices],
            'values': [self.values[i] for i in selected_indices]
        }


def train_ppo_agent(env, agent, num_episodes=1000, max_steps=500, 
                   save_interval=100, model_path="ppo_model.pth", save_best_only=False,
                   episodes_per_update=4, min_buffer_size=200, save_best_after_episode=0,
                   save_final_model=True, save_high_reward_after_episode=0):
    """
    Train PPO agent on the environment.
    
    Args:
        env: Environment instance
        agent: PPO agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_interval: Interval for saving model
        model_path: Path to save model
        save_best_only: If True, only save the best model (overwrites previous best)
        episodes_per_update: Number of episodes to accumulate before updating (default: 4)
        min_buffer_size: Minimum number of transitions before updating (default: 200)
        save_best_after_episode: Start saving best model only after this episode number (default: 0)
        save_final_model: Whether to save the final model after training completes (default: True)
        save_high_reward_after_episode: Start saving high-reward models (reward > 310) after this episode number (default: 0)
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
            'episodes_per_update': episodes_per_update,
            'min_buffer_size': min_buffer_size,
        }, allow_val_change=True)
    
    # Accumulate data across multiple episodes to reduce zig-zag pattern
    episodes_since_update = 0
    total_steps_collected = 0
    
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.initialize()
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.take_step(action)
            
            # Store transition (data accumulates in memory)
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps_collected += 1
            
            if done:
                break
        
        episodes_since_update += 1
        
        # Store episode statistics
        agent.training_stats['episode_rewards'].append(episode_reward)
        agent.training_stats['episode_lengths'].append(episode_length)
        
        # Update policy only when we have accumulated enough data
        # This reduces zig-zag pattern by:
        # 1. Providing larger, more stable batches
        # 2. Reducing overfitting to single episodes
        # 3. Better data diversity across episodes
        should_update = (
            episodes_since_update >= episodes_per_update and
            len(agent.memory.states) >= min_buffer_size
        )
        
        if should_update and len(agent.memory.states) >= 32:  # Minimum batch size check
            # Update agent on accumulated data
            agent.update()
            episodes_since_update = 0
            total_steps_collected = 0
        elif len(agent.memory.states) >= agent.memory.max_size:
            # Prevent memory overflow - update even if criteria not met
            agent.update()
            episodes_since_update = 0
            total_steps_collected = 0

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
            if agent.training_stats['learning_rate']:
                log_payload['train/learning_rate'] = agent.training_stats['learning_rate'][-1]
            if len(agent.training_stats['episode_rewards']) >= 10:
                log_payload['episode/avg_reward_10'] = float(np.mean(agent.training_stats['episode_rewards'][-10:]))
                log_payload['episode/avg_length_10'] = float(np.mean(agent.training_stats['episode_lengths'][-10:]))
            wandb.log(log_payload)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
            avg_length = np.mean(agent.training_stats['episode_lengths'][-10:])
            high_reward_count = len(agent.memory.high_reward_indices)
            buffer_size = len(agent.memory.states)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, "
                  f"Buffer Size: {buffer_size}, Episodes Since Update: {episodes_since_update}, "
                  f"High-Reward Samples: {high_reward_count}")
        
        if save_best_only:
                # Only save best model after the specified episode number
                if episode >= save_best_after_episode:
                    # Create proper filename: replace .pth with _best.pth
                    model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
                    best_model_path = f"{model_name}_best.pth"
                    # Use average reward (last 10 episodes) for best model saving
                    if len(agent.training_stats['episode_rewards']) >= 10:
                        avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                        best_model_saved = agent.save_best_model(avg_reward, best_model_path, episode=episode)
                    else:
                        # If less than 10 episodes, use current episode reward
                        best_model_saved = agent.save_best_model(episode_reward, best_model_path, episode=episode)
                    if best_model_saved and _WANDB_AVAILABLE and wandb.run is not None:
                        wandb.save(best_model_path)
        else:
                agent.save_model(f"{model_path}_{episode}")
                if _WANDB_AVAILABLE and wandb.run is not None:
                    wandb.save(f"{model_path}_{episode}")
        
        # Save high-reward models (reward > 310) after specified episode
        # This is independent of save_best_only setting
        if episode >= save_high_reward_after_episode:
            # Use average reward if available, otherwise use episode reward
            if len(agent.training_stats['episode_rewards']) >= 10:
                avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                agent.save_high_reward_model(avg_reward, episode, model_path)
            else:
                agent.save_high_reward_model(episode_reward, episode, model_path)
    
    # Final update on remaining data
    if len(agent.memory.states) >= 32:
        print(f"Final update on remaining {len(agent.memory.states)} transitions")
        agent.update()
    
    # Save final model if requested
    if save_final_model:
        # Create saved_final_models directory if it doesn't exist
        final_models_dir = "saved_final_models"
        os.makedirs(final_models_dir, exist_ok=True)
        
        # Extract model name from model_path (without extension)
        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
        # Remove directory path if present, keep only filename
        model_basename = os.path.basename(model_name)
        final_model_path = os.path.join(final_models_dir, f"{model_basename}_final.pth")
        
        agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.save(final_model_path)
    
    print("Training completed!")


if __name__ == "__main__":

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
    parser.add_argument('--no-clip-vloss', '--no_clip_vloss', action='store_false', dest='clip_vloss', help='Disable clipped value loss (default: clip_vloss=True)')
    parser.add_argument('--value_clip_coef', type=float, default=0.2, help='Value clipping coefficient (similar to eps_clip)')
    parser.add_argument('--value_loss_mode', type=str, default='min', choices=['max', 'min', 'mean'], 
                        help='How to combine clipped/unclipped value losses: max=conservative, min=prefer clipping (default), mean=average')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (if multiple GPUs available)')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training (GPU only)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between checkpoints')
    parser.add_argument('--model_path', type=str, default='saved_models/ppo_model.pth', help='Model save path')
    parser.add_argument('--save_best_only', action='store_true', help='Save only the best model (overwrites previous best)')
    parser.add_argument('--enable_viewer', action='store_true', help='Enable MuJoCo viewer')
    parser.add_argument('--visualize_collision_boxes', action='store_true', help='Visualize collision boxes')
    parser.add_argument('--rotation_sequence', type=str, nargs='*', default=None,
                        help='Rotation sequence of face names (e.g., --rotation_sequence blue)')
    parser.add_argument('--enable_gravity', dest='enable_gravity', action='store_true',
                        help='Override environment gravity with default vector [0, 0, 0]')
    parser.add_argument('--disable_gravity', dest='enable_gravity', action='store_false',
                        help='Disable gravity override and use gravity from XML')
    parser.set_defaults(enable_gravity=False)
    parser.add_argument('--high_reward_threshold', type=float, default=1.5, help='Reward threshold for high-reward experience prioritization')
    parser.add_argument('--episodes_per_update', type=int, default=4, help='Number of episodes to accumulate before updating (reduces zig-zag pattern)')
    parser.add_argument('--min_buffer_size', type=int, default=500, help='Minimum number of transitions before updating (reduces zig-zag pattern)')
    parser.add_argument('--save_best_after_episode', type=int, default=0, help='Start saving best model only after this episode number (default: 0, saves from start)')
    parser.add_argument('--save_high_reward_after_episode', type=int, default=0, help='Start saving high-reward models (reward > 310) after this episode number (default: 0, saves from start)')
    parser.add_argument('--save_final_model', action='store_true', default=True, help='Save the final model after training completes in saved_final_models directory (default: True)')
    parser.add_argument('--project', type=str, default='mujoco-rubiks-ppo', help='wandb project name')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_model', type=str, default=None, help='Path to model checkpoint to load before training')
    args = parser.parse_args()
    
    # Set default for clip_vloss if not provided (defaults to True)
    if not hasattr(args, 'clip_vloss') or args.clip_vloss is None:
        args.clip_vloss = True

    # Determine device
    device = get_device(args.device, args.gpu_id)
    
    # Print GPU information
    print_gpu_info(device)
    
    # Seeding
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        # Set CUDA device
        torch.cuda.set_device(device)

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
            'device': str(device),
            'seed': args.seed,
            'gravity_override': args.enable_gravity,
        })

    # Create environment
    rotation_sequence = args.rotation_sequence if args.rotation_sequence else None
    if rotation_sequence:
        print(f"Rotation sequence: {rotation_sequence}")
    env = RubiksCubeEnvironment(
        xml_path=args.xml,
        enable_viewer=args.enable_viewer,
        visualize_collision_boxes=args.visualize_collision_boxes,
        enable_gravity=args.enable_gravity,
        rotation_sequence=rotation_sequence,
        max_episode_steps=args.max_steps
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
        device=device,
        use_mixed_precision=args.use_mixed_precision,
        batch_size=args.batch_size,
        high_reward_threshold=args.high_reward_threshold,
        clip_vloss=args.clip_vloss,
        value_clip_coef=args.value_clip_coef,
        value_loss_mode=args.value_loss_mode,
    )

    # Load model if specified
    if args.load_model:
        agent.load_model(args.load_model)

    # Train
    try:
        train_ppo_agent(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            save_interval=args.save_interval,
            model_path=args.model_path,
            save_best_only=args.save_best_only,
            episodes_per_update=args.episodes_per_update,
            min_buffer_size=args.min_buffer_size,
            save_best_after_episode=args.save_best_after_episode,
            save_final_model=args.save_final_model,
            save_high_reward_after_episode=args.save_high_reward_after_episode,
        )
    finally:
        env.close()
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
