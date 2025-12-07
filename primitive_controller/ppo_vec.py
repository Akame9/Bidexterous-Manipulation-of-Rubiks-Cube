"""
Optimized Proximal Policy Optimization (PPO) implementation with Phase 1 optimizations.

Phase 1 Optimizations:
1. Vectorized environments support (parallel rollouts)
2. torch.compile() for faster execution
3. Vectorized GAE computation
4. Pre-allocated tensor buffers
5. Removed debug prints for faster training

This is a drop-in replacement for ppo.py with performance optimizations.
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
from environment.vectorized_env import VectorizedEnv
# Import ActorCritic from ppo.py to avoid duplication
from primitive_controller.ppo import ActorCritic

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


class OptimizedPPOMemory:
    """
    Optimized PPO memory buffer with pre-allocated NumPy arrays.
    This is much faster than using Python lists.
    """
    
    def __init__(self, max_size=2000, state_dim=None, action_dim=None, high_reward_threshold=1.5):
        self.max_size = max_size
        self.high_reward_threshold = high_reward_threshold
        self.high_reward_indices = []
        
        # Pre-allocate NumPy arrays if dimensions are known
        if state_dim is not None and action_dim is not None:
            self.states = np.zeros((max_size, state_dim), dtype=np.float32)
            self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
            self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
            self.rewards = np.zeros(max_size, dtype=np.float32)
            self.dones = np.zeros(max_size, dtype=bool)
            self.log_probs = np.zeros(max_size, dtype=np.float32)
            self.values = np.zeros(max_size, dtype=np.float32)
            self.env_ids = np.zeros(max_size, dtype=np.int32)  # Track which environment each transition belongs to
            self.use_preallocated = True
        else:
            # Fallback to lists if dimensions unknown
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.log_probs = []
            self.values = []
            self.env_ids = []
            self.use_preallocated = False
        
        self.idx = 0
        self.size = 0
    
    def store(self, state, action, reward, next_state, done, log_prob, value, env_id=0):
        """Store a transition with environment ID."""
        if self.use_preallocated:
            idx = self.idx % self.max_size
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = done
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.env_ids[idx] = env_id
            
            if reward >= self.high_reward_threshold:
                self.high_reward_indices.append(idx)
            
            self.idx += 1
            self.size = min(self.size + 1, self.max_size)
        else:
            # Fallback to list-based storage
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.env_ids.append(env_id)
            
            if reward >= self.high_reward_threshold:
                self.high_reward_indices.append(len(self.states) - 1)
            
            if len(self.states) > self.max_size:
                self.states.pop(0)
                self.actions.pop(0)
                self.rewards.pop(0)
                self.next_states.pop(0)
                self.dones.pop(0)
                self.log_probs.pop(0)
                self.values.pop(0)
                self.env_ids.pop(0)
                self.high_reward_indices = [idx - 1 for idx in self.high_reward_indices if idx > 0]
    
    def get_data(self):
        """Get all stored data as arrays."""
        if self.use_preallocated:
            if self.size < self.max_size:
                # Return only filled portion
                return {
                    'states': self.states[:self.size],
                    'actions': self.actions[:self.size],
                    'rewards': self.rewards[:self.size],
                    'next_states': self.next_states[:self.size],
                    'dones': self.dones[:self.size],
                    'log_probs': self.log_probs[:self.size],
                    'values': self.values[:self.size],
                    'env_ids': self.env_ids[:self.size]
                }
            else:
                # Circular buffer - return in order
                indices = np.arange(self.idx, self.idx + self.max_size) % self.max_size
                return {
                    'states': self.states[indices],
                    'actions': self.actions[indices],
                    'rewards': self.rewards[indices],
                    'next_states': self.next_states[indices],
                    'dones': self.dones[indices],
                    'log_probs': self.log_probs[indices],
                    'values': self.values[indices],
                    'env_ids': self.env_ids[indices]
                }
        else:
            return {
                'states': np.array(self.states),
                'actions': np.array(self.actions),
                'rewards': np.array(self.rewards),
                'next_states': np.array(self.next_states),
                'dones': np.array(self.dones),
                'log_probs': np.array(self.log_probs),
                'values': np.array(self.values),
                'env_ids': np.array(self.env_ids)
            }
    
    def clear(self):
        """Clear all stored data."""
        old_size = self.size if self.use_preallocated else len(self.states)
        if self.use_preallocated:
            self.idx = 0
            self.size = 0
            self.high_reward_indices.clear()
        else:
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()
            self.log_probs.clear()
            self.values.clear()
            self.env_ids.clear()
            self.high_reward_indices.clear()
    
    def __len__(self):
        return self.size if self.use_preallocated else len(self.states)


class PPOAgentVec:
    """
    Optimized PPO agent with Phase 1 improvements:
    - torch.compile() support
    - Vectorized GAE computation
    - Pre-allocated memory buffers
    - Batch action selection for vectorized environments
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=10, entropy_coef=0.01, 
                 value_coef=0.5, max_grad_norm=0.5, device='cpu', 
                 use_mixed_precision=False, batch_size=64, 
                 high_reward_threshold=1.5, clip_vloss=True, 
                 value_clip_coef=0.2, value_loss_mode='min',
                 use_torch_compile=True, num_envs=1, max_steps=500):
        """
        Initialize optimized PPO agent.
        
        Args:
            use_torch_compile: Whether to use torch.compile() for faster execution (PyTorch 2.0+)
            num_envs: Number of parallel environments (for calculating memory size)
            max_steps: Maximum steps per episode (for calculating memory size)
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
        self.value_loss_mode = value_loss_mode
        
        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        
        # Apply torch.compile() if available and requested
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.policy = torch.compile(self.policy, mode='reduce-overhead')
                print("Using torch.compile() for faster execution")
            except Exception as e:
                print(f"Warning: torch.compile() failed: {e}, continuing without it")
        
        safe_lr = min(lr, 1e-4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=safe_lr)
        if safe_lr != lr:
            print(f"Warning: Learning rate reduced from {lr} to {safe_lr} to prevent instability")
        
        # Dynamic learning rate configuration
        self.initial_lr = safe_lr
        self.current_lr = safe_lr
        self.max_lr = max(lr, safe_lr)
        self.min_lr = 1e-5
        self.target_policy_shift = 0.01
        self.policy_shift_tolerance = 0.005
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
        
        # Track best reward for saving best model
        self.best_reward = float('-inf')
        
        # Track high-reward models (reward > 310) - keep last 10
        self.high_reward_models = []  # List of (episode, reward, filepath) tuples
        self.high_reward_threshold = 310.0
        self.max_high_reward_models = 10
        
        # Track consecutive NaN occurrences
        self.nan_count = 0
        self.max_nan_count = 5
        
        # Calculate memory size based on num_envs * max_steps
        # This ensures we can store all transitions from one full episode across all environments
        memory_max_size = num_envs * max_steps
        
        # Memory for storing experiences (with pre-allocated buffers)
        self.memory = OptimizedPPOMemory(
            max_size=memory_max_size,
            state_dim=state_dim,
            action_dim=action_dim,
            high_reward_threshold=high_reward_threshold
        )
        
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
        Select action given current state (single state).
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        # Validate input state
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
        # print(f"Unique env_ids: {unique_env_ids}")
        for env_id in unique_env_ids:
            # print(f"Group transitions by environment env_id: {env_id}")
            env_mask = (env_ids == env_id)
            env_indices = torch.where(env_mask)[0]
            
            if len(env_indices) == 0:
                continue
            
            env_rewards = rewards[env_indices]
            env_values = values[env_indices]
            env_dones = dones[env_indices]
            
            done_mask = env_dones
            done_positions = torch.where(done_mask)[0]  # Positions within env_indices, not global indices
            print(f"Done positions: {done_positions}")
            # Process each episode separately
            episode_start = 0
            for done_pos in done_positions:
                # Get episode data (inclusive of done step)
                episode_local_indices = torch.arange(episode_start, done_pos + 1, device=env_indices.device)
                episode_global_indices = env_indices[episode_local_indices]  # Original indices
                
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
        """Update policy using PPO algorithm with optimizations."""
        memory_size = len(self.memory)
        if memory_size < 32:
            return
        
        # Get all stored data (already as arrays)
        data = self.memory.get_data()
        
        # Convert to tensors (faster than converting from lists)
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
        
        # PPO update with mini-batches
        total_samples = len(states)
        kl_values = []
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.k_epochs):
            # Shuffle indices
            indices = torch.randperm(total_samples)
            for start_idx in range(0, total_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_samples)
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
                        _, log_probs, values, entropy = self.policy.get_action(
                            batch_states, batch_actions, deterministic=False
                        )
                        log_probs = log_probs.squeeze(1) if log_probs.dim() > 1 and log_probs.shape[1] == 1 else log_probs
                        batch_old_log_probs = batch_old_log_probs.squeeze(1) if batch_old_log_probs.dim() > 1 and batch_old_log_probs.shape[1] == 1 else batch_old_log_probs
                        
                        log_ratio = log_probs - batch_old_log_probs
                        ratios = torch.exp(log_ratio)
                        
                        approx_kl = torch.abs(batch_old_log_probs - log_probs).mean()
                        kl_values.append(approx_kl.detach().cpu().item())
                        
                        # Compute surrogate losses
                        surr1 = ratios * batch_advantages
                        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
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
                        
                        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                            self.nan_count += 1
                            if self.nan_count >= self.max_nan_count:
                                self._reset_model_weights()
                                self.nan_count = 0
                            continue
                        else:
                            self.nan_count = 0
                    
                    # Update with mixed precision
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
                    # Standard precision training
                    _, log_probs, values, entropy = self.policy.get_action(
                        batch_states, batch_actions, deterministic=False
                    )
                    log_probs = log_probs.squeeze(1) if log_probs.dim() > 1 and log_probs.shape[1] == 1 else log_probs
                    batch_old_log_probs = batch_old_log_probs.squeeze(1) if batch_old_log_probs.dim() > 1 and batch_old_log_probs.shape[1] == 1 else batch_old_log_probs
                    
                    log_ratio = log_probs - batch_old_log_probs
                    ratios = torch.exp(log_ratio)
                    
                    approx_kl = torch.abs(batch_old_log_probs - log_probs).mean()
                    kl_values.append(approx_kl.detach().cpu().item())
                    
                    # Compute surrogate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
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
                    
                    # Update
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                        self.optimizer.zero_grad()
                        continue
                    
                    self.optimizer.step()
                
                # Store losses for statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Store training statistics
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
            self.training_stats['policy_shift'].append(float('nan'))
        
        if kl_values:
            self.training_stats['learning_rate'].append(self.current_lr)
        elif self.training_stats['learning_rate']:
            self.training_stats['learning_rate'].append(self.training_stats['learning_rate'][-1])
        else:
            self.training_stats['learning_rate'].append(self.current_lr)
        
        # Clear memory
        self.memory.clear()
    
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
    
    def _reset_model_weights(self):
        """Reset model weights to prevent persistent NaN issues."""
        print("Resetting model weights...")
        self.policy.apply(self.policy._init_weights)
        reset_lr = min(self.lr, 1e-4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=reset_lr)
        self.max_lr = max(self.lr, reset_lr)
        self.min_lr = 1e-5
        self.current_lr = reset_lr
        self.initial_lr = reset_lr
        self.policy_shift_history.clear()
        self.memory.clear()
        print("Model weights reset successfully")
    
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
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            state_dict = checkpoint['policy_state_dict']
            
            # Handle torch.compile() key mismatch
            # If model is compiled with torch.compile(), it wraps the model and adds "_orig_mod." prefix
            # We need to load into the underlying model if it's compiled
            target_model = self.policy
            is_compiled = hasattr(self.policy, '_orig_mod')
            
            # Check if state_dict keys have _orig_mod prefix
            state_dict_keys = list(state_dict.keys())
            has_orig_mod_prefix = any(key.startswith('_orig_mod.') for key in state_dict_keys)
            
            # Case 1: Model is compiled, checkpoint does NOT have _orig_mod prefix
            # Load into the underlying model (which expects keys without prefix)
            if is_compiled and not has_orig_mod_prefix:
                target_model = self.policy._orig_mod
                print("Loading non-compiled checkpoint into compiled model (using underlying model)")
            
            # Case 2: Model is compiled, checkpoint HAS _orig_mod prefix
            # Need to remove prefix and load into underlying model
            elif is_compiled and has_orig_mod_prefix:
                target_model = self.policy._orig_mod
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('_orig_mod.'):
                        new_key = key[len('_orig_mod.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
                print("Loading compiled checkpoint into compiled model (removed _orig_mod prefix)")
            
            # Case 3: Model is NOT compiled, checkpoint HAS _orig_mod prefix
            # Remove prefix from checkpoint keys
            elif not is_compiled and has_orig_mod_prefix:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('_orig_mod.'):
                        new_key = key[len('_orig_mod.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
                print("Removed _orig_mod prefix from checkpoint keys (loading into non-compiled model)")
            
            # Case 4: Model is NOT compiled, checkpoint does NOT have _orig_mod prefix
            # Normal loading, no changes needed
            else:
                print("Loading checkpoint (both model and checkpoint are non-compiled)")
            
            # Load the state dict
            try:
                target_model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                # If strict loading fails, try with strict=False
                print(f"Warning: Some keys didn't match, loading with strict=False")
                target_model.load_state_dict(state_dict, strict=False)
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return self.training_stats


def train_ppo_agent_vec(env_fn, agent, num_episodes=1000, max_steps=500, 
                       save_interval=100, model_path="ppo_model.pth", 
                       save_best_only=False, episodes_per_update=4, 
                       min_buffer_size=200, save_best_after_episode=0,
                       save_final_model=True, num_envs=8, use_vectorized=True,
                       save_high_reward_after_episode=0):
    """
    Train PPO agent with vectorized environments for faster data collection.
    
    Args:
        env_fn: Function that creates a single environment instance
        agent: PPO agent (PPOAgentVec)
        num_episodes: Number of training episodes (total across all envs)
        max_steps: Maximum steps per episode
        save_interval: Interval for saving model
        model_path: Path to save model
        save_best_only: If True, only save the best model
        episodes_per_update: Number of episodes to accumulate before updating
        min_buffer_size: Minimum number of transitions before updating
        save_best_after_episode: Start saving best model after this episode
        save_final_model: Whether to save final model
        num_envs: Number of parallel environments (only used if use_vectorized=True)
        use_vectorized: Whether to use vectorized environments
        save_high_reward_after_episode: Start saving high-reward models (reward > 310) after this episode number (default: 0)
    """
    print("Starting optimized PPO training...")
    if use_vectorized:
        print(f"Using {num_envs} parallel environments for {num_envs}x faster data collection")
    
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.config.update({
            'num_episodes': num_episodes,
            'max_steps': max_steps,
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
                if active_envs.any():
                    # Select actions only for active (non-done) environments
                    active_states = states[active_envs]
                    active_actions, active_log_probs, active_values = agent.select_actions_batch(active_states)
                    
                    # Create full arrays with dummy values for done environments
                    actions = np.zeros((num_envs, agent.action_dim), dtype=np.float32)
                    log_probs = np.zeros(num_envs, dtype=np.float32)
                    values = np.zeros(num_envs, dtype=np.float32)
                    
                    # Fill in actions, log_probs, and values for active environments
                    actions[active_envs] = active_actions
                    log_probs[active_envs] = active_log_probs
                    values[active_envs] = active_values
                else:
                    # All environments are done, create dummy arrays
                    actions = np.zeros((num_envs, agent.action_dim), dtype=np.float32)
                    log_probs = np.zeros(num_envs, dtype=np.float32)
                    values = np.zeros(num_envs, dtype=np.float32)
                
                # Step all environments in parallel (vectorized_env will skip done ones)
                next_states, rewards, dones, infos = vec_env.step(actions)
                
                # Store transitions for all environments
                # Only store if environment was not already done before this step
                # (to avoid storing transitions after episode has ended)
                transitions_stored = 0
                skipped_already_done = 0
                for i in range(num_envs):
                    # Skip if environment was already done before this step
                    if not dones_before_step[i]:
                        agent.store_transition(
                            states[i],
                            actions[i],
                            rewards[i],
                            next_states[i],
                            dones[i],
                            log_probs[i],
                            values[i],
                            env_id=i  # Track which environment this transition belongs to
                        )
                        transitions_stored += 1
                    else:
                        skipped_already_done += 1
                
                states = next_states
                
                # Track episode statistics when environments finish (but don't reset yet)
                # All environments will be reset together at the start of the next epoch
                # Only count environments that just became done (not already counted)
                if dones.any():
                    done_indices = np.where(dones)[0].tolist()
                    # Filter to only newly done environments (not already counted)
                    newly_done_indices = [i for i in done_indices if i not in counted_for_stats]
                    
                    if newly_done_indices:
                        # Store episode statistics only for newly done environments
                        stats = vec_env.get_episode_stats()
                        for i in newly_done_indices:
                            agent.training_stats['episode_rewards'].append(stats['rewards'][i])
                            agent.training_stats['episode_lengths'].append(stats['lengths'][i])
                            counted_for_stats.add(i)  # Mark as counted
                        episodes_since_update += len(newly_done_indices)
                        total_episodes += len(newly_done_indices)
                
                # Early break if all done (will reset all at start of next epoch)
                if dones.all():
                    break
            
            # Update policy when enough data collected
            buffer_size = len(agent.memory)
            should_update = (
                episodes_since_update >= episodes_per_update and
                buffer_size >= min_buffer_size
            )
            
            
            if should_update and buffer_size >= 32:
                agent.update()
                episodes_since_update = 0
            elif buffer_size >= agent.memory.max_size:
                agent.update()
                episodes_since_update = 0
            
            # Logging - log the last num_envs episodes' rewards, lengths, and losses
            if _WANDB_AVAILABLE and wandb.run is not None:
                # Log the last num_envs episodes
                num_episodes_to_log = min(num_envs, len(agent.training_stats['episode_rewards']))
                # print(f"num_episodes_to_log: {num_episodes_to_log}")
                if num_episodes_to_log > 0:
                    # Get the last num_envs episodes
                    start_idx = len(agent.training_stats['episode_rewards']) - num_episodes_to_log
                    
                    for i in range(num_episodes_to_log):
                        idx = start_idx + i
                        episode_num = total_episodes - num_episodes_to_log + i + 1  # Episode number (1-indexed)
                        log_payload = {
                            'episode': episode_num,
                            'episode/reward': agent.training_stats['episode_rewards'][idx],
                            'episode/length': agent.training_stats['episode_lengths'][idx],
                        }
                        
                        # Add loss metrics - log the last num_envs loss values
                        for key in ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss']:
                            if agent.training_stats[key]:
                                num_losses = len(agent.training_stats[key])
                                if num_losses > 0:
                                    # Log the last num_envs loss values (or most recent if fewer available)
                                    loss_start_idx = max(0, num_losses - num_episodes_to_log)
                                    loss_idx = min(loss_start_idx + i, num_losses - 1)
                                    log_payload[f'loss/{key}'] = agent.training_stats[key][loss_idx]
                        
                        # Add learning rate - log the last num_envs learning rate values
                        if agent.training_stats['learning_rate']:
                            num_lrs = len(agent.training_stats['learning_rate'])
                            if num_lrs > 0:
                                # Log the last num_envs learning rate values (or most recent if fewer available)
                                lr_start_idx = max(0, num_lrs - num_episodes_to_log)
                                lr_idx = min(lr_start_idx + i, num_lrs - 1)
                                log_payload['train/learning_rate'] = agent.training_stats['learning_rate'][lr_idx]
                        
                        # Add running averages - calculate for each episode being logged
                        # Calculate average of last 10 episodes up to and including current episode
                        if idx >= 0:
                            # Calculate 10-episode average ending at current episode
                            avg_start_idx = max(0, idx - 9)
                            avg_end_idx = idx + 1
                            if avg_end_idx > avg_start_idx:
                                log_payload['episode/avg_reward_10'] = float(np.mean(agent.training_stats['episode_rewards'][avg_start_idx:avg_end_idx]))
                                log_payload['episode/avg_length_10'] = float(np.mean(agent.training_stats['episode_lengths'][avg_start_idx:avg_end_idx]))
                        
                        wandb.log(log_payload)
            
            # Print progress
            if total_episodes % 10 == 0 and agent.training_stats['episode_rewards']:
                avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                avg_length = np.mean(agent.training_stats['episode_lengths'][-10:]) if len(agent.training_stats['episode_lengths']) >= 10 else 0.0
                print(f"Episode {total_episodes}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Buffer Size: {len(agent.memory)}")
            
            # Save model
            if save_best_only and total_episodes >= save_best_after_episode:
                if agent.training_stats['episode_rewards']:
                    # Use average reward (last 10 episodes) for best model saving
                    if len(agent.training_stats['episode_rewards']) >= 10:
                        avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
                        best_model_path = f"{model_name}_best.pth"
                        agent.save_best_model(avg_reward, best_model_path, episode=total_episodes)
                    else:
                        # If less than 10 episodes, use current episode reward
                        episode_reward = agent.training_stats['episode_rewards'][-1]
                        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
                        best_model_path = f"{model_name}_best.pth"
                        agent.save_best_model(episode_reward, best_model_path, episode=total_episodes)
            
            # Save high-reward models (reward > 310) after specified episode
            # This is independent of save_best_only setting
            if total_episodes >= save_high_reward_after_episode and agent.training_stats['episode_rewards']:
                # Use average reward if available, otherwise use episode reward
                if len(agent.training_stats['episode_rewards']) >= 10:
                    avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                    agent.save_high_reward_model(avg_reward, total_episodes, model_path)
                else:
                    episode_reward = agent.training_stats['episode_rewards'][-1]
                    agent.save_high_reward_model(episode_reward, total_episodes, model_path)
        
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
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.take_step(action)
                
                agent.store_transition(state, action, reward, next_state, done, log_prob, value, env_id=0)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episodes_since_update += 1
            agent.training_stats['episode_rewards'].append(episode_reward)
            agent.training_stats['episode_lengths'].append(episode_length)
            
            # Save high-reward models (reward > 310) after specified episode
            if episode >= save_high_reward_after_episode:
                # Use average reward if available, otherwise use episode reward
                if len(agent.training_stats['episode_rewards']) >= 10:
                    avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
                    agent.save_high_reward_model(avg_reward, episode, model_path)
                else:
                    agent.save_high_reward_model(episode_reward, episode, model_path)
            
            should_update = (
                episodes_since_update >= episodes_per_update and
                len(agent.memory) >= min_buffer_size
            )
            
            if should_update and len(agent.memory) >= 32:
                agent.update()
                episodes_since_update = 0
            elif len(agent.memory) >= agent.memory.max_size:
                agent.update()
                episodes_since_update = 0
        
        env.close()
    
    # Final update
    if len(agent.memory) >= 32:
        print(f"Final update on remaining {len(agent.memory)} transitions")
        agent.update()
    
    # Save final model
    if save_final_model:
        final_models_dir = "saved_final_models"
        os.makedirs(final_models_dir, exist_ok=True)
        model_name = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
        model_basename = os.path.basename(model_name)
        final_model_path = os.path.join(final_models_dir, f"{model_basename}_final.pth")
        agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train optimized PPO on Rubik's Cube environment")
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
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (increased for vectorized)')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between checkpoints')
    parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--use_vectorized', action='store_true', default=True, help='Use vectorized environments')
    parser.add_argument('--use_torch_compile', action='store_true', default=True, help='Use torch.compile()')
    parser.add_argument('--model_path', type=str, default='saved_models/ppo_vec_model.pth', help='Model save path')
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
    parser.add_argument('--save_high_reward_after_episode', type=int, default=90000, help='Start saving high-reward models (reward > 310) after this episode number (default: 0, saves from start)')
    parser.add_argument('--save_final_model', action='store_true', default=True, help='Save the final model after training completes in saved_final_models directory (default: True)')
    parser.add_argument('--project', type=str, default='mujoco-rubiks-ppo-vec', help='wandb project name')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_model', type=str, default=None, help='Path to model checkpoint to load before training')
    args = parser.parse_args()
    
    # Set default for clip_vloss if not provided (defaults to True)
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
            'algorithm': 'PPO-Vec',
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
    action_dim = temp_env.action_dim
    temp_env.close()
    
    # Create optimized agent
    agent = PPOAgentVec(
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
        use_torch_compile=args.use_torch_compile,
        num_envs=args.num_envs if args.use_vectorized else 1,
        max_steps=args.max_steps
    )
    
    # Load model if specified
    if args.load_model:
        agent.load_model(args.load_model)
    
    # Train
    try:
        train_ppo_agent_vec(
            env_fn=make_env,
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
            num_envs=args.num_envs,
            use_vectorized=args.use_vectorized,
            save_high_reward_after_episode=args.save_high_reward_after_episode
        )
    finally:
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

