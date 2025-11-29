"""
Vectorized Environment Wrapper for Parallel Rollouts

This module provides a vectorized environment wrapper that allows running
multiple environment instances in parallel to speed up data collection.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from environment.rubiks_cube import RubiksCubeEnvironment


class VectorizedEnv:
    """
    Vectorized environment wrapper that runs multiple environment instances in parallel.
    
    This provides Nx speedup where N is the number of parallel environments.
    """
    
    def __init__(self, 
                 env_fn,
                 num_envs: int = 8,
                 seed: Optional[int] = None):
        """
        Initialize vectorized environment.
        
        Args:
            env_fn: Function that creates an environment instance
            num_envs: Number of parallel environments to run
            seed: Random seed for environment initialization
        """
        self.num_envs = num_envs
        self.env_fn = env_fn
        
        # Create multiple environment instances
        self.envs: List[RubiksCubeEnvironment] = []
        for i in range(num_envs):
            env = env_fn()
            self.envs.append(env)
        
        # Initialize all environments
        self.states = np.array([env.initialize() for env in self.envs])
        self.dones = np.zeros(num_envs, dtype=bool)
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=int)
        
        # Get state and action dimensions from first environment
        self.state_dim = self.envs[0].state_dim
        self.action_dim = self.envs[0].action_dim
        
    def reset(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Reset environments.
        
        Args:
            indices: List of environment indices to reset. If None, reset all.
            
        Returns:
            Array of initial states with shape (num_envs, state_dim)
        """
        if indices is None:
            indices = range(self.num_envs)
        
        for i in indices:
            self.states[i] = self.envs[i].initialize()
            self.dones[i] = False
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
        
        return self.states.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments in parallel.
        
        Args:
            actions: Array of actions with shape (num_envs, action_dim)
            
        Returns:
            next_states: Array of next states (num_envs, state_dim)
            rewards: Array of rewards (num_envs,)
            dones: Array of done flags (num_envs,)
            infos: List of info dictionaries
        """
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        # Step each environment
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not self.dones[i]:
                next_state, reward, done, info = env.take_step(action)
                self.states[i] = next_state
                self.dones[i] = done
                self.episode_rewards[i] += reward
                self.episode_lengths[i] += 1
                
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            else:
                # Environment already done, return current state
                # AATHIRA : When exactly this case is triggered?
                next_states.append(self.states[i])
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
        
        # Update states array
        self.states = np.array(next_states)
        
        return (
            np.array(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos
        )
    
    def get_episode_stats(self) -> Dict[str, np.ndarray]:
        """Get episode statistics for all environments."""
        return {
            'rewards': self.episode_rewards.copy(),
            'lengths': self.episode_lengths.copy(),
            'dones': self.dones.copy()
        }
    
    def close(self):
        """Close all environment instances."""
        for env in self.envs:
            env.close()

