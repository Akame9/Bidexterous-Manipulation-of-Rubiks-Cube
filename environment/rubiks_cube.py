"""
Rubik's Cube manipulation environment using bidexhands simulation.
This module implements the MDP environment for training agents to manipulate
a Rubik's cube using dual Shadow Hands in MuJoCo.
"""

from collections import deque
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
import time
import math
from typing import Tuple, Dict, Any, Optional, List
import os
import argparse


class RubiksCubeEnvironment:
    """
    Rubik's Cube manipulation environment using bidexhands simulation.
    
    This environment provides:
    - State space: Joint positions, velocities, cube position/orientation, contact forces
    - Action space: Actuator control values for both hands and cube
    - Reward function: Based on cube manipulation success, stability, and efficiency
    - Episode termination: Based on success conditions or time limits
    """
    
    def __init__(self, xml_path="xmls/bidexhands.xml", timestep=0.002, 
                 max_episode_steps=1000, enable_viewer=False, 
                 visualize_collision_boxes=False, workspace_radius=0.8,
                 settle_on_reset=False):
        """
        Initialize the Rubik's Cube environment.
        
        Args:
            xml_path: Path to the MuJoCo XML model file
            timestep: Simulation timestep
            max_episode_steps: Maximum steps per episode
            enable_viewer: Whether to enable visual rendering
            visualize_collision_boxes: Whether to visualize collision boxes and object axes
        """
        self.xml_path = xml_path
        self.timestep = timestep
        self.max_episode_steps = max_episode_steps
        self.enable_viewer = enable_viewer
        self.visualize_collision_boxes = visualize_collision_boxes
        self.workspace_radius = workspace_radius
        self.settle_on_reset = settle_on_reset
        
        # Load MuJoCo model and data
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = timestep
        
        # Initialize viewer if enabled
        self.viewer = None
        if enable_viewer:
            try:
                self.viewer = mjv.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Warning: Could not initialize viewer: {e}")
                self.enable_viewer = False
        
        # Environment state
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        
        # Get actuator information
        self._setup_actuators()
        
        # Contact tracking
        self.contact_history = deque(maxlen=100)
        
        # Initialize state and action spaces
        self._setup_spaces()
        
        # Target cube configuration (for reward calculation)
        self.target_cube_config = self._get_initial_cube_config()
        
        
        
        print(f"Environment initialized with {self.model.nu} total actuators")
        print(f"Hand actuators: {len(self.hand_actuators)}")
        print(f"Cube actuators: {len(self.cube_actuators)} (excluded from action space)")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
    
    def _setup_actuators(self):
        """Setup actuator information and groupings."""
        # Get all actuator names
        self.actuator_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) 
                              for i in range(self.model.nu)]
        
        # Group actuators by hand and cube
        self.left_hand_actuators = []
        self.right_hand_actuators = []
        self.cube_actuators = []
        self.hand_actuators = []  # Combined hand actuators for action space
        
        for i, name in enumerate(self.actuator_names):
            if name.startswith('lh_A_'):
                self.left_hand_actuators.append(i)
                self.hand_actuators.append(i)
            elif name.startswith('rh_A_'):
                self.right_hand_actuators.append(i)
                self.hand_actuators.append(i)
            elif name in ['red', 'orange', 'blue', 'green', 'white', 'yellow']:
                self.cube_actuators.append(i)
                # Note: cube actuators are NOT added to hand_actuators
        
        # Get all cube body IDs (core + all children)
        self.cube_body_ids = set()
        core_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if core_body_id != -1:
            self.cube_body_ids.add(core_body_id)
            # Get all bodies and check if they're cube bodies (not hand bodies)
            for i in range(self.model.nbody):
                body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
                if body_name and not body_name.startswith('lh_') and not body_name.startswith('rh_') and body_name != 'world':
                    self.cube_body_ids.add(i)
        
        print(f"Left hand actuators: {len(self.left_hand_actuators)}")
        print(f"Right hand actuators: {len(self.right_hand_actuators)}")
        print(f"Cube actuators: {len(self.cube_actuators)} (excluded from action space)")
        print(f"Total hand actuators for action space: {len(self.hand_actuators)}")
        print(f"Found {len(self.cube_body_ids)} cube bodies")
    
    def _setup_spaces(self):
        """Setup state and action space dimensions."""
        # State space components:
        # - Joint positions (qpos)
        # - Joint velocities (qvel)
        # - Cube position and orientation
        # - Contact forces
        # - Previous actions
        
        # Aathira : How is 7,6 and contact_history is 3?
        self.state_dim = (
            self.model.nq +  # Joint positions
            self.model.nv +  # Joint velocities
            7 +              # Cube position (3) + quaternion (4)
            6 +              # Cube linear and angular velocities
            3 +              # Aggregate contact force (fx, fy, fz)
            len(self.hand_actuators)  # Previous hand actions only
        )
        
        # Action space: only hand actuator controls (cube actuators excluded)
        self.action_dim = len(self.hand_actuators)
    
    def initialize(self) -> np.ndarray:
        """
        Initialize the environment and return initial state.
        
        Returns:
            Initial state vector
        """
        # Reset MuJoCo simulation
        mj.mj_resetData(self.model, self.data)
        
        # Set initial joint positions (neutral pose)
        
        self._set_neutral_pose()
        # print the actuators
            
        
        # Set initial cube position
        self._set_initial_cube_pose()
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        self.contact_history.clear()
        
        # Optionally settle the simulation by stepping a few times
        if self.settle_on_reset:
            for _ in range(50):
                mj.mj_step(self.model, self.data)
        
        # Get initial state
        initial_state = self.get_state()
        
        return initial_state
    
    def _set_neutral_pose(self):
        """Set both hands to a neutral grasping pose."""
        # Get actuator control ranges
        ctrl_ranges = self.model.actuator_ctrlrange.copy()
        
        # Set neutral positions for each actuator type
        for i, name in enumerate(self.actuator_names):
            lo, hi = ctrl_ranges[i]
            
            if name in {"lh_A_WRJ1", "lh_A_WRJ2", "rh_A_WRJ1", "rh_A_WRJ2"}:
                # Wrist joints - keep neutral
                self.data.ctrl[i] = 0.0
            elif any(k in name for k in ["lh_A_FFJ4", "lh_A_MFJ4", "lh_A_RFJ4", "lh_A_LFJ4",
                                        "rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4"]):
                # Knuckle joints - keep neutral
                self.data.ctrl[i] = 0.0
            elif any(k in name for k in ["lh_A_FFJ3", "lh_A_MFJ3", "lh_A_RFJ3", "lh_A_LFJ3",
                                        "rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"]):
                # Proximal joints - slightly flexed for grasping
                self.data.ctrl[i] = lo + 0.3 * (hi - lo)
            elif name.endswith("J0") and (name.startswith("lh_A_") or name.startswith("rh_A_")):
                # Tendon controls - slightly flexed
                self.data.ctrl[i] = lo + 0.5 * (hi - lo)
            elif name in {"lh_A_THJ4", "lh_A_THJ1", "rh_A_THJ4", "rh_A_THJ1"}:
                # Thumb flexion - slightly flexed
                self.data.ctrl[i] = lo + 0.6 * (hi - lo)
            elif name in {"lh_A_THJ5", "lh_A_THJ3", "lh_A_THJ2", "rh_A_THJ5", "rh_A_THJ3", "rh_A_THJ2"}:
                # Thumb other joints - neutral
                self.data.ctrl[i] = 0.0
            elif name in {"lh_A_LFJ5", "rh_A_LFJ5"}:
                # Metacarpal joints - slightly flexed
                self.data.ctrl[i] = lo + 0.2 * (hi - lo)
            else:
                # Default to neutral
                self.data.ctrl[i] = 0.0
        
        # Run forward kinematics and apply control values to update joint positions
        mj.mj_forward(self.model, self.data)
        
        # Apply control values by running a few simulation steps
        # This is necessary because mj_forward alone doesn't apply actuator controls
        for _ in range(100):
            mj.mj_step(self.model, self.data)
    
    def _set_initial_cube_pose(self):
        """Set initial cube position between the hands at finger level."""
        # Find cube body ID
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            # Position cube between hands at finger level:
            # - X: 0.15 (forward from hands, in line with finger tips)
            # - Y: 0.0 (centered between hands at Y=±0.2)
            # - Z: 0.25 (same height as hands)
            # The hands are at X=0, so moving forward to X=0.15 puts the cube
            # in the natural grasping zone where fingers can reach it
            self.model.body_pos[cube_body_id] = [0.35, 0.0, 0.25]
            
            # Forward kinematics to update world positions
            mj.mj_forward(self.model, self.data)
    
    def get_state(self) -> np.ndarray:
        """
        Get current state of the environment.
        
        Returns:
            State vector containing joint positions, velocities, cube state, and contact info
        """
        # Joint positions and velocities
        joint_pos = self.data.qpos.copy()
        joint_vel = self.data.qvel.copy()
        
        # Cube state (position, orientation, velocities)
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            # Use world-frame body pose and composite velocity
            cube_pos = self.data.xpos[cube_body_id].copy()
            cube_quat = self.data.xquat[cube_body_id].copy()
            cvel = self.data.cvel[cube_body_id].copy()  # [ang(3), lin(3)] in world frame
            cube_ang_vel = cvel[:3]
            cube_lin_vel = cvel[3:]
        else:
            cube_pos = np.zeros(3)
            cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
            cube_lin_vel = np.zeros(3)
            cube_ang_vel = np.zeros(3)
        
        # Contact forces (simplified - use total contact force magnitude)
        contact_force = self._get_contact_forces()
        
        # Previous hand actions only (current control values for hand actuators)
        prev_actions = self.data.ctrl[self.hand_actuators].copy()
        
        # Combine all state components
        state = np.concatenate([
            joint_pos,
            joint_vel,
            cube_pos,
            cube_quat,
            cube_lin_vel,
            cube_ang_vel,
            contact_force,
            prev_actions
        ])
        
        return state.astype(np.float32)
    
    # Aathira : Explain this function?
    def _get_contact_forces(self) -> np.ndarray:
        """Get contact forces between hands and cube only."""
        contact_force = np.zeros(3)  # [fx, fy, fz]
        
        # Aggregate contact forces using MuJoCo API, filtering for hand-cube contacts only
        ncon = self.data.ncon
        hand_cube_contacts = 0
        total_raw_force = 0.0
        
        if ncon > 0:
            efc_force = np.zeros(6)
            for i in range(ncon):
                # Get contact information
                contact = self.data.contact[i]
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                # Get body IDs for the geometries
                body1_id = self.model.geom_bodyid[geom1_id]
                body2_id = self.model.geom_bodyid[geom2_id]
                
                body1_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body2_id)
                
                is_hand_cube_contact = False
                
                # Check if one body is a cube body and the other is a hand body
                body1_is_cube = body1_id in self.cube_body_ids
                body2_is_cube = body2_id in self.cube_body_ids
                body1_is_hand = body1_name and ('lh_' in body1_name or 'rh_' in body1_name)
                body2_is_hand = body2_name and ('lh_' in body2_name or 'rh_' in body2_name)
                
                if (body1_is_cube and body2_is_hand) or (body2_is_cube and body1_is_hand):
                    is_hand_cube_contact = True
                
                # Only accumulate forces from hand-cube contacts
                if is_hand_cube_contact:
                    mj.mj_contactForce(self.model, self.data, i, efc_force)
                    contact_force += efc_force[0:3] * 1.0  # accumulate normal force (fx, fy, fz)
                    hand_cube_contacts += 1
            
        
        return contact_force
    
    def get_action(self, action_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parse action vector into separate actions for each hand.
        
        Args:
            action_vector: Combined action vector (hand actions only)
            
        Returns:
            Dictionary with separate actions for left hand and right hand
        """
        # Map action_vector indices to actual actuator indices
        left_hand_actions = []
        right_hand_actions = []
        
        for i, action_value in enumerate(action_vector):
            actual_actuator_idx = self.hand_actuators[i]
            if actual_actuator_idx in self.left_hand_actuators:
                left_hand_actions.append(action_value)
            elif actual_actuator_idx in self.right_hand_actuators:
                right_hand_actions.append(action_value)
        
        actions = {
            'left_hand': np.array(left_hand_actions),
            'right_hand': np.array(right_hand_actions)
        }
        
        return actions
    
    def take_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action vector for hand actuators only
            
        Returns:
            next_state: Next state vector
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Initialize all controls to zero
        self.data.ctrl[:] = 0.0
        
        # Apply hand actions to the correct actuators
        for i, action_value in enumerate(action):
            actual_actuator_idx = self.hand_actuators[i]
            # Clip action to valid range for this actuator
            ctrl_range = self.model.actuator_ctrlrange[actual_actuator_idx]
            clipped_action = np.clip(action_value, ctrl_range[0], ctrl_range[1])
            self.data.ctrl[actual_actuator_idx] = clipped_action
        
        # Forward kinematics to apply control values
        mj.mj_forward(self.model, self.data)
        
        # Step simulation (single timestep)
        mj.mj_step(self.model, self.data)
        
        # Update contact history
        self._update_contact_history()
        
        # Get next state
        next_state = self.get_state()
        
        # Calculate reward
        reward = self.calculate_reward(action)
        
        # Check if episode is done
        done = self._is_done()
        
        # Update episode info
        self.episode_reward += reward
        self.current_step += 1
        
        info = {
            'episode_reward': self.episode_reward,
            'step': self.current_step,
            'cube_position': self._get_cube_position(),
            'cube_orientation': self._get_cube_orientation(),
            'contact_count': len(self.contact_history)
        }
        if done:
            info['termination_reason'] = self.episode_info.get('termination_reason', 'unknown')
        
        # Update viewer if enabled
        if self.enable_viewer and self.viewer is not None:
            self.viewer.sync()
            time.sleep(0.001)  # Small delay for visualization
        
        # Visualize collision boxes and axes if enabled
        # self.visualize_collision_boxes_and_axes()
        
        return next_state, reward, done, info
    
    
    def _update_contact_history(self):
        """Update contact force history."""
        contact_force = self._get_contact_forces()
        self.contact_history.append(contact_force)
    
    def calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward for current state and action.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # 1. Grasping reward (based on contact forces) - PRIORITY
        grasp_reward = self._calculate_grasp_reward()
        reward += grasp_reward #* 0.5 #0.4
        
        # 2. Manipulation reward (based on cube rotation) - PRIORITY
        # manipulation_reward = self._calculate_manipulation_reward()
        # reward += manipulation_reward * 0.5 #0.3
        
        # 3. Cube manipulation reward (based on cube movement and stability)
        # cube_reward = self._calculate_cube_reward()
        # reward += cube_reward * 0.2
        
        # 4. Efficiency reward (penalize excessive actions)
        # efficiency_reward = self._calculate_efficiency_reward(action)
        # reward += efficiency_reward * 0.05
        
        # 5. Stability reward (penalize unstable configurations)
        # stability_reward = self._calculate_stability_reward()
        # reward += stability_reward * 0.05
        
        return reward
    
    def _calculate_cube_reward(self) -> float:
        """Calculate reward based on cube manipulation."""
        cube_pos = self._get_cube_position()
        cube_vel = self._get_cube_velocity()
        
        # Reward for keeping cube in workspace
        workspace_penalty = 0.0
        if np.linalg.norm(cube_pos[:2]) > max(0.1, 0.5 * self.workspace_radius):  # Keep cube reasonably centered
            workspace_penalty = -1.0
        
        # Reward for controlled movement (not too fast)
        velocity_penalty = -np.linalg.norm(cube_vel) * 0.1
        
        return workspace_penalty + velocity_penalty
    
    def _calculate_grasp_reward(self) -> float:
        """Calculate reward based on grasping quality."""
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        # Enhanced grasping reward for learning to grab the cube
        if force_magnitude > 0.05:  # Any contact is good
            # Reward increases with contact up to optimal range
            if force_magnitude < 1.0:
                # print(f"GRASP REWARD: Very light grasp (force={force_magnitude:.3f}) -> +0.5")
                return 0.0 
            elif force_magnitude < 3.0:
                # print(f"GRASP REWARD: Gentle grasp (force={force_magnitude:.3f}) -> +2.0")
                return 2.0  # Strong reward for gentle but firm grasp
            elif force_magnitude < 5.0:
                # print(f"GRASP REWARD: Moderate grasp (force={force_magnitude:.3f}) -> +1.0")
                return 1.0  
            else:
                # print(f"GRASP REWARD: Excessive force (force={force_magnitude:.3f}) -> +0.5")
                return 0.5  # Reduced reward for excessive force
        else:
            # print(f"GRASP REWARD: No contact (force={force_magnitude:.3f}) -> -0.5")
            return -0.5  # Penalty for no contact (encourage grasping)
    
    def _calculate_manipulation_reward(self) -> float:
        """Calculate reward for cube manipulation success."""
        # Get cube orientation and angular velocity
        cube_quat = self._get_cube_orientation()
        cube_ang_vel = self._get_cube_angular_velocity()
        
        # Reward for cube rotation (any rotation from initial state)
        initial_quat = self.target_cube_config['orientation']
        
        # Calculate rotation angle from initial orientation
        quat_dot = np.abs(np.dot(cube_quat, initial_quat))
        rotation_angle = 2 * np.arccos(np.clip(quat_dot, 0, 1))
        
        # Base rotation reward (encourage any rotation)
        rotation_reward = min(rotation_angle / np.pi, 1.0) * 1.0
        print(f"MANIPULATION REWARD: Base rotation (angle={rotation_angle:.3f} rad) -> +{rotation_reward:.3f}")
        
        # Bonus reward for active rotation (angular velocity)
        angular_velocity_magnitude = np.linalg.norm(cube_ang_vel)
        if angular_velocity_magnitude > 0.1:  # If cube is actively rotating
            velocity_bonus = min(angular_velocity_magnitude, 2.0) * 0.5
            rotation_reward += velocity_bonus
            print(f"MANIPULATION REWARD: Active rotation (ang_vel={angular_velocity_magnitude:.3f}) -> +{velocity_bonus:.3f}")
        else:
            print(f"MANIPULATION REWARD: No active rotation (ang_vel={angular_velocity_magnitude:.3f}) -> +0.0")
        
        # Check if we have contact while rotating (good manipulation)
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        if force_magnitude > 0.1 and angular_velocity_magnitude > 0.05:
            # Bonus for rotating while maintaining grasp
            rotation_reward += 1.0
            print(f"MANIPULATION REWARD: Grasp while rotating (force={force_magnitude:.3f}, ang_vel={angular_velocity_magnitude:.3f}) -> +1.0")
        else:
            print(f"MANIPULATION REWARD: No grasp while rotating (force={force_magnitude:.3f}, ang_vel={angular_velocity_magnitude:.3f}) -> +0.0")
        
        print(f"MANIPULATION REWARD: Total -> +{rotation_reward:.3f}")
        return rotation_reward
    
    def _calculate_efficiency_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on action efficiency."""
        # Penalize large actions
        action_penalty = -np.linalg.norm(action) * 0.01
        return action_penalty
    
    def _calculate_stability_reward(self) -> float:
        """Calculate reward based on system stability."""
        joint_vel = self.data.qvel.copy()
        velocity_penalty = -np.linalg.norm(joint_vel) * 0.01
        return velocity_penalty
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Episode length limit
        if self.current_step >= self.max_episode_steps:
            self.episode_info['termination_reason'] = 'max_steps_reached'
            return True
        
        # Cube dropped (fell too low)
        cube_pos = self._get_cube_position()
        if cube_pos[2] < 0.05:  # Cube below 5cm
            self.episode_info['termination_reason'] = 'cube_dropped'
            return True
        
        # Cube moved too far from workspace
        if np.linalg.norm(cube_pos[:2]) > self.workspace_radius:  # Cube moved too far from center
            self.episode_info['termination_reason'] = 'cube_out_of_workspace'
            return True
        
        return False
    
    def _get_cube_position(self) -> np.ndarray:
        """Get current cube position."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.xpos[cube_body_id].copy()
        return np.zeros(3)
    
    def _get_cube_orientation(self) -> np.ndarray:
        """Get current cube orientation (quaternion)."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.xquat[cube_body_id].copy()
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _get_cube_velocity(self) -> np.ndarray:
        """Get current cube velocity."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.cvel[cube_body_id][3:].copy()
        return np.zeros(3)
    
    def _get_cube_angular_velocity(self) -> np.ndarray:
        """Get current cube angular velocity."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.cvel[cube_body_id][:3].copy()  # First 3 elements are angular velocity
        return np.zeros(3)
    
    def _get_initial_cube_config(self) -> Dict[str, Any]:
        """Get initial cube configuration for reference."""
        return {
            'position': self._get_cube_position(),
            'orientation': self._get_cube_orientation()
        }
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        return self.initialize()
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
    
    def render(self, mode='human'):
        """Render the environment."""
        if self.enable_viewer and self.viewer is not None:
            self.viewer.sync()
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            mj.mj_resetData(self.model, self.data)


# Example usage and testing
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Rubik\'s Cube Environment')
    parser.add_argument('--enable-viewer', action='store_true', 
                       help='Enable MuJoCo viewer (requires display)')
    parser.add_argument('--visualize-collision', action='store_true',
                       help='Enable collision box visualization')
    parser.add_argument('--xml-path', type=str, default='xmls/bidexhands.xml',
                       help='Path to MuJoCo XML file')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum steps per episode')
    parser.add_argument('--settle-on-reset', action='store_true',
                       help='Settle simulation on reset')
    
    args = parser.parse_args()
    
    # Create environment
    env = RubiksCubeEnvironment(
        xml_path=args.xml_path,
        enable_viewer=args.enable_viewer,
        max_episode_steps=args.max_steps,
        visualize_collision_boxes=args.visualize_collision,
        settle_on_reset=args.settle_on_reset,
    )
    
    print("Environment created successfully!")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    
    # Initialize without stepping, keep neutral pose, and visualize
    state = env.initialize()
    print(f"Initial state shape: {state.shape}")

    # Idle visualization loop (no physics steps, no actions applied)
    try:
        print("Viewer running. Close the window or press Ctrl+C to exit.")
        while True:
            if env.enable_viewer and env.viewer is not None:
                env.viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nViewer closed by user.")
    except Exception as e:
        print(f"Viewer ended due to error: {e}")
    
    env.close()
    print("Environment test completed!")
