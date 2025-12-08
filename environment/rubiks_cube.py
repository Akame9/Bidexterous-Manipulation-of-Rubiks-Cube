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
                 max_episode_steps=500, enable_viewer=False, 
                 visualize_collision_boxes=False, workspace_radius=0.8,
                 settle_on_reset=False, enable_gravity=False,
                 gravity_vector=None, rotation_sequence=None):
        """
        Initialize the Rubik's Cube environment.
        
        Args:
            xml_path: Path to the MuJoCo XML model file
            timestep: Simulation timestep
            max_episode_steps: Maximum steps per episode
            enable_viewer: Whether to enable visual rendering
            visualize_collision_boxes: Whether to visualize collision boxes and object axes
            enable_gravity: Whether to override the model gravity during initialization
            gravity_vector: Gravity vector to apply when enable_gravity is True
            rotation_sequence: List of face names to rotate (e.g., ['red', 'blue', 'white'])
                              Each face name should be one of: 'red', 'orange', 'blue', 'green', 'white', 'yellow'
        """
        self.xml_path = xml_path
        self.timestep = timestep
        self.max_episode_steps = max_episode_steps
        self.enable_viewer = enable_viewer
        self.visualize_collision_boxes = visualize_collision_boxes
        self.workspace_radius = workspace_radius
        self.settle_on_reset = settle_on_reset
        self.enable_gravity = enable_gravity
        if gravity_vector is None:
            self.gravity_vector = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            gravity_array = np.array(gravity_vector, dtype=np.float64)
            if gravity_array.shape != (3,):
                raise ValueError("gravity_vector must be an iterable with three elements (x, y, z).")
            self.gravity_vector = gravity_array
        
        # Load MuJoCo model and data
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = timestep
        self._apply_gravity_override()
        
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
        
        # Initial cube position (stored during initialize() for displacement reward)
        self.initial_cube_position = None
        
        # Rotation sequence tracking
        # rotation_sequence stores tuples of (face_name, direction) where direction is +1 for clockwise (+90°) or -1 for anti-clockwise (-90°)
        self.rotation_sequence = []
        if rotation_sequence is not None:
            self._parse_rotation_sequence(rotation_sequence)
        self.current_rotation_index = 0  # Index of current face in sequence
        self.rotation_started = False  # Whether current rotation has started
        self.rotation_completed = False  # Whether current rotation has completed
        self.initial_joint_angle = None  # Joint angle when rotation started
        self.rotation_angle_accumulated = 0.0  # Accumulated rotation angle
        self.rotation_direction_actual = 0.0  # Actual rotation direction (positive or negative)
        self.rotation_start_threshold = 0.1  # Angular velocity threshold to detect rotation start (rad/s)
        self.rotation_complete_threshold = np.pi / 2.0  # 90 degrees for complete rotation
        self.face_initial_angles = {}  # Store initial angles for all faces when rotation starts (for wrong face penalty)
        self.wrong_face_rotation_penalty_coef = 0.5  # Coefficient for wrong face rotation penalty
        # Map face names to joint names (from cube_rad.xml structure)
        self.face_to_joint_map = {
            'red': 'pX',      # Red face uses pX joint (positive X axis)
            'orange': 'nX',   # Orange face uses nX joint (negative X axis)
            'blue': 'pY',     # Blue face uses pY joint (positive Y axis)
            'green': 'nY',    # Green face uses nY joint (negative Y axis)
            'white': 'pZ',    # White face uses pZ joint (positive Z axis)
            'yellow': 'nZ'    # Yellow face uses nZ joint (negative Z axis)
        }
        # Cache joint IDs for faster lookup
        self.face_joint_ids = {}
        self._initialize_face_joint_ids()
        self.rotation_rewards_given = set()  # Track which rotations have been rewarded
        
        print(f"Environment initialized with {self.model.nu} total actuators")
        print(f"Hand actuators: {len(self.hand_actuators)}")
        print(f"Cube actuators: {len(self.cube_actuators)} (excluded from action space)")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
    
    def _initialize_face_joint_ids(self):
        """Initialize joint IDs for each face for faster lookup."""
        for face_name, joint_name in self.face_to_joint_map.items():
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                self.face_joint_ids[face_name] = joint_id
            else:
                print(f"Warning: Joint '{joint_name}' for face '{face_name}' not found in model")
    
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
        
        # Get all fingertip body IDs (distal phalanges only)
        self.fingertip_body_ids = set()
        fingertip_suffixes = ['ffdistal', 'mfdistal', 'rfdistal', 'lfdistal', 'thdistal']
        for i in range(self.model.nbody):
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            if body_name:
                # Check if body name ends with any fingertip suffix
                for suffix in fingertip_suffixes:
                    if body_name.endswith(suffix):
                        self.fingertip_body_ids.add(i)
                        break
        
        # Get palm body IDs (left and right palm)
        self.palm_body_ids = set()
        palm_names = ['lh_palm', 'rh_palm']
        for palm_name in palm_names:
            palm_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, palm_name)
            if palm_id != -1:
                self.palm_body_ids.add(palm_id)
            else:
                print(f"Warning: Palm body '{palm_name}' not found in model")
        
        print(f"Left hand actuators: {len(self.left_hand_actuators)}")
        print(f"Right hand actuators: {len(self.right_hand_actuators)}")
        print(f"Cube actuators: {len(self.cube_actuators)} (excluded from action space)")
        print(f"Total hand actuators for action space: {len(self.hand_actuators)}")
        print(f"Found {len(self.cube_body_ids)} cube bodies")
        print(f"Found {len(self.fingertip_body_ids)} fingertip bodies")
        print(f"Found {len(self.palm_body_ids)} palm bodies")
    
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
        self._apply_gravity_override()
        
        # Set initial joint positions (neutral pose)
        
        self._set_neutral_pose()
        # print the actuators
            
        
        # Set initial cube position
        self._set_initial_cube_pose()
        
        # Store initial cube position for displacement reward calculation
        self.initial_cube_position = self._get_cube_position().copy()
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info = {}
        self.contact_history.clear()
        
        # Reset rotation tracking
        self.current_rotation_index = 0
        self.rotation_started = False
        self.rotation_completed = False
        self.initial_joint_angle = None
        self.rotation_angle_accumulated = 0.0
        self.rotation_direction_actual = 0.0
        self.rotation_rewards_given.clear()
        
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
        target_position = np.array([0.35, 0.0, 0.25], dtype=np.float64)
        if cube_body_id != -1:
            cube_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "cube_free")
            if cube_joint_id != -1:
                # For a free joint, qpos stores [x, y, z, quat_w, quat_x, quat_y, quat_z]
                qpos_adr = self.model.jnt_qposadr[cube_joint_id]
                self.data.qpos[qpos_adr:qpos_adr + 7] = np.concatenate(
                    [target_position, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)]
                )
                # Reset joint velocity to zero to avoid residual motion
                qvel_adr = self.model.jnt_dofadr[cube_joint_id]
                self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
            else:
                # Fallback for legacy models without named free joint
                self.model.body_pos[cube_body_id] = target_position
            
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
    
    def _apply_gravity_override(self):
        """Override the environment gravity if requested."""
        if not self.enable_gravity:
            return
        self.model.opt.gravity[:] = self.gravity_vector
        # Update forward dynamics to account for gravity change
        mj.mj_forward(self.model, self.data)
    
    # Aathira : Explain this function?
    def _get_contact_forces(self) -> np.ndarray:
        """Get contact forces between fingertips and cube only."""
        contact_force = np.zeros(3)  # [fx, fy, fz]
        
        # Aggregate contact forces using MuJoCo API, filtering for fingertip-cube contacts only
        ncon = self.data.ncon
        fingertip_cube_contacts = 0
        
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
                
                is_fingertip_cube_contact = False
                
                # Check if one body is a cube body and the other is a fingertip body
                body1_is_cube = body1_id in self.cube_body_ids
                body2_is_cube = body2_id in self.cube_body_ids
                body1_is_fingertip = body1_id in self.fingertip_body_ids
                body2_is_fingertip = body2_id in self.fingertip_body_ids
                
                if (body1_is_cube and body2_is_fingertip) or (body2_is_cube and body1_is_fingertip):
                    is_fingertip_cube_contact = True
                
                # Only accumulate forces from fingertip-cube contacts
                if is_fingertip_cube_contact:
                    mj.mj_contactForce(self.model, self.data, i, efc_force)
                    contact_force += efc_force[0:3]  # accumulate normal force (fx, fy, fz)
                    fingertip_cube_contacts += 1
            
        
        return contact_force
    
    def _get_palm_contact_forces(self) -> float:
        """
        Get contact forces between palms and cube.
        
        Returns:
            Force magnitude of palm-cube contacts (scalar)
        """
        palm_contact_force_magnitude = 0.0
        
        # Check for palm-cube contacts using MuJoCo API
        ncon = self.data.ncon
        
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
                
                is_palm_cube_contact = False
                
                # Check if one body is a cube body and the other is a palm body
                body1_is_cube = body1_id in self.cube_body_ids
                body2_is_cube = body2_id in self.cube_body_ids
                body1_is_palm = body1_id in self.palm_body_ids
                body2_is_palm = body2_id in self.palm_body_ids
                
                if (body1_is_cube and body2_is_palm) or (body2_is_cube and body1_is_palm):
                    is_palm_cube_contact = True
                
                # Accumulate forces from palm-cube contacts
                if is_palm_cube_contact:
                    mj.mj_contactForce(self.model, self.data, i, efc_force)
                    contact_force = efc_force[0:3]  # Normal force (fx, fy, fz)
                    force_magnitude = np.linalg.norm(contact_force)
                    palm_contact_force_magnitude += force_magnitude
        
        return palm_contact_force_magnitude
    
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
        self.current_step += 1
        done = self._is_done()
        
        # Update episode info
        self.episode_reward += reward
        
        
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
        
        #1. Grasping reward (based on contact forces) - PRIORITY
        # grasp_reward = self._calculate_grasp_reward()
        # reward += grasp_reward #* 0.5 #0.4
        
        # # 2. Palm contact penalty (penalize palm-cube contact) - PRIORITY
        # palm_penalty = self._calculate_palm_penalty()
        # reward += palm_penalty
        
        # 3. Displacement reward (based on cube displacement) - PRIORITY
        # displacement_reward = self._calculate_displacement_reward()
        # reward += displacement_reward

        # 3. Rotation sequence reward (based on face rotation sequence) - PRIORITY
        rotation_reward = self._calculate_rotation_reward_v3()
        reward += rotation_reward
        
        # 4. Manipulation reward (based on cube rotation) - PRIORITY
        # manipulation_reward = self._calculate_manipulation_reward()
        # reward += manipulation_reward * 0.5 #0.3
        
        # 5. Cube manipulation reward (based on cube movement and stability)
        # cube_reward = self._calculate_cube_reward()
        # reward += cube_reward * 0.2
        
        # 6. Efficiency reward (penalize excessive actions)
        # efficiency_reward = self._calculate_efficiency_reward(action)
        # reward += efficiency_reward * 0.05
        
        # 7. Stability reward (penalize unstable configurations)
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
            # elif force_magnitude < 3.0:
            #     # print(f"GRASP REWARD: Gentle grasp (force={force_magnitude:.3f}) -> +2.0")
            #     return 2.0  # Strong reward for gentle but firm grasp
            elif force_magnitude < 30.0:
                # print(f"GRASP REWARD: Moderate grasp (force={force_magnitude:.3f}) -> +1.0")
                return 1.0  
            else:
                print(f"GRASP REWARD: Excessive force (force={force_magnitude:.3f}) -> +0.5")
            return 0.5  # Reduced reward for excessive force
        else:
            # print(f"GRASP REWARD: No contact (force={force_magnitude:.3f}) -> -0.5")
            return -0.5  # Penalty for no contact (encourage grasping)
    
    def _calculate_palm_penalty(self) -> float:
        """
        Calculate penalty for palm-cube contact.
        Palms should not contact the cube - only fingertips should.
        
        Returns:
            Negative penalty value (penalty increases with palm contact force)
        """
        palm_contact_force = self._get_palm_contact_forces()
        
        if palm_contact_force > 0.01:  # Small threshold to ignore noise
            # Penalty proportional to palm contact force
            # Higher force = larger penalty
            penalty = -palm_contact_force * 0.5  # Scale penalty
            # Cap the penalty to avoid extreme values
            penalty = max(penalty, -5.0)
            # print(f"PALM PENALTY: Palm-cube contact (force={palm_contact_force:.3f}) -> {penalty:.3f}")
            return penalty
        else:
            # No palm contact - no penalty
            return 0.0
    
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
    
    def _calculate_displacement_reward(self) -> float:
        """
        Calculate penalty based on how far the cube is displaced from its original initial position.
        
        Returns:
            Negative penalty value (penalty increases with displacement distance)
        """
        # Get current cube position
        current_pos = self._get_cube_position()
        
        # Check if initial position has been set
        if self.initial_cube_position is None:
            return 0.0
        
        # Calculate Euclidean distance from initial position
        displacement = np.linalg.norm(current_pos - self.initial_cube_position)
        
        # Calculate penalty proportional to displacement
        # Penalty increases with distance from initial position
        penalty = -displacement
        
        return penalty
    
    def _get_face_joint_angle(self, face_name: str) -> float:
        """
        Get the current joint angle for a specific face.
        
        Args:
            face_name: Name of the face ('red', 'orange', 'blue', 'green', 'white', 'yellow')
            
        Returns:
            Joint angle in radians, or 0.0 if joint not found
        """
        if face_name not in self.face_joint_ids:
            return 0.0
        
        joint_id = self.face_joint_ids[face_name]
        qpos_adr = self.model.jnt_qposadr[joint_id]
        # For hinge joints, qpos stores the angle directly
        # AATHIRA: How to test this?
        return float(self.data.qpos[qpos_adr])
    
    def _get_face_joint_velocity(self, face_name: str) -> float:
        """
        Get the current joint angular velocity for a specific face.
        
        Args:
            face_name: Name of the face ('red', 'orange', 'blue', 'green', 'white', 'yellow')
            
        Returns:
            Joint angular velocity in rad/s, or 0.0 if joint not found
        """
        if face_name not in self.face_joint_ids:
            return 0.0
        
        joint_id = self.face_joint_ids[face_name]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        # For hinge joints, qvel stores the angular velocity directly
        # AATHIRA: How to test this?
        return float(self.data.qvel[qvel_adr])
    
    def _calculate_rotation_reward(self) -> float:
        """
        Calculate reward for rotation sequence completion.
        Tracks individual face joint rotations, not the entire cube body rotation.
        
        Returns:
            Reward value for rotation progress
        """
        # If no rotation sequence is set, return 0
        if not self.rotation_sequence or self.current_rotation_index >= len(self.rotation_sequence):
            # print(f"[ROTATION_REWARD] Case: No rotation sequence or sequence completed")
            return 0.0
        
        # Extract face name and desired direction from tuple
        current_face, desired_direction = self.rotation_sequence[self.current_rotation_index]
        reward = 0.0
        
        # Check if joint exists for this face
        if current_face not in self.face_joint_ids:
            # print(f"[ROTATION_REWARD] Case: Invalid face '{current_face}' - joint not found")
            return 0.0
        
        # Get current face joint state
        current_joint_angle = self._get_face_joint_angle(current_face)
        current_joint_velocity = self._get_face_joint_velocity(current_face)
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        # Check if rotation has started (hands are rotating the face)
        if not self.rotation_started:
            # Rotation starts when there's significant joint angular velocity AND contact
            if abs(current_joint_velocity) > self.rotation_start_threshold and force_magnitude > 0.5:
                self.rotation_started = True
                self.initial_joint_angle = current_joint_angle
                self.rotation_angle_accumulated = 0.0
                # Initialize angles for all faces to track wrong face rotations
                if not self.face_initial_angles:
                    for face_name in self.face_joint_ids.keys():
                        self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
                # Give reward for starting rotation
                reward += 10.0
                rotation_key = f"{current_face}_start_{self.current_rotation_index}"
                if rotation_key not in self.rotation_rewards_given:
                    self.rotation_rewards_given.add(rotation_key)
                    # print(f"[ROTATION_REWARD] Case: ROTATION_START | Face: {current_face} | Index: {self.current_rotation_index} | Reward: +1.0 | Velocity: {abs(current_joint_velocity):.3f} rad/s | Force: {force_magnitude:.3f}")
                else:
                    pass
                    # print(f"[ROTATION_REWARD] Case: ROTATION_START (already rewarded) | Face: {current_face} | Index: {self.current_rotation_index} | Reward: 0.0")
            else:
                pass
                # Conditions not met to start rotation
                # print(f"[ROTATION_REWARD] Case: ROTATION_NOT_STARTED | Face: {current_face} | Index: {self.current_rotation_index} | Velocity: {abs(current_joint_velocity):.3f} (need >{self.rotation_start_threshold}) | Force: {force_magnitude:.3f} (need >0.5) | Reward: 0.0")
        else:
            pass
            # If rotation started but stopped (low angular velocity for a while), allow restart
            # This allows the agent to try again if rotation was interrupted
            # if abs(current_joint_velocity) < self.rotation_start_threshold * 0.5 and force_magnitude < 0.3:
            #     # Rotation has stopped, but don't reset immediately - give it a chance to continue
            #     # Only reset if we've accumulated very little rotation
            #     if abs(self.rotation_angle_accumulated) < 0.1:  # Less than ~6 degrees
            #         # print(f"[ROTATION_REWARD] Case: ROTATION_RESET | Face: {current_face} | Index: {self.current_rotation_index} | Accumulated angle too small: {abs(self.rotation_angle_accumulated):.3f} rad | Resetting rotation state")
            #         self.rotation_started = False
            #         self.initial_joint_angle = None
            #         self.rotation_angle_accumulated = 0.0
            #         self.rotation_direction_actual = 0.0
            #         # Reset initial angles when rotation is reset
            #         self.face_initial_angles = {}
        
        # If rotation has started, track progress
        if self.rotation_started and not self.rotation_completed:
            # Calculate rotation angle from initial joint angle
            if self.initial_joint_angle is not None:
                # Calculate the change in joint angle
                angle_change = current_joint_angle - self.initial_joint_angle
                
                # Handle angle wrapping (joint angles can wrap around ±π)
                # Normalize to [-π, π] range
                # AATHIRA: Understand this?
                while angle_change > np.pi:
                    angle_change -= 2 * np.pi
                while angle_change < -np.pi:
                    angle_change += 2 * np.pi
                
                # Track actual rotation direction (sign of angle_change)
                # Positive angle_change means clockwise rotation, negative means anti-clockwise
                actual_direction = 1 if angle_change > 0 else -1 if angle_change < 0 else 0
                self.rotation_direction_actual = actual_direction
                
                # Check if rotation is in the correct direction
                direction_correct = (actual_direction == desired_direction) if actual_direction != 0 else False
                
                # Only accumulate angle if rotating in the correct direction
                # If rotating in wrong direction, don't accumulate (or accumulate negatively as penalty)
                if direction_correct:
                    # Accumulate the absolute rotation angle only if direction is correct
                    self.rotation_angle_accumulated = abs(angle_change)
                else:
                    # Rotating in wrong direction - reset accumulated angle or penalize
                    # Don't accumulate progress if going in wrong direction
                    self.rotation_angle_accumulated = 0.0
                    # Could add a small penalty here if desired
                    # reward -= 0.01  # Small penalty for wrong direction
                
                # Check if rotation is complete (90 degrees = π/2) AND in correct direction
                if self.rotation_angle_accumulated >= self.rotation_complete_threshold and direction_correct:
                    self.rotation_completed = True
                    # Big reward for completing a rotation in the correct direction
                    reward += 100.0
                    rotation_key = f"{current_face}_{'clock' if desired_direction > 0 else 'anti_clock'}_complete_{self.current_rotation_index}"
                    if rotation_key not in self.rotation_rewards_given:
                        self.rotation_rewards_given.add(rotation_key)
                    
                    # Move to next rotation in sequence
                    self.current_rotation_index += 1
                    if self.current_rotation_index < len(self.rotation_sequence):
                        # Reset for next rotation
                        next_face, next_dir = self.rotation_sequence[self.current_rotation_index]
                        # print(f"[ROTATION_REWARD] Case: ROTATION_COMPLETE | Face: {current_face} | Direction: {'clock' if desired_direction > 0 else 'anti_clock'} | Index: {self.current_rotation_index-1} | Angle: {np.degrees(self.rotation_angle_accumulated):.1f}° | Reward: +10.0 | Next: {next_face}_{'clock' if next_dir > 0 else 'anti_clock'}")
                        self.rotation_started = False
                        self.rotation_completed = False
                        self.initial_joint_angle = None
                        self.rotation_angle_accumulated = 0.0
                        self.rotation_direction_actual = 0.0
                        # Reset initial angles for all faces to track wrong rotations for next face
                        self.face_initial_angles = {}
                        for face_name in self.face_joint_ids.keys():
                            self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
                    else:
                        # Sequence completed! Give big bonus reward
                        reward += 50.0
                        sequence_key = "sequence_complete"
                        if sequence_key not in self.rotation_rewards_given:
                            self.rotation_rewards_given.add(sequence_key)
                            # print(f"[ROTATION_REWARD] Case: SEQUENCE_COMPLETE | Face: {current_face} | Direction: {'clock' if desired_direction > 0 else 'anti_clock'} | Index: {self.current_rotation_index-1} | Angle: {np.degrees(self.rotation_angle_accumulated):.1f}° | Reward: +10.0 (rotation) +50.0 (sequence) = +60.0 | TOTAL: {reward:.2f}")
                        else:
                            pass
                            # print(f"[ROTATION_REWARD] Case: ROTATION_COMPLETE (sequence already rewarded) | Face: {current_face} | Index: {self.current_rotation_index-1} | Reward: +10.0")
                elif direction_correct:
                    # Small continuous reward for making progress in correct direction
                    progress = self.rotation_angle_accumulated / self.rotation_complete_threshold
                    progress_reward = 10 * progress #0.1
                    reward += progress_reward
                    # print(f"[ROTATION_REWARD] Case: ROTATION_PROGRESS | Face: {current_face} | Direction: {'clock' if desired_direction > 0 else 'anti_clock'} | Index: {self.current_rotation_index} | Angle: {np.degrees(self.rotation_angle_accumulated):.1f}° / {np.degrees(self.rotation_complete_threshold):.1f}° | Progress: {progress*100:.1f}% | Reward: +{progress_reward:.3f}")
                # else: rotating in wrong direction, no reward
        
        # Calculate penalty for wrong face rotations
        # wrong_face_penalty = self._calculate_wrong_face_rotation_penalty(current_face)
        # reward -= wrong_face_penalty
        
        return reward
    
    def _calculate_rotation_reward_v2(self) -> float:
        """
        Calculate reward for rotation sequence completion using continuous penalty approach.
        Gives a continuous penalty based on how far the face rotation is from the desired 90-degree rotation.
        
        Returns:
            Reward value (negative penalty) for rotation progress
        """
        # If no rotation sequence is set, return 0
        if not self.rotation_sequence or self.current_rotation_index >= len(self.rotation_sequence):
            return 0.0
        
        # Extract face name and desired direction from tuple
        current_face, desired_direction = self.rotation_sequence[self.current_rotation_index]
        
        # Check if joint exists for this face
        if current_face not in self.face_joint_ids:
            return 0.0
        
        # Get current face joint state
        current_joint_angle = self._get_face_joint_angle(current_face)
        current_joint_velocity = self._get_face_joint_velocity(current_face)
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        # Initialize reference angle when rotation starts
        if not self.rotation_started:
            # Rotation starts when there's significant joint angular velocity AND contact
            if abs(current_joint_velocity) > self.rotation_start_threshold and force_magnitude > 0.5:
                self.rotation_started = True
                self.initial_joint_angle = current_joint_angle
                self.rotation_angle_accumulated = 0.0
                # Initialize angles for all faces to track wrong face rotations
                if not self.face_initial_angles:
                    for face_name in self.face_joint_ids.keys():
                        self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
            else:
                # No rotation started yet - return small penalty to encourage starting
                return -0.1
        
        # If rotation has started, calculate continuous penalty
        if self.rotation_started and not self.rotation_completed:
            if self.initial_joint_angle is not None:
                # Calculate the change in joint angle from initial position
                angle_change = current_joint_angle - self.initial_joint_angle
                
                # Handle angle wrapping (joint angles can wrap around ±π)
                # Normalize to [-π, π] range
                while angle_change > np.pi:
                    angle_change -= 2 * np.pi
                while angle_change < -np.pi:
                    angle_change += 2 * np.pi
                
                # Calculate rotation angle in the desired direction
                target_angle = self.rotation_complete_threshold  # π/2 radians (90 degrees)
                
                # Determine if we're rotating in the correct direction
                actual_direction = 1 if angle_change > 0 else -1 if angle_change < 0 else 0
                is_correct_direction = (actual_direction == desired_direction) if actual_direction != 0 else False
                
                # Calculate the current rotation angle (absolute value)
                rotation_angle = abs(angle_change)
                
                # Calculate distance from target (90 degrees)
                if is_correct_direction:
                    # Rotating in correct direction: distance is how far we are from 90 degrees
                    distance_from_target = abs(target_angle - rotation_angle)
                else:
                    # Rotating in wrong direction: we need to reverse and then rotate 90 degrees
                    # Distance = current rotation + target rotation = rotation_angle + target_angle
                    distance_from_target = rotation_angle + target_angle
                
                # Continuous penalty: proportional to distance from target
                # Penalty increases as we get further from 90 degrees
                # Normalize by target_angle to get penalty in [0, 2] range (can be >1 if wrong direction)
                normalized_distance = distance_from_target / target_angle
                penalty = -normalized_distance  # Negative because it's a penalty
                
                # Check if rotation is complete (within small threshold of 90 degrees AND correct direction)
                if rotation_angle >= target_angle * 0.95 and is_correct_direction:
                    self.rotation_completed = True
                    # Zero penalty for completion
                    penalty = 0.0
                    
                    # Move to next rotation in sequence
                    self.current_rotation_index += 1
                    if self.current_rotation_index < len(self.rotation_sequence):
                        # Reset for next rotation
                        self.rotation_started = False
                        self.rotation_completed = False
                        self.initial_joint_angle = None
                        self.rotation_angle_accumulated = 0.0
                        self.rotation_direction_actual = 0.0
                        # Reset initial angles for all faces
                        self.face_initial_angles = {}
                        for face_name in self.face_joint_ids.keys():
                            self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
                
                return penalty
            else:
                # Initial angle not set - return small penalty
                return -0.1
        elif self.rotation_completed:
            # Rotation completed - no penalty
            return 0.0
        else:
            # Rotation not started - small penalty to encourage starting
            return -0.1
    
    def _calculate_rotation_reward_v3(self) -> float:
        """
        Calculate reward for rotation sequence completion (v3).
        - Gives a positive reward when rotation starts
        - Once rotation starts, gives continuous penalty based on distance from desired rotation angle (90 degrees)
        - Does not handle rotation stopping and restarting (assumes rotation continues once started)
        
        Returns:
            Reward value (positive for start, negative penalty during rotation based on distance)
        """
        # If no rotation sequence is set, return 0
        if not self.rotation_sequence or self.current_rotation_index >= len(self.rotation_sequence):
            return 0.0
        
        # Extract face name and desired direction from tuple
        current_face, desired_direction = self.rotation_sequence[self.current_rotation_index]
        
        # Check if joint exists for this face
        if current_face not in self.face_joint_ids:
            return 0.0
        
        # Get current face joint state
        current_joint_angle = self._get_face_joint_angle(current_face)
        current_joint_velocity = self._get_face_joint_velocity(current_face)
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        reward = 0.0
        
        # Check if rotation has started
        if not self.rotation_started:
            # Rotation starts when there's significant joint angular velocity AND contact
            if abs(current_joint_velocity) > self.rotation_start_threshold and force_magnitude > 0.5:
                self.rotation_started = True
                self.initial_joint_angle = 0.0 #current_joint_angle
                self.rotation_angle_accumulated = 0.0
                # Initialize angles for all faces to track wrong face rotations
                if not self.face_initial_angles:
                    for face_name in self.face_joint_ids.keys():
                        self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
                # Give positive reward for starting rotation
                reward += 10.0
                rotation_key = f"{current_face}_start_{self.current_rotation_index}"
                if rotation_key not in self.rotation_rewards_given:
                    self.rotation_rewards_given.add(rotation_key)
            else:
                # Rotation not started yet - return 0 (no reward, no penalty)
                return 0.0
        
        # If rotation has started, calculate continuous penalty based on distance from desired angle
        if self.rotation_started and not self.rotation_completed:
            if self.initial_joint_angle is not None:
                # Calculate the change in joint angle from initial position
                angle_change = current_joint_angle - self.initial_joint_angle
                
                # Handle angle wrapping (joint angles can wrap around ±π)
                # Normalize to [-π, π] range
                while angle_change > np.pi:
                    angle_change -= 2 * np.pi
                while angle_change < -np.pi:
                    angle_change += 2 * np.pi
                
                # Target rotation angle is 90 degrees (π/2 radians)
                target_angle = self.rotation_complete_threshold  # π/2 radians (90 degrees)
                
                # Calculate the current rotation angle (absolute value)
                rotation_angle = abs(angle_change)
                
                # Determine if we're rotating in the correct direction
                actual_direction = 1 if angle_change > 0 else -1 if angle_change < 0 else 0
                is_correct_direction = (actual_direction == desired_direction) if actual_direction != 0 else False
                
                # Calculate distance from desired rotation angle (90 degrees)
                if is_correct_direction:
                    # Rotating in correct direction: distance is how far we are from 90 degrees
                    distance_from_target = abs(target_angle - rotation_angle)
                else:
                    # Rotating in wrong direction: distance = current rotation + target rotation
                    distance_from_target = rotation_angle + target_angle
                
                # Continuous penalty: proportional to distance from target
                # Penalty increases as we get further from 90 degrees
                # Normalize by target_angle to get normalized distance
                normalized_distance = distance_from_target / target_angle
                penalty = -normalized_distance  # Negative because it's a penalty
                
                # Check if rotation is complete (within threshold of 90 degrees AND correct direction)
                if rotation_angle >= target_angle * 0.95 and is_correct_direction:
                    self.rotation_completed = True
                    # Zero penalty for completion
                    penalty = 0.0
                    
                    # Move to next rotation in sequence
                    self.current_rotation_index += 1
                    if self.current_rotation_index < len(self.rotation_sequence):
                        # Reset for next rotation
                        self.rotation_started = False
                        self.rotation_completed = False
                        self.initial_joint_angle = None
                        self.rotation_angle_accumulated = 0.0
                        self.rotation_direction_actual = 0.0
                        # Reset initial angles for all faces
                        self.face_initial_angles = {}
                        for face_name in self.face_joint_ids.keys():
                            self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
                
                reward += penalty
                return reward
            else:
                # Initial angle not set - return 0
                return 0.0
        elif self.rotation_completed:
            # Rotation completed - no penalty
            return 0.0
        else:
            # Rotation not started - return 0
            return 0.0
    
    def _calculate_wrong_face_rotation_penalty(self, current_target_face: str) -> float:
        """
        Calculate penalty for rotating faces other than the current target face.
        Penalty is a continuous function of:
        - Number of wrong faces being rotated
        - Magnitude of rotation for each wrong face
        
        Args:
            current_target_face: The face that should be rotated according to rotation sequence
            
        Returns:
            Penalty value (positive number, will be subtracted from reward)
        """
        if not self.rotation_started:
            # If rotation hasn't started, initialize angles for all faces
            # This allows us to detect any rotation from the initial state
            if not self.face_initial_angles:
                for face_name in self.face_joint_ids.keys():
                    self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
            return 0.0
        
        # If rotation has started, check all faces for unwanted rotation
        total_penalty = 0.0
        wrong_faces_rotated = []
        
        for face_name in self.face_joint_ids.keys():
            # Skip the current target face (it's allowed to rotate)
            if face_name == current_target_face:
                continue
            
            # Get current and initial angles for this face
            if face_name not in self.face_initial_angles:
                # Initialize if not already stored
                self.face_initial_angles[face_name] = self._get_face_joint_angle(face_name)
                continue
            
            current_angle = self._get_face_joint_angle(face_name)
            initial_angle = self.face_initial_angles[face_name]
            
            # Calculate rotation angle change
            angle_change = current_angle - initial_angle
            
            # Handle angle wrapping (normalize to [-π, π] range)
            while angle_change > np.pi:
                angle_change -= 2 * np.pi
            while angle_change < -np.pi:
                angle_change += 2 * np.pi
            
            # Calculate absolute rotation magnitude
            rotation_magnitude = abs(angle_change)
            
            # Only penalize if there's significant rotation (above noise threshold)
            if rotation_magnitude > 0.05:  # ~3 degrees threshold to ignore noise
                # Continuous penalty: proportional to rotation magnitude
                # Penalty increases quadratically with rotation magnitude to strongly discourage large wrong rotations
                face_penalty = self.wrong_face_rotation_penalty_coef * (rotation_magnitude ** 2)
                total_penalty += face_penalty
                wrong_faces_rotated.append((face_name, rotation_magnitude))
        
        # Additional penalty based on number of wrong faces rotated
        # This encourages the agent to rotate only the target face
        if len(wrong_faces_rotated) > 0:
            num_penalty = 0.1 * len(wrong_faces_rotated)  # Small penalty per wrong face
            total_penalty += num_penalty
        
        # Optional: Log wrong face rotations for debugging
        # if wrong_faces_rotated:
        #     print(f"[WRONG_FACE_PENALTY] Wrong faces rotated: {wrong_faces_rotated} | Total penalty: {total_penalty:.4f}")
        
        return total_penalty
    
    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute inverse of quaternion (conjugate for unit quaternion)."""
        # For unit quaternion, inverse is conjugate: [w, -x, -y, -z]
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])
    
    def _parse_rotation_sequence(self, sequence: List[str]):
        """
        Parse rotation sequence to extract face names and directions.
        
        Args:
            sequence: List of rotation specifications (e.g., ['red_clock', 'blue_anti_clock', 'white'])
                     Format: 'face_clock' for +90° clockwise, 'face_anti_clock' for -90° anti-clockwise
                     Or just 'face' for backward compatibility (defaults to clockwise)
        """
        valid_faces = {'red', 'orange', 'blue', 'green', 'white', 'yellow'}
        parsed_sequence = []
        
        for item in sequence:
            item_lower = item.lower().strip()
            
            # Check for _clock or _anti_clock suffix
            if item_lower.endswith('_clock'):
                face = item_lower[:-6]  # Remove '_clock'
                direction = 1  # +90 degrees clockwise
            elif item_lower.endswith('_anti_clock'):
                face = item_lower[:-11]  # Remove '_anti_clock'
                direction = -1  # -90 degrees anti-clockwise
            else:
                # Backward compatibility: if no suffix, default to clockwise
                face = item_lower
                direction = 1  # Default to clockwise
            
            # Validate face name
            if face in valid_faces:
                parsed_sequence.append((face, direction))
            else:
                print(f"Warning: Invalid face name '{face}' in rotation sequence item '{item}'. Skipping.")
        
        self.rotation_sequence = parsed_sequence
    
    def set_rotation_sequence(self, sequence: List[str]):
        """
        Set the rotation sequence for the episode.
        
        Args:
            sequence: List of rotation specifications (e.g., ['red_clock', 'blue_anti_clock', 'white'])
                     Format: 'face_clock' for +90° clockwise, 'face_anti_clock' for -90° anti-clockwise
                     Or just 'face' for backward compatibility (defaults to clockwise)
        """
        if sequence is None:
            self.rotation_sequence = []
        else:
            self._parse_rotation_sequence(sequence)
        
        # Reset rotation tracking
        self.current_rotation_index = 0
        self.rotation_started = False
        self.rotation_completed = False
        self.initial_joint_angle = None
        self.rotation_angle_accumulated = 0.0
        self.rotation_direction_actual = 0.0
        self.face_initial_angles = {}
        self.rotation_rewards_given.clear()
    
    def apply_rotation(self, rotation_spec: str) -> bool:
        """
        Directly apply a rotation to the cube by setting the joint angle.
        
        This function directly sets the cube joint angle to the target rotation angle,
        bypassing the need for manual manipulation. Useful for testing or initialization.
        
        Args:
            rotation_spec: Rotation specification string (e.g., 'blue_clock', 'red_anti_clock', 'white')
                         Format: 'face_clock' for +90° clockwise, 'face_anti_clock' for -90° anti-clockwise
                         Or just 'face' for backward compatibility (defaults to clockwise)
        
        Returns:
            True if rotation was successfully applied, False otherwise
        
        Example:
            >>> env.apply_rotation('blue_clock')  # Rotate blue face 90° clockwise
            >>> env.apply_rotation('red_anti_clock')  # Rotate red face 90° anti-clockwise
            >>> env.apply_rotation('white')  # Rotate white face 90° clockwise (default)
        """
        # Parse the rotation specification
        valid_faces = {'red', 'orange', 'blue', 'green', 'white', 'yellow'}
        rotation_spec_lower = rotation_spec.lower().strip()
        
        # Extract face name and direction
        if rotation_spec_lower.endswith('_anti_clock'):
            print(f"Anti-clockwise rotation: {rotation_spec_lower}")
            face = rotation_spec_lower[:-11]  # Remove '_anti_clock'
            direction = -1
        elif rotation_spec_lower.endswith('_clock'):
            face = rotation_spec_lower[:-6]  # Remove '_clock'
            direction = 1  # +90 degrees clockwise
          # -90 degrees anti-clockwise
        else:
            # Backward compatibility: if no suffix, default to clockwise
            print(f"No suffix: {rotation_spec_lower}")
            face = rotation_spec_lower
            direction = 1  # Default to clockwise
        
        # Validate face name
        if face not in valid_faces:
            print(f"Error: Invalid face name '{face}' in rotation specification '{rotation_spec}'.")
            return False
        
        # Check if joint exists for this face
        if face not in self.face_joint_ids:
            print(f"Error: Joint for face '{face}' not found in model.")
            return False
        
        # Get the joint ID and calculate target angle
        joint_id = self.face_joint_ids[face]
        
        # Debug: Check joint type and properties
        joint_type = self.model.jnt_type[joint_id]
        joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        print(f"DEBUG: Joint ID: {joint_id}, Name: {joint_name}, Type: {joint_type} (1=hinge, 2=slide, 3=free, 4=ball)")
        
        # Get the joint axis direction to account for negative axis joints
        # Joints with negative axes (nX, nY, nZ) need sign flip for rotation direction
        joint_axis = self.model.jnt_axis[joint_id]
        # Find the non-zero component and get its sign
        non_zero_idx = np.nonzero(joint_axis)[0]
        if len(non_zero_idx) > 0:
            axis_sign = np.sign(joint_axis[non_zero_idx[0]])
        else:
            axis_sign = 1.0  # Default to positive if axis is zero (shouldn't happen)
        
        print(f"DEBUG: Joint axis: {joint_axis}, Non-zero index: {non_zero_idx}, Axis sign: {axis_sign}")
        
        # Calculate target angle: multiply by axis_sign to account for joint axis direction
        # For joints with negative axes, we need to flip the rotation direction
        target_angle = direction * axis_sign * (np.pi / 2.0)  # ±90 degrees (±π/2 radians)
        
        # Get the current joint angle
        qpos_adr = self.model.jnt_qposadr[joint_id]# Number of qpos elements for this joint
        print(f"DEBUG: qpos address: {qpos_adr}")
        current_angle = self.data.qpos[qpos_adr]
        
        # Get the body ID for this face to check its transform
        joint_body_id = self.model.jnt_bodyid[joint_id]
        joint_body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, joint_body_id)
        print(f"DEBUG: Joint body ID: {joint_body_id}, Name: {joint_body_name}")
        
        # Get body transform before rotation
        body_xpos_before = self.data.xpos[joint_body_id].copy()
        body_xquat_before = self.data.xquat[joint_body_id].copy()
        body_xmat_before = self.data.xmat[joint_body_id].copy()
        print(f"DEBUG: Body position before: {body_xpos_before}")
        print(f"DEBUG: Body quaternion before: {body_xquat_before}")
        
        # Calculate target angle (relative to current angle for cumulative rotations)
        new_angle = current_angle + target_angle
        print(f"Face: {face}, Direction: {direction}, Axis sign: {axis_sign:.1f}, Current angle: {np.degrees(current_angle):.1f}°, Target angle: {np.degrees(target_angle):.1f}°, New angle: {np.degrees(new_angle):.1f}°")
        
        # Find the actuator ID for this face
        actuator_id = None
        for i, name in enumerate(self.actuator_names):
            if name == face:
                actuator_id = i
                break
        
        if actuator_id is None:
            print(f"Error: Actuator for face '{face}' not found.")
            return False
        
        # Get actuator control range and type
        # Note: Control values are in radians (XML specifies angle="radian")
        ctrl_range = self.model.actuator_ctrlrange[actuator_id]
        actuator_type = self.model.actuator_gaintype[actuator_id]
        print(f"DEBUG: Actuator ID: {actuator_id}, Type: {actuator_type}, Control range: [{ctrl_range[0]:.4f}, {ctrl_range[1]:.4f}] rad "
              f"([{np.degrees(ctrl_range[0]):.3f}°, {np.degrees(ctrl_range[1]):.3f}°])")
        
        # Reset joint velocity to zero to avoid residual motion
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr] = 0.0
        
        # Disable all other cube actuators to prevent interference
        for other_actuator_id in self.cube_actuators:
            if other_actuator_id != actuator_id:
                self.data.ctrl[other_actuator_id] = 0.0
        
        # Control loop: Move actuator incrementally toward target angle
        # Use proportional control: control_value = Kp * error
        # where error = target_angle - current_angle
        Kp = 10.0  # Proportional gain (adjust based on actuator response)
        tolerance = 0.0  # Tolerance in radians (zero tolerance - will run until max_steps or exact match)
        max_steps = 500  # Maximum steps to prevent infinite loops
        
        print(f"DEBUG: Starting control loop to reach target angle {np.degrees(new_angle):.1f}° ({new_angle:.4f} rad)")
        
        for step_idx in range(max_steps):
            # Get current joint angle
            current_angle_now = self.data.qpos[qpos_adr]
            
            # Calculate error (target - current)
            angle_error = new_angle - current_angle_now
            
            # Handle angle wrapping (normalize error to [-π, π])
            while angle_error > np.pi:
                angle_error -= 2 * np.pi
            while angle_error < -np.pi:
                angle_error += 2 * np.pi
            
            # Check if we've reached the target
            if abs(angle_error) < tolerance:
                print(f"DEBUG: Target reached at step {step_idx}! Current: {np.degrees(current_angle_now):.1f}°, "
                      f"Target: {np.degrees(new_angle):.1f}°, Error: {np.degrees(angle_error):.3f}°")
                break
            
            # Calculate control value using proportional control
            # Control is proportional to error, with sign matching desired direction
            control_value = Kp * angle_error
            
            # Clip control value to actuator range if it has limits
            # But allow larger values if needed for faster convergence
            if ctrl_range[0] != ctrl_range[1]:  # If range is not zero (unlimited)
                # Use a larger effective range for control, but clip extreme values
                effective_max = max(abs(ctrl_range[0]), abs(ctrl_range[1])) * 2.0
                control_value = np.clip(control_value, -effective_max, effective_max)
            
            # Apply control value to actuator
            self.data.ctrl[actuator_id] = control_value
            
            # Step physics simulation
            mj.mj_step(self.model, self.data)
            
            # Sync viewer periodically for visualization
            if self.enable_viewer and self.viewer is not None:
                if step_idx % 5 == 0:  # Sync every 5 steps for smoother visualization
                    self.viewer.sync()
            
            # Print progress every 50 steps
            if step_idx % 50 == 0:
                print(f"DEBUG: Step {step_idx}: Current angle: {np.degrees(current_angle_now):.1f}°, "
                      f"Error: {np.degrees(angle_error):.3f}°, Control: {control_value:.4f}")
        
        # Set actuator to zero after reaching target to maintain position
        self.data.ctrl[actuator_id] = 0.0
        
        # Final forward kinematics update
        mj.mj_forward(self.model, self.data)
        
        # Verify final angle
        final_angle = self.data.qpos[qpos_adr]
        final_error = new_angle - final_angle
        while final_error > np.pi:
            final_error -= 2 * np.pi
        while final_error < -np.pi:
            final_error += 2 * np.pi
        print(f"DEBUG: Final angle: {np.degrees(final_angle):.1f}°, Target: {np.degrees(new_angle):.1f}°, "
              f"Final error: {np.degrees(final_error):.3f}°")
        
        # Get body transform after rotation
        body_xpos_after = self.data.xpos[joint_body_id].copy()
        body_xquat_after = self.data.xquat[joint_body_id].copy()
        final_joint_angle = self.data.qpos[qpos_adr]
        print(f"DEBUG: Final joint angle: {np.degrees(final_joint_angle):.1f}°")
        print(f"DEBUG: Body position after: {body_xpos_after}")
        print(f"DEBUG: Body quaternion after: {body_xquat_after}")
        print(f"DEBUG: Position changed: {not np.allclose(body_xpos_before, body_xpos_after)}")
        print(f"DEBUG: Quaternion changed: {not np.allclose(body_xquat_before, body_xquat_after)}")
        
        # Final viewer sync
        if self.enable_viewer and self.viewer is not None:
            self.viewer.sync()
        
        print(f"Applied rotation: {rotation_spec} -> {face} face rotated {direction * 90}° "
              f"(joint angle: {np.degrees(current_angle):.1f}° -> {np.degrees(new_angle):.1f}°)")
        
        return True
    
    def apply_rotation_sequence(self, rotation_sequence: List[str]) -> bool:
        """
        Apply a sequence of rotations to the cube.
        
        This function applies multiple rotations in sequence, useful for setting up
        specific cube configurations or testing rotation sequences.
        
        Args:
            rotation_sequence: List of rotation specifications (e.g., ['blue_clock', 'red_anti_clock', 'white'])
                             Each item follows the same format as apply_rotation()
        
        Returns:
            True if all rotations were successfully applied, False otherwise
        
        Example:
            >>> env.apply_rotation_sequence(['blue_clock', 'red_anti_clock', 'white'])
        """
        success = True
        for i, rotation_spec in enumerate(rotation_sequence):
            if not self.apply_rotation(rotation_spec):
                print(f"Error: Failed to apply rotation {i+1}/{len(rotation_sequence)}: '{rotation_spec}'")
                success = False
        return success
    
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
    parser.add_argument('--max-steps', type=int, default=500,
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
