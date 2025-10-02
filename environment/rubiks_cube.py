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
        
        
        
        # Aathira : Isn't self.action_dim and self.model.nu the same?
        print(f"Environment initialized with {self.model.nu} actuators")
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
        
        for i, name in enumerate(self.actuator_names):
            if name.startswith('lh_A_'):
                self.left_hand_actuators.append(i)
            elif name.startswith('rh_A_'):
                self.right_hand_actuators.append(i)
            elif name in ['red', 'orange', 'blue', 'green', 'white', 'yellow']:
                self.cube_actuators.append(i)
        
        print(f"Left hand actuators: {len(self.left_hand_actuators)}")
        print(f"Right hand actuators: {len(self.right_hand_actuators)}")
        print(f"Cube actuators: {len(self.cube_actuators)}")
    
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
            self.model.nu    # Previous actions
        )
        
        # Action space: all actuator controls
        self.action_dim = self.model.nu
    
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
        
        # Previous actions (current control values)
        prev_actions = self.data.ctrl.copy()
        
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
        """Get contact forces between hands and cube."""
        # Simplified contact force representation
        # In a full implementation, you would extract specific contact forces
        contact_force = np.zeros(3)  # [fx, fy, fz]
        
        # Aggregate normal/contact forces using MuJoCo API
        ncon = self.data.ncon
        if ncon > 0:
            efc_force = np.zeros(6)
            for i in range(ncon):
                mj.mj_contactForce(self.model, self.data, i, efc_force)
                contact_force += efc_force[3:] * 0.1  # accumulate linear force (fx, fy, fz)
        
        return contact_force
    
    def get_action(self, action_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parse action vector into separate actions for each component.
        
        Args:
            action_vector: Combined action vector
            
        Returns:
            Dictionary with separate actions for left hand, right hand, and cube
        """
        actions = {
            'left_hand': action_vector[self.left_hand_actuators],
            'right_hand': action_vector[self.right_hand_actuators],
            'cube': action_vector[self.cube_actuators] if self.cube_actuators else np.array([])
        }
        
        return actions
    
    def take_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action vector for all actuators
            
        Returns:
            next_state: Next state vector
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Clip actions to valid ranges
        action = self._clip_actions(action)
        
        # Apply actions
        self.data.ctrl[:] = action
        
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
        self.visualize_collision_boxes_and_axes()
        
        return next_state, reward, done, info
    
    def _clip_actions(self, action: np.ndarray) -> np.ndarray:
        """Clip actions to valid actuator ranges."""
        ctrl_ranges = self.model.actuator_ctrlrange
        clipped_action = np.clip(action, ctrl_ranges[:, 0], ctrl_ranges[:, 1])
        return clipped_action
    
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
        reward += grasp_reward * 0.4
        
        # 2. Manipulation reward (based on cube rotation) - PRIORITY
        manipulation_reward = self._calculate_manipulation_reward()
        reward += manipulation_reward * 0.3
        
        # 3. Cube manipulation reward (based on cube movement and stability)
        cube_reward = self._calculate_cube_reward()
        reward += cube_reward * 0.2
        
        # 4. Efficiency reward (penalize excessive actions)
        efficiency_reward = self._calculate_efficiency_reward(action)
        reward += efficiency_reward * 0.05
        
        # 5. Stability reward (penalize unstable configurations)
        stability_reward = self._calculate_stability_reward()
        reward += stability_reward * 0.05
        
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
            if force_magnitude < 2.0:
                return 2.0  # Strong reward for gentle but firm grasp
            elif force_magnitude < 5.0:
                return 1.0  # Good reward for moderate grasp
            else:
                return 0.5  # Reduced reward for excessive force
        else:
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
        
        # Bonus reward for active rotation (angular velocity)
        angular_velocity_magnitude = np.linalg.norm(cube_ang_vel)
        if angular_velocity_magnitude > 0.1:  # If cube is actively rotating
            rotation_reward += min(angular_velocity_magnitude, 2.0) * 0.5
        
        # Check if we have contact while rotating (good manipulation)
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        if force_magnitude > 0.1 and angular_velocity_magnitude > 0.05:
            # Bonus for rotating while maintaining grasp
            rotation_reward += 1.0
        
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
    
    def visualize_collision_boxes_and_axes(self):
        """
        Visualize collision boxes and object axes for debugging purposes.
        This function adds visual elements to help understand spatial relationships.
        """
        if not self.visualize_collision_boxes or not self.enable_viewer:
            return
        
        # Get viewer context
        if self.viewer is None:
            return
        
        # Clear previous visualization lines
        self._clear_visualization_lines()
        
        # Visualize collision boxes for key objects
        self._visualize_body_collision_boxes()
        
        # Visualize coordinate axes for key objects
        self._visualize_object_axes()
    
    def _clear_visualization_lines(self):
        """Clear previous visualization lines from the scene."""
        # Clear stored line data
        if hasattr(self, '_debug_lines'):
            self._debug_lines.clear()
    
    def _visualize_body_collision_boxes(self):
        """Visualize collision boxes for important bodies."""
        # Key bodies to visualize
        important_bodies = [
            "core",  # Rubik's cube
            "lh_palm",  # Left hand palm
            "rh_palm",  # Right hand palm
            "lh_ffdistal",  # Left hand index finger tip
            "rh_ffdistal",  # Right hand index finger tip
            "lh_thdistal",  # Left hand thumb tip
            "rh_thdistal",  # Right hand thumb tip
        ]
        
        for body_name in important_bodies:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                self._draw_body_bounding_box(body_id, body_name)
    
    def _draw_body_bounding_box(self, body_id: int, body_name: str):
        """Draw collision geometries for a specific body using actual XML definitions."""
        # Get body position and orientation
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.zeros(9, dtype=np.float64)  # Flattened 3x3 matrix
        mj.mju_quat2Mat(rot_matrix, quat.astype(np.float64))
        rot_matrix = rot_matrix.reshape(3, 3)  # Reshape to 3x3
        
        # Find all geoms belonging to this body
        geom_start = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]
        
        for i in range(geom_num):
            geom_id = geom_start + i
            self._draw_geom_collision_box(geom_id, pos, rot_matrix)
    
    def _draw_geom_collision_box(self, geom_id: int, body_pos: np.ndarray, body_rot: np.ndarray):
        """Draw collision box for a specific geom based on its type and size."""
        # Get geom type
        geom_type = self.model.geom_type[geom_id]
        
        # Get geom position and orientation relative to body
        geom_pos = self.model.geom_pos[geom_id].copy()
        geom_quat = self.model.geom_quat[geom_id].copy()
        
        # Transform geom position to world coordinates
        world_pos = body_pos + np.dot(body_rot, geom_pos)
        
        # Convert geom quaternion to rotation matrix
        geom_rot = np.zeros(9, dtype=np.float64)  # Flattened 3x3 matrix
        mj.mju_quat2Mat(geom_rot, geom_quat.astype(np.float64))
        geom_rot = geom_rot.reshape(3, 3)  # Reshape to 3x3
        
        # Combine body and geom rotations
        combined_rot = np.dot(body_rot, geom_rot)
        
        # Get geom size based on type
        geom_size = self.model.geom_size[geom_id].copy()
        
        # Draw based on geometry type
        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            # Box geometry
            box_size = geom_size * 2  # MuJoCo size is half-extents
            self._draw_wireframe_box(world_pos, combined_rot, box_size, 
                                   color=[1.0, 0.0, 0.0, 0.7])
            
        elif geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
            # Cylinder geometry
            radius = geom_size[0]
            height = geom_size[1] * 2
            self._draw_wireframe_cylinder(world_pos, combined_rot, radius, height,
                                        color=[0.0, 1.0, 0.0, 0.7])
            
        elif geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
            # Capsule geometry
            radius = geom_size[0]
            height = geom_size[1] * 2
            self._draw_wireframe_capsule(world_pos, combined_rot, radius, height,
                                       color=[0.0, 0.0, 1.0, 0.7])
            
        elif geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            # Sphere geometry
            radius = geom_size[0]
            self._draw_wireframe_sphere(world_pos, radius, color=[1.0, 1.0, 0.0, 0.7])
            
        elif geom_type == mj.mjtGeom.mjGEOM_MESH:
            # Mesh geometry - draw bounding box approximation
            # For meshes, we'll use a default size since extracting mesh bounds is complex
            mesh_size = np.array([0.02, 0.02, 0.02])  # 2cm default
            self._draw_wireframe_box(world_pos, combined_rot, mesh_size,
                                   color=[1.0, 0.5, 0.0, 0.7])
    
    def _visualize_object_axes(self):
        """Visualize coordinate axes for key objects."""
        # Key objects to show axes for
        important_bodies = ["core", "lh_palm", "rh_palm"]
        
        for body_name in important_bodies:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                self._draw_coordinate_axes(body_id, body_name)
    
    def _draw_coordinate_axes(self, body_id: int, body_name: str):
        """Draw coordinate axes for a specific body."""
        # Get body position and orientation
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.zeros(9, dtype=np.float64)  # Flattened 3x3 matrix
        mj.mju_quat2Mat(rot_matrix, quat.astype(np.float64))
        rot_matrix = rot_matrix.reshape(3, 3)  # Reshape to 3x3
        
        # Axis length
        axis_length = 0.05  # 5cm
        
        # Draw X axis (red)
        x_end = pos + rot_matrix[:, 0] * axis_length
        self._draw_line(pos, x_end, color=[1.0, 0.0, 0.0, 1.0], width=3)
        
        # Draw Y axis (green)
        y_end = pos + rot_matrix[:, 1] * axis_length
        self._draw_line(pos, y_end, color=[0.0, 1.0, 0.0, 1.0], width=3)
        
        # Draw Z axis (blue)
        z_end = pos + rot_matrix[:, 2] * axis_length
        self._draw_line(pos, z_end, color=[0.0, 0.0, 1.0, 1.0], width=3)
    
    def _draw_wireframe_box(self, center: np.ndarray, rotation: np.ndarray, 
                           size: np.ndarray, color: List[float]):
        """Draw a wireframe box at the specified position and orientation."""
        # Define box vertices in local coordinates
        half_size = size / 2
        vertices = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0],  half_size[1],  half_size[2]],
            [-half_size[0],  half_size[1],  half_size[2]]
        ])
        
        # Transform vertices to world coordinates
        world_vertices = np.dot(vertices, rotation.T) + center
        
        # Define edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Draw each edge
        for edge in edges:
            start = world_vertices[edge[0]]
            end = world_vertices[edge[1]]
            self._draw_line(start, end, color, width=2)
    
    def _draw_wireframe_cylinder(self, center: np.ndarray, rotation: np.ndarray,
                                radius: float, height: float, color: List[float]):
        """Draw a wireframe cylinder."""
        # Create cylinder wireframe
        num_segments = 16
        angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)
        
        # Bottom circle
        for i in range(num_segments):
            angle1 = angles[i]
            angle2 = angles[(i + 1) % num_segments]
            
            # Bottom circle points
            p1_bottom = center + np.dot(rotation, np.array([radius * np.cos(angle1), 
                                                           radius * np.sin(angle1), -height/2]))
            p2_bottom = center + np.dot(rotation, np.array([radius * np.cos(angle2), 
                                                           radius * np.sin(angle2), -height/2]))
            
            # Top circle points
            p1_top = center + np.dot(rotation, np.array([radius * np.cos(angle1), 
                                                        radius * np.sin(angle1), height/2]))
            p2_top = center + np.dot(rotation, np.array([radius * np.cos(angle2), 
                                                        radius * np.sin(angle2), height/2]))
            
            # Draw bottom circle edge
            self._draw_line(p1_bottom, p2_bottom, color, width=2)
            # Draw top circle edge
            self._draw_line(p1_top, p2_top, color, width=2)
            # Draw vertical edge
            self._draw_line(p1_bottom, p1_top, color, width=2)
    
    def _draw_wireframe_capsule(self, center: np.ndarray, rotation: np.ndarray,
                               radius: float, height: float, color: List[float]):
        """Draw a wireframe capsule (cylinder with hemispherical caps)."""
        # Draw the cylindrical part
        self._draw_wireframe_cylinder(center, rotation, radius, height, color)
        
        # Draw top hemisphere
        self._draw_wireframe_hemisphere(center + np.dot(rotation, np.array([0, 0, height/2])), 
                                       rotation, radius, color)
        
        # Draw bottom hemisphere
        bottom_rot = np.dot(rotation, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        self._draw_wireframe_hemisphere(center + np.dot(rotation, np.array([0, 0, -height/2])), 
                                       bottom_rot, radius, color)
    
    def _draw_wireframe_hemisphere(self, center: np.ndarray, rotation: np.ndarray,
                                  radius: float, color: List[float]):
        """Draw a wireframe hemisphere."""
        num_rings = 4
        num_segments = 12
        
        for ring in range(num_rings):
            phi = (ring + 1) * np.pi / (2 * num_rings)
            ring_radius = radius * np.sin(phi)
            z = radius * np.cos(phi)
            
            angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)
            for i in range(num_segments):
                angle1 = angles[i]
                angle2 = angles[(i + 1) % num_segments]
                
                p1 = center + np.dot(rotation, np.array([ring_radius * np.cos(angle1), 
                                                        ring_radius * np.sin(angle1), z]))
                p2 = center + np.dot(rotation, np.array([ring_radius * np.cos(angle2), 
                                                        ring_radius * np.sin(angle2), z]))
                
                self._draw_line(p1, p2, color, width=1)
    
    def _draw_wireframe_sphere(self, center: np.ndarray, radius: float, color: List[float]):
        """Draw a wireframe sphere."""
        num_rings = 6
        num_segments = 12
        
        # Draw latitude rings
        for ring in range(num_rings):
            phi = (ring + 1) * np.pi / (num_rings + 1)
            ring_radius = radius * np.sin(phi)
            z = radius * np.cos(phi)
            
            angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)
            for i in range(num_segments):
                angle1 = angles[i]
                angle2 = angles[(i + 1) % num_segments]
                
                p1 = center + np.array([ring_radius * np.cos(angle1), 
                                       ring_radius * np.sin(angle1), z])
                p2 = center + np.array([ring_radius * np.cos(angle2), 
                                       ring_radius * np.sin(angle2), z])
                
                self._draw_line(p1, p2, color, width=1)
        
        # Draw longitude lines
        for i in range(num_segments):
            angle = i * 2 * np.pi / num_segments
            for ring in range(num_rings - 1):
                phi1 = (ring + 1) * np.pi / (num_rings + 1)
                phi2 = (ring + 2) * np.pi / (num_rings + 1)
                
                r1 = radius * np.sin(phi1)
                z1 = radius * np.cos(phi1)
                r2 = radius * np.sin(phi2)
                z2 = radius * np.cos(phi2)
                
                p1 = center + np.array([r1 * np.cos(angle), r1 * np.sin(angle), z1])
                p2 = center + np.array([r2 * np.cos(angle), r2 * np.sin(angle), z2])
                
                self._draw_line(p1, p2, color, width=1)
    
    def _draw_line(self, start: np.ndarray, end: np.ndarray, 
                   color: List[float], width: int = 2):
        """Draw a line between two points using MuJoCo's visualization API."""
        if not self.visualize_collision_boxes or not self.enable_viewer or self.viewer is None:
            return
        
        try:
            # For now, we'll use a simpler approach that works with current MuJoCo versions
            # Store line data for potential future rendering or debugging
            if not hasattr(self, '_debug_lines'):
                self._debug_lines = []
            
            # Store line information for debugging/analysis
            self._debug_lines.append({
                'start': start.copy(),
                'end': end.copy(),
                'color': color.copy(),
                'width': width
            })
            
            # Print line information for debugging (can be removed in production)
            if len(self._debug_lines) <= 10:  # Limit output to avoid spam
                # print(f"Line: {start} -> {end}, Color: {color}, Width: {width}")
                pass
                
        except Exception as e:
            # Fallback to debug printing if visualization fails
            if self.visualize_collision_boxes:
                print(f"Line: {start} -> {end}, Color: {color}, Width: {width}")
                print(f"Visualization error: {e}")
    
    def toggle_collision_visualization(self):
        """Toggle collision box visualization on/off."""
        self.visualize_collision_boxes = not self.visualize_collision_boxes
        print(f"Collision box visualization: {'ON' if self.visualize_collision_boxes else 'OFF'}")
    
    def get_debug_lines(self):
        """Get the current debug lines for analysis."""
        return getattr(self, '_debug_lines', [])
    
    def save_debug_lines_to_file(self, filename: str = "debug_lines.txt"):
        """Save debug lines to a file for analysis."""
        if not hasattr(self, '_debug_lines') or not self._debug_lines:
            print("No debug lines to save.")
            return
        
        with open(filename, 'w') as f:
            f.write("Debug Lines from Collision Box Visualization\n")
            f.write("=" * 50 + "\n")
            for i, line in enumerate(self._debug_lines):
                f.write(f"Line {i+1}:\n")
                f.write(f"  Start: {line['start']}\n")
                f.write(f"  End: {line['end']}\n")
                f.write(f"  Color: {line['color']}\n")
                f.write(f"  Width: {line['width']}\n")
                f.write("\n")
        
        print(f"Debug lines saved to {filename}")
    
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
    # Create environment
    env = RubiksCubeEnvironment(
        xml_path="xmls/bidexhands.xml",
        enable_viewer=True,
        max_episode_steps=100000,
        visualize_collision_boxes=True,  # Enable collision box visualization
        settle_on_reset=False,
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
