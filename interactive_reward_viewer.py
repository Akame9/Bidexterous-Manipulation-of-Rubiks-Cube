"""
Interactive MuJoCo Viewer with Reward and Contact Force Visualization

This script allows users to interactively control hand actuators and see:
- Contact forces between hands and cube
- Grasp reward
- Manipulation reward
- Total reward

Controls:
- Use keyboard to adjust actuator values
- See real-time feedback on forces and rewards
- Press ESC to exit

This script extracts the reward calculation functions from RubiksCubeEnvironment
for standalone interactive visualization.
"""

import numpy as np
import mujoco as mj
import mujoco.viewer
import time
import argparse
from typing import Dict, Tuple
from collections import deque


class InteractiveRewardViewer:
    """
    Interactive viewer for exploring contact forces and rewards.
    """
    
    def __init__(self, xml_path="xmls/bidexhands.xml"):
        """
        Initialize the interactive viewer.
        
        Args:
            xml_path: Path to the MuJoCo XML model file
        """
        self.xml_path = xml_path
        
        # Load MuJoCo model and data
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Setup actuators
        self._setup_actuators()
        
        # Contact tracking
        self.contact_history = deque(maxlen=100)
        
        # Target cube configuration (for reward calculation)
        self.target_cube_config = self._get_initial_cube_config()
        
        # Control state
        self.selected_actuator_idx = 0  # Currently selected actuator from hand_actuators list
        self.control_increment = 0.1
        
        # Display info
        self.show_help = True
        
        # Initialize to neutral pose
        self._set_neutral_pose()
        self._set_initial_cube_pose()
        
        print(f"Interactive Viewer initialized!")
        print(f"Total actuators: {self.model.nu}")
        print(f"Hand actuators: {len(self.hand_actuators)}")
        print(f"Cube actuators: {len(self.cube_actuators)}")
        
    def _setup_actuators(self):
        """Setup actuator information and groupings."""
        # Get all actuator names
        self.actuator_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) 
                              for i in range(self.model.nu)]
        
        # Group actuators by hand and cube
        self.left_hand_actuators = []
        self.right_hand_actuators = []
        self.cube_actuators = []
        self.hand_actuators = []  # Combined hand actuators
        
        for i, name in enumerate(self.actuator_names):
            if name.startswith('lh_A_'):
                self.left_hand_actuators.append(i)
                self.hand_actuators.append(i)
            elif name.startswith('rh_A_'):
                self.right_hand_actuators.append(i)
                self.hand_actuators.append(i)
            elif name in ['red', 'orange', 'blue', 'green', 'white', 'yellow']:
                self.cube_actuators.append(i)
    
    def _set_neutral_pose(self):
        """Set both hands to a neutral grasping pose."""
        ctrl_ranges = self.model.actuator_ctrlrange.copy()
        
        for i, name in enumerate(self.actuator_names):
            lo, hi = ctrl_ranges[i]
            
            if name in {"lh_A_WRJ1", "lh_A_WRJ2", "rh_A_WRJ1", "rh_A_WRJ2"}:
                self.data.ctrl[i] = 0.0
            elif any(k in name for k in ["lh_A_FFJ4", "lh_A_MFJ4", "lh_A_RFJ4", "lh_A_LFJ4",
                                        "rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4"]):
                self.data.ctrl[i] = 0.0
            elif any(k in name for k in ["lh_A_FFJ3", "lh_A_MFJ3", "lh_A_RFJ3", "lh_A_LFJ3",
                                        "rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"]):
                self.data.ctrl[i] = lo + 0.3 * (hi - lo)
            elif name.endswith("J0") and (name.startswith("lh_A_") or name.startswith("rh_A_")):
                self.data.ctrl[i] = lo + 0.5 * (hi - lo)
            elif name in {"lh_A_THJ4", "lh_A_THJ1", "rh_A_THJ4", "rh_A_THJ1"}:
                self.data.ctrl[i] = lo + 0.6 * (hi - lo)
            elif name in {"lh_A_THJ5", "lh_A_THJ3", "lh_A_THJ2", "rh_A_THJ5", "rh_A_THJ3", "rh_A_THJ2"}:
                self.data.ctrl[i] = 0.0
            elif name in {"lh_A_LFJ5", "rh_A_LFJ5"}:
                self.data.ctrl[i] = lo + 0.2 * (hi - lo)
            else:
                self.data.ctrl[i] = 0.0
        
        # Run simulation to settle
        for _ in range(100):
            mj.mj_step(self.model, self.data)
    
    def _set_initial_cube_pose(self):
        """Set initial cube position between the hands."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        target_position = np.array([0.35, 0.0, 0.25], dtype=np.float64)
        if cube_body_id != -1:
            cube_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "cube_free")
            if cube_joint_id != -1:
                qpos_adr = self.model.jnt_qposadr[cube_joint_id]
                self.data.qpos[qpos_adr:qpos_adr + 7] = np.concatenate(
                    [target_position, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)]
                )
                qvel_adr = self.model.jnt_dofadr[cube_joint_id]
                self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
            else:
                self.model.body_pos[cube_body_id] = target_position
            mj.mj_forward(self.model, self.data)
    
    def _get_initial_cube_config(self) -> Dict:
        """Get initial cube configuration for reference."""
        return {
            'position': self._get_cube_position(),
            'orientation': self._get_cube_orientation()
        }
    
    # ==================== REWARD FUNCTIONS (from RubiksCubeEnvironment) ====================
    
    def _get_contact_forces(self) -> np.ndarray:
        """Get contact forces between hands and cube only."""
        contact_force = np.zeros(3)  # [fx, fy, fz]
        
        # Get cube body ID
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        
        # Aggregate contact forces using MuJoCo API, filtering for hand-cube contacts only
        ncon = self.data.ncon
        hand_cube_contacts = 0
        total_raw_force = 0.0
        
        if ncon > 0 and cube_body_id != -1:
            efc_force = np.zeros(6)
            for i in range(ncon):
                # Get contact information
                contact = self.data.contact[i]
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                # Get body IDs for the geometries
                body1_id = self.model.geom_bodyid[geom1_id]
                body2_id = self.model.geom_bodyid[geom2_id]
                
                # Check if this is a hand-cube contact
                is_hand_cube_contact = False
                
                if body1_id == cube_body_id or body2_id == cube_body_id:
                    # Check if the other body is a hand
                    if body1_id == cube_body_id:
                        other_body_id = body2_id
                    else:
                        other_body_id = body1_id
                    
                    # Get body name to check if it's a hand
                    other_body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, other_body_id)
                    if other_body_name and ('lh_' in other_body_name or 'rh_' in other_body_name):
                        is_hand_cube_contact = True
                
                # Only accumulate forces from hand-cube contacts
                if is_hand_cube_contact:
                    mj.mj_contactForce(self.model, self.data, i, efc_force)
                    contact_force += efc_force[3:]  # accumulate linear force (fx, fy, fz)
                    total_raw_force += np.sum(np.abs(efc_force[3:]))
                    hand_cube_contacts += 1
        
        return contact_force
    
    def _calculate_grasp_reward(self) -> Tuple[float, Dict]:
        """Calculate reward based on grasping quality."""
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        info = {
            'force_magnitude': force_magnitude,
            'contact_force': contact_force.copy()
        }
        
        # Enhanced grasping reward
        if force_magnitude > 0.05:
            if force_magnitude < 2.0:
                reward = 2.0
                info['message'] = f"Gentle grasp (force={force_magnitude:.3f})"
            elif force_magnitude < 5.0:
                reward = 1.0
                info['message'] = f"Moderate grasp (force={force_magnitude:.3f})"
            else:
                reward = 0.5
                info['message'] = f"Excessive force (force={force_magnitude:.3f})"
        else:
            reward = -0.5
            info['message'] = f"No contact (force={force_magnitude:.3f})"
        
        return reward, info
    
    def _calculate_manipulation_reward(self) -> Tuple[float, Dict]:
        """Calculate reward for cube manipulation success."""
        # Get cube orientation and angular velocity
        cube_quat = self._get_cube_orientation()
        cube_ang_vel = self._get_cube_angular_velocity()
        
        # Reward for cube rotation (any rotation from initial state)
        initial_quat = self.target_cube_config['orientation']
        
        # Calculate rotation angle from initial orientation
        quat_dot = np.abs(np.dot(cube_quat, initial_quat))
        rotation_angle = 2 * np.arccos(np.clip(quat_dot, 0, 1))
        
        # Base rotation reward
        rotation_reward = min(rotation_angle / np.pi, 1.0) * 1.0
        
        info = {
            'rotation_angle': rotation_angle,
            'angular_velocity': cube_ang_vel.copy(),
            'angular_velocity_magnitude': np.linalg.norm(cube_ang_vel),
            'messages': []
        }
        
        info['messages'].append(f"Base rotation (angle={rotation_angle:.3f} rad) -> +{rotation_reward:.3f}")
        
        # Bonus reward for active rotation
        angular_velocity_magnitude = np.linalg.norm(cube_ang_vel)
        if angular_velocity_magnitude > 0.1:
            velocity_bonus = min(angular_velocity_magnitude, 2.0) * 0.5
            rotation_reward += velocity_bonus
            info['messages'].append(f"Active rotation (ang_vel={angular_velocity_magnitude:.3f}) -> +{velocity_bonus:.3f}")
        
        # Check if we have contact while rotating
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        if force_magnitude > 0.1 and angular_velocity_magnitude > 0.05:
            rotation_reward += 1.0
            info['messages'].append(f"Grasp while rotating -> +1.0")
        
        info['total_reward'] = rotation_reward
        return rotation_reward, info
    
    def calculate_reward(self, action: np.ndarray = None) -> Tuple[float, Dict]:
        """
        Calculate total reward with detailed breakdown.
        
        Returns:
            total_reward: Combined reward value
            info: Detailed breakdown of reward components
        """
        # Calculate individual rewards
        grasp_reward, grasp_info = self._calculate_grasp_reward()
        manipulation_reward, manip_info = self._calculate_manipulation_reward()
        
        # Weighted total
        total_reward = grasp_reward * 0.5 + manipulation_reward * 0.5
        
        info = {
            'total_reward': total_reward,
            'grasp_reward': grasp_reward,
            'manipulation_reward': manipulation_reward,
            'grasp_info': grasp_info,
            'manipulation_info': manip_info
        }
        
        return total_reward, info
    
    # ==================== HELPER FUNCTIONS ====================
    
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
    
    def _get_cube_angular_velocity(self) -> np.ndarray:
        """Get current cube angular velocity."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.cvel[cube_body_id][:3].copy()
        return np.zeros(3)
    
    # ==================== CONTROL FUNCTIONS ====================
    
    def adjust_actuator(self, increment: float):
        """Adjust the currently selected actuator."""
        if 0 <= self.selected_actuator_idx < len(self.hand_actuators):
            actual_idx = self.hand_actuators[self.selected_actuator_idx]
            ctrl_range = self.model.actuator_ctrlrange[actual_idx]
            current_value = self.data.ctrl[actual_idx]
            new_value = np.clip(current_value + increment, ctrl_range[0], ctrl_range[1])
            self.data.ctrl[actual_idx] = new_value
    
    def select_next_actuator(self):
        """Select the next actuator."""
        self.selected_actuator_idx = (self.selected_actuator_idx + 1) % len(self.hand_actuators)
    
    def select_prev_actuator(self):
        """Select the previous actuator."""
        self.selected_actuator_idx = (self.selected_actuator_idx - 1) % len(self.hand_actuators)
    
    def reset_all_actuators(self):
        """Reset all actuators to neutral pose."""
        self._set_neutral_pose()
    
    def reset_current_actuator(self):
        """Reset current actuator to 0."""
        if 0 <= self.selected_actuator_idx < len(self.hand_actuators):
            actual_idx = self.hand_actuators[self.selected_actuator_idx]
            self.data.ctrl[actual_idx] = 0.0
    
    # ==================== DISPLAY FUNCTIONS ====================
    
    def get_display_text(self) -> str:
        """Generate text for overlay display."""
        lines = []
        
        # Help text
        if self.show_help:
            lines.append("=== INTERACTIVE REWARD VIEWER ===")
            lines.append("")
            lines.append("CONTROLS:")
            lines.append("  TAB / SHIFT+TAB  : Select next/previous actuator")
            lines.append("  UP / DOWN        : Increase/decrease selected actuator")
            lines.append("  LEFT / RIGHT     : Large increase/decrease")
            lines.append("  R                : Reset current actuator to 0")
            lines.append("  SPACE            : Reset all to neutral pose")
            lines.append("  H                : Toggle this help")
            lines.append("  ESC              : Exit")
            lines.append("")
        
        # Current actuator info
        if 0 <= self.selected_actuator_idx < len(self.hand_actuators):
            actual_idx = self.hand_actuators[self.selected_actuator_idx]
            actuator_name = self.actuator_names[actual_idx]
            ctrl_value = self.data.ctrl[actual_idx]
            ctrl_range = self.model.actuator_ctrlrange[actual_idx]
            
            lines.append(f"SELECTED ACTUATOR [{self.selected_actuator_idx+1}/{len(self.hand_actuators)}]:")
            lines.append(f"  Name:  {actuator_name}")
            lines.append(f"  Value: {ctrl_value:.3f} (range: [{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}])")
            lines.append("")
        
        # Contact forces
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        lines.append("CONTACT FORCES:")
        lines.append(f"  Force vector: [{contact_force[0]:+.3f}, {contact_force[1]:+.3f}, {contact_force[2]:+.3f}]")
        lines.append(f"  Magnitude:    {force_magnitude:.3f}")
        lines.append("")
        
        # Rewards
        total_reward, reward_info = self.calculate_reward()
        
        lines.append("REWARDS:")
        lines.append(f"  Grasp Reward:        {reward_info['grasp_reward']:+.3f}")
        lines.append(f"    {reward_info['grasp_info']['message']}")
        lines.append(f"  Manipulation Reward: {reward_info['manipulation_reward']:+.3f}")
        for msg in reward_info['manipulation_info']['messages']:
            lines.append(f"    {msg}")
        lines.append(f"  TOTAL REWARD:        {total_reward:+.3f}")
        lines.append("")
        
        # Cube state
        cube_pos = self._get_cube_position()
        cube_ang_vel = self._get_cube_angular_velocity()
        lines.append("CUBE STATE:")
        lines.append(f"  Position:    [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        lines.append(f"  Angular vel: [{cube_ang_vel[0]:.3f}, {cube_ang_vel[1]:.3f}, {cube_ang_vel[2]:.3f}]")
        lines.append(f"  Rotation:    {reward_info['manipulation_info']['rotation_angle']:.3f} rad")
        
        return "\n".join(lines)
    
    def print_status(self):
        """Print current status to console."""
        print("\n" + "="*60)
        print(self.get_display_text())
        print("="*60)
    
    # ==================== MAIN LOOP ====================
    
    def run(self):
        """Run the interactive viewer."""
        print("\nStarting interactive viewer...")
        print("Close the viewer window or press Ctrl+C to exit.")
        
        def key_callback(keycode):
            """Handle keyboard input."""
            # Convert keycode to character
            if keycode == 256:  # ESC
                return False
            elif keycode == 72 or keycode == 104:  # H or h
                self.show_help = not self.show_help
            elif keycode == 258:  # TAB
                self.select_next_actuator()
            elif keycode == 265:  # UP arrow
                self.adjust_actuator(self.control_increment)
            elif keycode == 264:  # DOWN arrow
                self.adjust_actuator(-self.control_increment)
            elif keycode == 263:  # LEFT arrow
                self.adjust_actuator(-self.control_increment * 5)
            elif keycode == 262:  # RIGHT arrow
                self.adjust_actuator(self.control_increment * 5)
            elif keycode == 82 or keycode == 114:  # R or r
                self.reset_current_actuator()
            elif keycode == 32:  # SPACE
                self.reset_all_actuators()
            
            return True
        
        # Launch viewer with passive mode
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # Set camera for better view
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0.2, 0.0, 0.3]
            
            step_count = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # Step simulation
                mj.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Print status periodically
                if step_count % 50 == 0:
                    self.print_status()
                
                step_count += 1
                
                # Maintain real-time rate
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Interactive Reward Viewer for MuJoCo')
    parser.add_argument('--xml', type=str, default='xmls/bidexhands.xml',
                       help='Path to MuJoCo XML file')
    args = parser.parse_args()
    
    try:
        viewer = InteractiveRewardViewer(xml_path=args.xml)
        viewer.run()
    except KeyboardInterrupt:
        print("\nViewer closed by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

