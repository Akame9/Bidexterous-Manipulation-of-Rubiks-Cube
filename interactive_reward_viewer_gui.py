"""
Interactive MuJoCo Viewer with GUI Overlay for Reward and Contact Force Visualization

This version includes an on-screen overlay showing real-time information about:
- Contact forces between hands and cube
- Grasp reward
- Manipulation reward
- Total reward
- Current actuator controls

Controls:
- Arrow Keys UP/DOWN: Adjust selected actuator value
- Arrow Keys LEFT/RIGHT: Large adjustments
- TAB: Select next actuator
- SHIFT+TAB: Select previous actuator
- R: Reset current actuator to 0
- SPACE: Reset all actuators to neutral pose
- H: Toggle help display
- ESC: Exit
"""

import numpy as np
import mujoco as mj
import mujoco.viewer
import time
import argparse
from typing import Dict, Tuple
from collections import deque
import glfw


class InteractiveRewardViewerGUI:
    """
    Interactive viewer with GUI overlay for exploring contact forces and rewards.
    """
    
    def __init__(self, xml_path="xmls/bidexhands.xml"):
        """Initialize the interactive viewer with GUI."""
        self.xml_path = xml_path
        
        # Load MuJoCo model and data
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Setup actuators
        self._setup_actuators()
        
        # Target cube configuration (for reward calculation)
        self.target_cube_config = self._get_initial_cube_config()
        
        # Control state
        self.selected_actuator_idx = 0
        self.control_increment = 0.05
        
        # Display state
        self.show_help = True
        self.update_counter = 0
        
        # Initialize to neutral pose
        self._set_neutral_pose()
        self._set_initial_cube_pose()
        
        print(f"Interactive Viewer with GUI initialized!")
        print(f"Total hand actuators: {len(self.hand_actuators)}")
    
    def _setup_actuators(self):
        """Setup actuator information."""
        self.actuator_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) 
                              for i in range(self.model.nu)]
        
        self.left_hand_actuators = []
        self.right_hand_actuators = []
        self.cube_actuators = []
        self.hand_actuators = []
        
        for i, name in enumerate(self.actuator_names):
            if name.startswith('lh_A_'):
                self.left_hand_actuators.append(i)
                self.hand_actuators.append(i)
            elif name.startswith('rh_A_'):
                self.right_hand_actuators.append(i)
                self.hand_actuators.append(i)
            elif name in ['red', 'orange', 'blue', 'green', 'white', 'yellow']:
                self.cube_actuators.append(i)
        
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
        
        print(f"Found {len(self.cube_body_ids)} cube bodies")
    
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
        
        for _ in range(100):
            mj.mj_step(self.model, self.data)
    
    def _set_initial_cube_pose(self):
        """Set initial cube position."""
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
            # self.model.opt.gravity[:] = [0.0, 0.0, 0.0]
            mj.mj_forward(self.model, self.data)
    
    def _get_initial_cube_config(self) -> Dict:
        """Get initial cube configuration."""
        return {
            'position': self._get_cube_position(),
            'orientation': self._get_cube_orientation()
        }
    
    # ==================== REWARD FUNCTIONS ====================
    
    def _get_contact_forces(self) -> np.ndarray:
        """Get contact forces between hands and cube."""
        contact_force = np.zeros(3)
        ncon = self.data.ncon
        print(f"Number of contacts: {ncon}")
        
        if ncon > 0:
            efc_force = np.zeros(6)
            for i in range(ncon):
                contact = self.data.contact[i]
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
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
                
                if is_hand_cube_contact:
                    mj.mj_contactForce(self.model, self.data, i, efc_force)
                    print(f"Contact Force: {efc_force}")
                    print(f"Contact Force Magnitude: {np.linalg.norm(efc_force[0:3])}")
                    contact_force += efc_force[0:3]
        
        return contact_force
    
    def _calculate_grasp_reward(self) -> Tuple[float, Dict]:
        """Calculate grasping reward."""
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        info = {
            'force_magnitude': force_magnitude,
            'contact_force': contact_force.copy()
        }
        
        if force_magnitude > 0.05:
            if force_magnitude < 2.0:
                reward = 2.0
                info['message'] = "Gentle grasp [OK]"
            elif force_magnitude < 5.0:
                reward = 1.0
                info['message'] = "Moderate grasp"
            else:
                reward = 0.5
                info['message'] = "Excessive force!"
        else:
            reward = -0.5
            info['message'] = "No contact"
        
        return reward, info
    
    def _calculate_manipulation_reward(self) -> Tuple[float, Dict]:
        """Calculate manipulation reward."""
        cube_quat = self._get_cube_orientation()
        cube_ang_vel = self._get_cube_angular_velocity()
        
        initial_quat = self.target_cube_config['orientation']
        
        quat_dot = np.abs(np.dot(cube_quat, initial_quat))
        rotation_angle = 2 * np.arccos(np.clip(quat_dot, 0, 1))
        
        rotation_reward = min(rotation_angle / np.pi, 1.0) * 1.0
        
        info = {
            'rotation_angle': rotation_angle,
            'angular_velocity_magnitude': np.linalg.norm(cube_ang_vel),
            'reward_breakdown': []
        }
        
        angular_velocity_magnitude = np.linalg.norm(cube_ang_vel)
        if angular_velocity_magnitude > 0.1:
            velocity_bonus = min(angular_velocity_magnitude, 2.0) * 0.5
            rotation_reward += velocity_bonus
            info['reward_breakdown'].append(f"Velocity bonus: +{velocity_bonus:.2f}")
        
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        
        if force_magnitude > 0.1 and angular_velocity_magnitude > 0.05:
            rotation_reward += 1.0
            info['reward_breakdown'].append("Grasp+rotate: +1.0")
        
        return rotation_reward, info
    
    def calculate_reward(self) -> Tuple[float, Dict]:
        """Calculate total reward."""
        grasp_reward, grasp_info = self._calculate_grasp_reward()
        
        total_reward = grasp_reward
        
        info = {
            'total_reward': total_reward,
            'grasp_reward': grasp_reward,
            'grasp_info': grasp_info
        }
        
        return total_reward, info
    
    # ==================== HELPER FUNCTIONS ====================
    
    def _get_cube_position(self) -> np.ndarray:
        """Get cube position."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.xpos[cube_body_id].copy()
        return np.zeros(3)
    
    def _get_cube_orientation(self) -> np.ndarray:
        """Get cube orientation."""
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
            return self.data.xquat[cube_body_id].copy()
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _get_cube_angular_velocity(self) -> np.ndarray:
        """Get cube angular velocity."""
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
            print(f"Actuator {self.actuator_names[actual_idx]}: {new_value:.3f}")
    
    def select_next_actuator(self):
        """Select next actuator."""
        self.selected_actuator_idx = (self.selected_actuator_idx + 1) % len(self.hand_actuators)
        actual_idx = self.hand_actuators[self.selected_actuator_idx]
        print(f"Selected: {self.actuator_names[actual_idx]}")
    
    def select_prev_actuator(self):
        """Select previous actuator."""
        self.selected_actuator_idx = (self.selected_actuator_idx - 1) % len(self.hand_actuators)
        actual_idx = self.hand_actuators[self.selected_actuator_idx]
        print(f"Selected: {self.actuator_names[actual_idx]}")
    
    def reset_current_actuator(self):
        """Reset current actuator."""
        if 0 <= self.selected_actuator_idx < len(self.hand_actuators):
            actual_idx = self.hand_actuators[self.selected_actuator_idx]
            self.data.ctrl[actual_idx] = 0.0
            print(f"Reset {self.actuator_names[actual_idx]} to 0.0")
    
    # ==================== DISPLAY ====================
    
    def render_overlay(self, viewport_width, viewport_height, context):
        """Render text overlay on the viewer."""
        # This would require custom rendering with MuJoCo's mjr functions
        # For simplicity, we'll print to console
        pass
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector (similar to rubiks_cube.py get_state method)."""
        # Joint positions and velocities
        joint_pos = self.data.qpos.copy()
        joint_vel = self.data.qvel.copy()
        
        # Cube state (position, orientation, velocities)
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "core")
        if cube_body_id != -1:
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
        
        # Contact forces
        contact_force = self._get_contact_forces()
        
        # Current hand actions (control values for hand actuators)
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
    
    def print_state_space(self):
        """Print detailed state space information."""
        state = self.get_state_vector()
        
        # Extract components from state vector
        joint_pos_len = self.model.nq
        joint_vel_len = self.model.nv
        
        joint_pos = state[:joint_pos_len]
        joint_vel = state[joint_pos_len:joint_pos_len + joint_vel_len]
        cube_pos = state[joint_pos_len + joint_vel_len:joint_pos_len + joint_vel_len + 3]
        cube_quat = state[joint_pos_len + joint_vel_len + 3:joint_pos_len + joint_vel_len + 7]
        cube_lin_vel = state[joint_pos_len + joint_vel_len + 7:joint_pos_len + joint_vel_len + 10]
        cube_ang_vel = state[joint_pos_len + joint_vel_len + 10:joint_pos_len + joint_vel_len + 13]
        contact_force = state[joint_pos_len + joint_vel_len + 13:joint_pos_len + joint_vel_len + 16]
        prev_actions = state[joint_pos_len + joint_vel_len + 16:]
        
        print("\n" + "=" * 80)
        print("STATE SPACE ANALYSIS")
        print("=" * 80)
        
        # Joint positions
        print(f"\nJOINT POSITIONS ({len(joint_pos)} values):")
        print(f"  Range: [{joint_pos.min():.4f}, {joint_pos.max():.4f}]")
        print(f"  Mean: {joint_pos.mean():.4f}, Std: {joint_pos.std():.4f}")
        print(f"  Sample values: {joint_pos[:5]} ... {joint_pos[-5:]}")
        
        # Joint velocities
        print(f"\nJOINT VELOCITIES ({len(joint_vel)} values):")
        print(f"  Range: [{joint_vel.min():.4f}, {joint_vel.max():.4f}]")
        print(f"  Mean: {joint_vel.mean():.4f}, Std: {joint_vel.std():.4f}")
        print(f"  Sample values: {joint_vel[:5]} ... {joint_vel[-5:]}")
        
        # Cube position
        print(f"\nCUBE POSITION (3 values):")
        print(f"  X: {cube_pos[0]:.4f}, Y: {cube_pos[1]:.4f}, Z: {cube_pos[2]:.4f}")
        print(f"  Distance from origin: {np.linalg.norm(cube_pos):.4f}")
        
        # Cube orientation (quaternion)
        print(f"\nCUBE ORIENTATION (4 values - quaternion):")
        print(f"  W: {cube_quat[0]:.4f}, X: {cube_quat[1]:.4f}, Y: {cube_quat[2]:.4f}, Z: {cube_quat[3]:.4f}")
        print(f"  Quaternion norm: {np.linalg.norm(cube_quat):.4f}")
        
        # Cube velocities
        print(f"\nCUBE LINEAR VELOCITY (3 values):")
        print(f"  X: {cube_lin_vel[0]:.4f}, Y: {cube_lin_vel[1]:.4f}, Z: {cube_lin_vel[2]:.4f}")
        print(f"  Speed: {np.linalg.norm(cube_lin_vel):.4f}")
        
        print(f"\nCUBE ANGULAR VELOCITY (3 values):")
        print(f"  X: {cube_ang_vel[0]:.4f}, Y: {cube_ang_vel[1]:.4f}, Z: {cube_ang_vel[2]:.4f}")
        print(f"  Angular speed: {np.linalg.norm(cube_ang_vel):.4f}")
        
        # Contact forces
        print(f"\nCONTACT FORCES (3 values):")
        print(f"  X: {contact_force[0]:.4f}, Y: {contact_force[1]:.4f}, Z: {contact_force[2]:.4f}")
        print(f"  Force magnitude: {np.linalg.norm(contact_force):.4f}")
        
        # Previous actions
        print(f"\nPREVIOUS ACTIONS ({len(prev_actions)} values):")
        print(f"  Range: [{prev_actions.min():.4f}, {prev_actions.max():.4f}]")
        print(f"  Mean: {prev_actions.mean():.4f}, Std: {prev_actions.std():.4f}")
        print(f"  Sample values: {prev_actions[:5]} ... {prev_actions[-5:]}")
        
        # Total state info
        print(f"\nTOTAL STATE VECTOR:")
        print(f"  Dimension: {len(state)}")
        print(f"  Range: [{state.min():.4f}, {state.max():.4f}]")
        print(f"  Mean: {state.mean():.4f}, Std: {state.std():.4f}")
        
        print("=" * 80)
    
    def print_status(self):
        """Print status to console."""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        print("=" * 70)
        print("INTERACTIVE REWARD VIEWER WITH GUI")
        print("=" * 70)
        
        if self.show_help:
            print("\nCONTROLS:")
            print("  UP/DOWN  : Increase/decrease actuator value")
            print("  LEFT/RIGHT : Large increase/decrease")
            print("  TAB      : Select next actuator")
            print("  R        : Reset current actuator")
            print("  SPACE    : Reset all to neutral")
            print("  H        : Toggle help")
            print("  S        : Print state space analysis")
            print("  ESC      : Exit")
            print()
        
        # Current actuator
        if 0 <= self.selected_actuator_idx < len(self.hand_actuators):
            actual_idx = self.hand_actuators[self.selected_actuator_idx]
            actuator_name = self.actuator_names[actual_idx]
            ctrl_value = self.data.ctrl[actual_idx]
            ctrl_range = self.model.actuator_ctrlrange[actual_idx]
            
            print(f"SELECTED ACTUATOR [{self.selected_actuator_idx+1}/{len(self.hand_actuators)}]:")
            print(f"  Name:  {actuator_name}")
            print(f"  Value: {ctrl_value:+.3f}  (range: [{ctrl_range[0]:+.2f}, {ctrl_range[1]:+.2f}])")
            
            # Progress bar
            if ctrl_range[1] != ctrl_range[0]:
                normalized = (ctrl_value - ctrl_range[0]) / (ctrl_range[1] - ctrl_range[0])
                bar_width = 40
                filled = int(normalized * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(f"  [{bar}] {normalized*100:.1f}%")
            print()
        
        # Contact forces
        contact_force = self._get_contact_forces()
        force_magnitude = np.linalg.norm(contact_force)
        print("CONTACT FORCES:")
        print(f"  Vector:    [{contact_force[0]:+7.3f}, {contact_force[1]:+7.3f}, {contact_force[2]:+7.3f}]")
        print(f"  Magnitude: {force_magnitude:7.3f}")
        print()
        
        # Rewards
        total_reward, reward_info = self.calculate_reward()
        
        print("REWARD:")
        print(f"  Grasp Reward: {reward_info['grasp_reward']:+7.3f}  ({reward_info['grasp_info']['message']})")
        print()
        
        # Cube state
        cube_pos = self._get_cube_position()
        cube_ang_vel = self._get_cube_angular_velocity()
        
        print("CUBE STATE:")
        print(f"  Position:    [{cube_pos[0]:+.3f}, {cube_pos[1]:+.3f}, {cube_pos[2]:+.3f}]")
        print(f"  Angular vel: {np.linalg.norm(cube_ang_vel):.3f} rad/s")
        
        print("=" * 70)
    
    # ==================== MAIN LOOP ====================
    
    def run(self):
        """Run the interactive viewer."""
        print("\nStarting interactive viewer...")
        print("Use the controls to manipulate actuators and see rewards!\n")
        
        # Keyboard state
        key_pressed = {}
        
        def key_callback(keycode):
            """Handle keyboard input."""
            if keycode == 256:  # ESC
                return False
            elif keycode in [72, 104]:  # H
                self.show_help = not self.show_help
                self.print_status()
            elif keycode in [83, 115]:  # S
                self.print_state_space()
            elif keycode == 258:  # TAB
                self.select_next_actuator()
                self.print_status()
            elif keycode == 265:  # UP
                self.adjust_actuator(self.control_increment)
            elif keycode == 264:  # DOWN
                self.adjust_actuator(-self.control_increment)
            elif keycode == 263:  # LEFT
                self.adjust_actuator(-self.control_increment * 3)
            elif keycode == 262:  # RIGHT
                self.adjust_actuator(self.control_increment * 3)
            elif keycode in [82, 114]:  # R
                self.reset_current_actuator()
                self.print_status()
            elif keycode == 32:  # SPACE
                print("Resetting all actuators to neutral pose...")
                self._set_neutral_pose()
                self.print_status()
            
            return True
        
        # Launch passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # Set camera
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0.2, 0.0, 0.3]
            
            # Print initial status
            self.print_status()
            
            step_count = 0
            last_print = time.time()
            
            while viewer.is_running():
                step_start = time.time()
                
                # Step simulation
                mj.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Print status every 0.5 seconds
                if time.time() - last_print > 0.5:
                    self.print_status()
                    last_print = time.time()
                
                step_count += 1
                
                # Maintain timestep
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Interactive Reward Viewer with GUI')
    parser.add_argument('--xml', type=str, default='xmls/bidexhands.xml',
                       help='Path to MuJoCo XML file')
    args = parser.parse_args()
    
    try:
        viewer = InteractiveRewardViewerGUI(xml_path=args.xml)
        viewer.run()
    except KeyboardInterrupt:
        print("\n\nViewer closed by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

