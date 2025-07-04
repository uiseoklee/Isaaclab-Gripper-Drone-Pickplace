# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import matplotlib.pyplot as plt
import os
import datetime


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 7
    observation_space = 17
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "panda_hand_joint": ImplicitActuatorCfg(
                joint_names_expr=["panda_hand_joint"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_finger_joint": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=500.0,
                velocity_limit=0.1,
                stiffness=5e3,
                damping=3e2,
            ),
        })
    
    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        )           
    )

    # cube
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.15, 0.15, 0.15),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00003, density=None),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        )
    )

    thrust_to_weight = 1.9
    moment_scale = 0.00005

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "cube_gripper_distance",
                "cube_lifted",
                "goal_distance",               
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize gripper-related indices
        self.hand_link_idx = self._robot.find_bodies("panda_hand")[0]
        self.left_finger_idx = self._robot.find_bodies("panda_leftfinger")[0]
        self.right_finger_idx = self._robot.find_bodies("panda_rightfinger")[0]
        
        # Track cube grabbing state
        self._keep_gripper_closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cube_grabbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Gripper-cube related state variables (prevent duplicate calculations)
        self._gripper_center = torch.zeros(self.num_envs, 3, device=self.device)
        self._cube_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gripper_to_cube = torch.zeros(self.num_envs, 3, device=self.device)

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Add data storage for reward graphs
        self.reward_history = {
            "episodes": [],
            "lin_vel": [],
            "ang_vel": [],
            "cube_gripper_distance": [],
            "cube_lifted": [],
            "goal_distance": [],
            "total_reward": []
        }
        self.episode_counter = 0        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cube = RigidObject(self.cfg.cube)
        self._table = RigidObject(self.cfg.table)        
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.rigid_objects["table"] = self._table        

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Create indices for all environments
        env_ids = torch.arange(self.num_envs, device=self.device)
        
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # Thrust control
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        
        # Moment control
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]
        
        # Gripper joint control
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = self._robot.data.joint_vel.clone()
        
        # Hand joint control
        hand_joint_idx = self._robot.find_joints("panda_hand_joint")[0]
        if isinstance(hand_joint_idx, (list, torch.Tensor)):
            hand_joint_idx = hand_joint_idx[0]
        
        # Only allow rotation in environments where cube is not grabbed
        not_grabbed_mask = ~self._cube_grabbed
        if torch.any(not_grabbed_mask):
            # Apply rotation only to environments where cube is not grabbed
            joint_pos[not_grabbed_mask, hand_joint_idx] += self._actions[not_grabbed_mask, 4] * 0.1
        
        # When cube is grabbed, maintain current position (no additional rotation)
        # Set velocity to zero to prevent rotation
        if torch.any(self._cube_grabbed):
            joint_vel[self._cube_grabbed, hand_joint_idx] = 0.0
        
        # Finger joint control - control each finger independently
        finger1_joint_idx = self._robot.find_joints("panda_finger_joint1")[0]
        finger2_joint_idx = self._robot.find_joints("panda_finger_joint2")[0]
        
        if isinstance(finger1_joint_idx, (list, torch.Tensor)):
            finger1_joint_idx = finger1_joint_idx[0]
        if isinstance(finger2_joint_idx, (list, torch.Tensor)):
            finger2_joint_idx = finger2_joint_idx[0]
        
        # Independent finger control
        left_grip_action = self._actions[:, 5].unsqueeze(1)   # Left finger action
        right_grip_action = self._actions[:, 6].unsqueeze(1)  # Right finger action
        
        # Further restrict finger joint range (improve stability)
        min_grip = 0.003  # Min grip distance
        max_grip = 0.008  # Max grip distance
        
        # Calculate finger positions with range limits
        left_grip_pos = 0.01 - 0.007 * (left_grip_action + 1.0) / 2.0
        right_grip_pos = 0.01 - 0.007 * (right_grip_action + 1.0) / 2.0
        
        # Adaptive grip adjustment based on distance to cube
        if hasattr(self, "_gripper_to_cube"):
            cube_distance = torch.norm(self._gripper_to_cube, dim=1)
            close_to_cube = (cube_distance < 0.03).unsqueeze(1)
            
            # More precise grip control near cube
            if torch.any(close_to_cube):
                # Apply narrower grip range when close to cube
                left_grip_pos = torch.where(
                    close_to_cube, 
                    torch.clamp(left_grip_pos, min_grip, max_grip),
                    left_grip_pos
                )
                right_grip_pos = torch.where(
                    close_to_cube, 
                    torch.clamp(right_grip_pos, min_grip, max_grip),
                    right_grip_pos
                )
        
        # Apply positions independently to each finger
        joint_pos[:, finger1_joint_idx] = left_grip_pos.squeeze()
        joint_pos[:, finger2_joint_idx] = right_grip_pos.squeeze()
        
        # Update joint state
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Calculate gripper-cube relationship"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # Calculate gripper center position
        left_finger_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_idx]
        right_finger_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_idx]
        
        if len(left_finger_pos.shape) == 3:
            left_finger_pos = left_finger_pos.squeeze(1)
        if len(right_finger_pos.shape) == 3:
            right_finger_pos = right_finger_pos.squeeze(1)
        
        # Calculate gripper center
        if env_ids.shape[0] == self.num_envs:
            self._gripper_center = (left_finger_pos + right_finger_pos) / 2
        else:
            self._gripper_center[env_ids] = (left_finger_pos + right_finger_pos) / 2
        
        # Cube position
        cube_pos_w = self._cube.data.root_state_w[env_ids, :3]
        
        # Gripper-to-cube relative position
        if env_ids.shape[0] == self.num_envs:
            self._gripper_to_cube = torch.abs(self._gripper_center - cube_pos_w)
        else:
            self._gripper_to_cube[env_ids] = torch.abs(self._gripper_center[env_ids] - cube_pos_w)
        
        # Check if cube is lifted
        table_height = 0.05
        cube_height = cube_pos_w[:, 2] - table_height
        is_lifted = cube_height > 0.06
        
        # Update for all environments
        if env_ids.shape[0] == self.num_envs:
            self._cube_lifted = is_lifted
        else:
            self._cube_lifted[env_ids] = is_lifted
        
        # Improved cube grabbing detection
        finger_distance = torch.norm(right_finger_pos - left_finger_pos, dim=1)
        cube_distance = torch.norm(self._gripper_center[env_ids] - cube_pos_w, dim=1)
        
        # Optimal grip distance considering cube size
        optimal_grip_distance = 0.0045  # Optimal grip distance for cube size
        grip_tolerance = 0.003  # Tolerance
        
        # Calculate grip quality score (closer to 1 = better grip)
        grip_quality = 1.0 - torch.abs(finger_distance - optimal_grip_distance) / grip_tolerance
        grip_quality = torch.clamp(grip_quality, 0.0, 1.0)
        
        # Check for stable grip formation (with hysteresis)
        if not hasattr(self, "_prev_grabbed"):
            self._prev_grabbed = torch.zeros_like(is_lifted, dtype=torch.bool)
            self._grip_stability = torch.zeros_like(is_lifted, dtype=torch.float)
        
        # Basic grip condition
        basic_grip = (finger_distance < 0.008) & (finger_distance > 0.003) & (cube_distance < 0.012) & is_lifted
        
        # Include previous state (add hysteresis)
        stable_grip = self._prev_grabbed & (finger_distance < 0.01) & (cube_distance < 0.018) & is_lifted
        
        # Combine conditions
        is_grabbed = basic_grip | stable_grip
        
        # Update grip stability score
        self._grip_stability = torch.where(
            is_grabbed,
            torch.clamp(self._grip_stability + 0.1, 0.0, 1.0),  # Increase stability if grabbed
            torch.zeros_like(self._grip_stability)  # Reset if dropped
        )
        
        # Only consider grips maintained for some time
        is_grabbed = is_grabbed & (self._grip_stability > 0.2)  # Only accept grips maintained for a while
        
        # Update states
        self._prev_grabbed = is_grabbed
        self._cube_grabbed[env_ids] = is_grabbed

    def _apply_action(self):
        # Apply base thrust and moment
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        # For environments with grabbed cube, maintain closed gripper
        if torch.any(self._cube_grabbed):
            grabbed_env_ids = torch.where(self._cube_grabbed)[0]
            joint_pos = self._robot.data.joint_pos.clone()
            joint_vel = self._robot.data.joint_vel.clone()
            
            finger1_joint_idx = self._robot.find_joints("panda_finger_joint1")[0]
            finger2_joint_idx = self._robot.find_joints("panda_finger_joint2")[0]
            
            if isinstance(finger1_joint_idx, (list, torch.Tensor)):
                finger1_joint_idx = finger1_joint_idx[0]
            if isinstance(finger2_joint_idx, (list, torch.Tensor)):
                finger2_joint_idx = finger2_joint_idx[0]
            
            # Store current finger positions (maintain initial grip state)
            if not hasattr(self, "_grip_positions"):
                self._grip_positions = torch.zeros((self.num_envs, 2), device=self.device)
                self._grip_positions[:, 0] = 0.0035  # Optimal grip position
                self._grip_positions[:, 1] = 0.0035  # Optimal grip position
                
            for idx in grabbed_env_ids:
                if not self._keep_gripper_closed[idx]:
                    optimal_grip = 0.0035
                    self._grip_positions[idx, 0] = optimal_grip
                    self._grip_positions[idx, 1] = optimal_grip
                    self._keep_gripper_closed[idx] = True
            
            # Force apply grip position for each environment (stronger grip)
            for idx in grabbed_env_ids:
                joint_pos[idx, finger1_joint_idx] = self._grip_positions[idx, 0] * 0.85  
                joint_pos[idx, finger2_joint_idx] = self._grip_positions[idx, 1] * 0.85
                joint_vel[idx, finger1_joint_idx] = 0.0
                joint_vel[idx, finger2_joint_idx] = 0.0
            
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None)

    def _get_observations(self) -> dict:
        # First update gripper-cube relationship by calculating intermediate values
        self._compute_intermediate_values()
        
        # Basic drone observations
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # Combine observations
        obs = torch.cat(
            [
                # Basic drone state
                self._robot.data.root_lin_vel_b,       # Linear velocity (3D)
                self._robot.data.root_ang_vel_b,       # Angular velocity (3D) 
                self._robot.data.projected_gravity_b,  # Gravity direction (3D)
                desired_pos_b,                        # Target relative position (3D)
                
                # Cube related information
                self._gripper_to_cube,                # Gripper-to-cube relative position (3D)   
                self._cube_lifted.unsqueeze(1),       # Is cube lifted flag (1D)
                self._cube_grabbed.unsqueeze(1),      # Is cube grabbed flag (1D)
            ],
            dim=-1,
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Get values directly from properties for reward calculation
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        
        # 1. Gripper-cube distance reward
        cube_gripper_distance = torch.norm(self._gripper_to_cube, dim=1)
        distance_std = 0.3
        cube_gripper_reward = 1 - torch.tanh(cube_gripper_distance / distance_std)
        
        # 2. Cube lifted reward
        lifted_mask = self._cube_lifted
        
        # Estimate height (use lifted state as exact height is not in observations)
        lift_reward = torch.zeros_like(cube_gripper_distance)
        lift_reward[lifted_mask] = 25.0  # Fixed reward for lifted state
        
        # 3. Cube-to-goal distance reward (using robot-to-goal distance)
        distance_to_goal = torch.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        goal_std = 0.8
        goal_distance_reward = torch.zeros_like(distance_to_goal)
        
        # Only apply goal distance reward when cube is grabbed
        goal_reward_mask = self._cube_grabbed
        goal_distance_reward[goal_reward_mask] = (1.0 - torch.tanh(distance_to_goal[goal_reward_mask] / goal_std))
        goal_distance_reward = goal_distance_reward * self.cfg.distance_to_goal_reward_scale

        # Basic reward calculation
        rewards = {
            "lin_vel": lin_vel * -50.0, # Linear velocity penalty
            "ang_vel": ang_vel * -10.0, # Angular velocity penalty
            "cube_gripper_distance": cube_gripper_reward * 15.0, # Approach to cube
            "cube_lifted": lift_reward * 80, # Lifting the cube
            "goal_distance": goal_distance_reward * 5000, # Approaching goal with grabbed cube
        }
        
        # Safe reward aggregation
        total_reward = torch.zeros_like(lin_vel)
        for value in rewards.values():
            total_reward += value
        
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
            else:
                self._episode_sums[key] = value

        # Log reward changes
        self.log_reward_changes(rewards, total_reward)

        return total_reward * self.step_dt

    def log_reward_changes(self, rewards, total_reward=None):
        # Calculate total reward
        if total_reward is None:
            reward_values = [value.mean().item() for value in rewards.values()]
            total_reward = sum(reward_values)
        else:
            if isinstance(total_reward, torch.Tensor):
                total_reward = total_reward.mean().item()
        
        # Calculate absolute value sum
        reward_abs_values = [abs(value.mean().item()) for value in rewards.values()]
        abs_sum = sum(reward_abs_values)
        
        # Prevent division by zero
        eps = 1e-10
        safe_abs_sum = max(abs_sum, eps)
        
        # Store or compare initial reward values
        if not hasattr(self, "_initial_rewards") or self._initial_rewards is None:
            self._initial_rewards = {key: value.mean().item() for key, value in rewards.items()}
            initial_total = sum(self._initial_rewards.values())
            initial_abs_values = [abs(value) for value in self._initial_rewards.values()]
            initial_abs_sum = sum(initial_abs_values)
            safe_initial_abs_sum = max(initial_abs_sum, eps)
            
            print(f"Initial total reward: {initial_total:.4f}")
            print("Storing initial reward values:")
            for key, value in self._initial_rewards.items():
                percentage = (abs(value) / safe_initial_abs_sum) * 100
                sign_str = "" if value >= 0 else "-"
                print(f"Initial Reward {key}: {value:.4f} ({sign_str}{percentage:.2f}% contribution)")
        else:
            print(f"\nTotal reward: {total_reward:.4f}")
            print("Current reward values (contribution %, change from initial):")
            for key, value in rewards.items():
                current_value = value.mean().item()
                initial_value = self._initial_rewards.get(key, 0.0)
                
                # Calculate contribution
                abs_percentage = (abs(current_value) / safe_abs_sum) * 100
                sign_str = "" if current_value >= 0 else "-"
                
                # Calculate change rate
                if abs(initial_value) > eps:
                    percentage_change = ((current_value - initial_value) / abs(initial_value)) * 100
                    change_sign = "+" if percentage_change >= 0 else ""
                    print(f"Reward {key}: {current_value:.4f} ({sign_str}{abs_percentage:.2f}% contribution, {change_sign}{percentage_change:.2f}% change)")
                else:
                    if abs(current_value) <= eps:
                        print(f"Reward {key}: {current_value:.4f} ({sign_str}{abs_percentage:.2f}% contribution, no change)")
                    else:
                        direction = "increase" if current_value > 0 else "decrease"
                        print(f"Reward {key}: {current_value:.4f} ({sign_str}{abs_percentage:.2f}% contribution, significant {direction})")

        # Record episode reward data
        if hasattr(self, "reward_history"):
            # Finalize total reward
            if total_reward is None:
                total_reward = sum([value.mean().item() for value in rewards.values()])
            elif isinstance(total_reward, torch.Tensor):
                total_reward = total_reward.mean().item()
            
            # Store data
            self.reward_history["episodes"].append(self.episode_counter)
            self.reward_history["total_reward"].append(total_reward)
            
            # Store each reward component
            for key, value in rewards.items():
                if key in self.reward_history:
                    self.reward_history[key].append(value.mean().item())
            
            # Increment episode counter
            self.episode_counter += 1                        

    def plot_reward_graph(self, save=True, show=False, filename=None):
        """Generate and save/display reward graphs. Filter extreme values for visualization.
        
        Args:
            save (bool): Whether to save the graph
            show (bool): Whether to display the graph
            filename (str): Specific filename (None for auto-generated)
        """
        if len(self.reward_history["episodes"]) == 0:
            print("No reward data available to plot.")
            return
            
        try:
            import numpy as np  # Required for filtering
            
            # Create graph with increased rows for additional plots
            plt.figure(figsize=(20, 20))
            
            # Individual reward components
            reward_keys = ["lin_vel", "ang_vel", "cube_gripper_distance", "cube_lifted", "goal_distance"]
            
            # 1. Individual reward graphs
            for i, key in enumerate(reward_keys):
                if key in self.reward_history and len(self.reward_history[key]) > 0:
                    plt.subplot(4, 2, i+1)
                    plt.plot(self.reward_history["episodes"], self.reward_history[key], 'b-')
                    plt.title(f'Reward: {key}')
                    plt.xlabel('Episodes')
                    plt.ylabel('Reward Value')
                    plt.grid(True)
            
            # 2. Total reward graph
            plt.subplot(4, 2, 6)
            plt.plot(self.reward_history["episodes"], self.reward_history["total_reward"], 'r-')
            plt.title('Total Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.grid(True)
            
            # 3. Compare all rewards in one graph (absolute value)
            plt.subplot(4, 2, 7)
            colors = ['b', 'g', 'r', 'c', 'm']
            for i, key in enumerate(reward_keys):
                if key in self.reward_history and len(self.reward_history[key]) > 0:
                    # Differentiate lines based on scale
                    data = self.reward_history[key]
                    min_val = min(data) if data else 0
                    max_val = max(data) if data else 0
                    if min_val == max_val:
                        max_val = min_val + 1
                    
                    line_style = '-' if max_val - min_val > 10 else '--' if max_val - min_val > 1 else '-.'
                    line_width = 1 if abs(min_val) < 100 and abs(max_val) < 100 else 2
                    
                    plt.plot(
                        self.reward_history["episodes"], 
                        data, 
                        color=colors[i % len(colors)], 
                        linestyle=line_style,
                        linewidth=line_width,
                        label=key
                    )
            plt.title('Compare all rewards (absolute value)')
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.grid(True)
            plt.legend(loc='best')
            
            # 4. Reward contribution percentage graph
            plt.subplot(4, 2, 8)
            
            # Calculate total absolute rewards for each episode
            total_abs_rewards = []
            for ep_idx in range(len(self.reward_history["episodes"])):
                ep_abs_sum = 0
                for key in reward_keys:
                    if key in self.reward_history and ep_idx < len(self.reward_history[key]):
                        ep_abs_sum += abs(self.reward_history[key][ep_idx])
                total_abs_rewards.append(max(ep_abs_sum, 1e-10))  # Prevent division by zero
                
            # Generate and plot percentage data
            for i, key in enumerate(reward_keys):
                if key in self.reward_history and len(self.reward_history[key]) > 0:
                    # Calculate percentages
                    percentage_data = []
                    for ep_idx in range(len(self.reward_history["episodes"])):
                        if ep_idx < len(self.reward_history[key]):
                            percentage = (abs(self.reward_history[key][ep_idx]) / total_abs_rewards[ep_idx]) * 100
                            percentage_data.append(percentage)
                        else:
                            percentage_data.append(0)
                    
                    plt.plot(
                        self.reward_history["episodes"], 
                        percentage_data, 
                        color=colors[i % len(colors)],
                        linestyle='-',
                        linewidth=2, 
                        label=key
                    )
            plt.title('Compensation Contribution Percentage (%)')
            plt.xlabel('Episodes')
            plt.ylabel('Contribution (%)')
            plt.ylim(0, 100)  # Set y-axis range to 0-100%
            plt.grid(True)
            plt.legend(loc='best')
            
            plt.tight_layout()
            
            # Save graph
            if save:
                # Create directory if needed
                save_dir = os.path.expanduser("~/reward_graphs")
                os.makedirs(save_dir, exist_ok=True)
                
                # Create filename with episode range
                if filename is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    start_ep = 0
                    end_ep = self.episode_counter
                    filename = f"reward_graph_{timestamp}_ep{start_ep}-{end_ep}.png"
                    
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath)
                print(f"Reward graph saved: {filepath}")
            
            # Display graph
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error occurred while creating graph: {e}")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Basic termination conditions
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        
        # Goal reaching success condition
        robot_pos = self._robot.data.root_pos_w
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - robot_pos, dim=1)
        
        # Success condition: cube is grabbed and close to goal position
        success = self._cube_grabbed & (distance_to_goal < 0.05)

        terminated = torch.logical_or(died, success)
        
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # Additional: cube related final statistics
        if env_ids is not None and len(env_ids) > 0:
            cube_pos = self._cube.data.root_state_w[env_ids, :3]
            final_cube_height = (cube_pos[:, 2] - 0.05).mean()  # Height from table
            final_cube_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - cube_pos, dim=1).mean()
            
            # Additional logging
            if "Metrics" not in self.extras["log"]:
                self.extras["log"]["Metrics"] = {}
            self.extras["log"]["Metrics"]["final_cube_height"] = final_cube_height.item()
            self.extras["log"]["Metrics"]["final_cube_to_goal"] = final_cube_to_goal.item()        

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Initialize table
        table_state = torch.zeros((len(env_ids), 13), device=self.device)
        table_state[:, :3] = self._terrain.env_origins[env_ids]
        table_state[:, 2] += 0.05  # Table height center
        table_state[:, 3] = 1.0    # Quaternion w
        self._table.write_root_state_to_sim(table_state, env_ids)
        
        # Initialize cube
        cube_state = torch.zeros((len(env_ids), 13), device=self.device)
        cube_state[:, :3] = self._terrain.env_origins[env_ids]
        cube_state[:, 2] += 0.075  # Cube height on table
        cube_state[:, 3] = 1.0     # Quaternion w
        self._cube.write_root_state_to_sim(cube_state, env_ids)

        # Reset gripper state
        self._keep_gripper_closed[env_ids] = False
        
        # Determine whether to create graph
        should_create_graph = False

        if self.episode_counter <= 1000:
            # Before 1000 episodes: create graph every 100 episodes (100, 200, ..., 900, 1000)
            should_create_graph = self.episode_counter % 100 == 0
            interval_text = "100"
        elif self.episode_counter <= 10000:
            # 1000~10000 episodes: create graph every 1000 episodes (2000, 3000, ..., 9000, 10000)
            should_create_graph = self.episode_counter % 1000 == 0
            interval_text = "1000"
        else:
            # After 10000 episodes
            # Check for exact multiples of 10000
            is_exact_multiple = self.episode_counter % 10000 == 0
            
            # Calculate previous and next 10000 milestones
            prev_milestone = (self.episode_counter // 10000) * 10000
            next_milestone = prev_milestone + 10000
            
            # 1. Exactly multiple of 10000, or 
            # 2. Graph wasn't created at the last milestone
            if is_exact_multiple:
                should_create_graph = True
                milestone_ep = self.episode_counter
            elif hasattr(self, "last_graph_episode"):
                # Check if the graph was missed at the previous milestone
                if self.last_graph_episode < prev_milestone and self.episode_counter > prev_milestone:
                    should_create_graph = True
                    milestone_ep = prev_milestone
                    print(f"Missed creating graph at milestone ({prev_milestone}). Creating now.")
                
            interval_text = "10000"

        # Create graph if needed
        if should_create_graph:
            # Track last graph creation episode
            if not hasattr(self, "last_graph_episode") or self.last_graph_episode is None:
                self.last_graph_episode = 0
            
            # Only create graph if episodes have advanced since last creation
            if self.episode_counter > self.last_graph_episode:
                timestamp = datetime.datetime.now().strftime("%Y%m%d")
                current_ep = self.episode_counter
                
                # Add interval info to filename
                filename = f"reward_graph_{timestamp}_ep{current_ep}_interval{interval_text}.png"
                
                # Create graph
                try:
                    self.plot_reward_graph(save=True, show=False, filename=filename)
                    print(f"Graph successfully created at episode {current_ep} (interval: {interval_text})")
                    
                    # Update last graph creation episode
                    self.last_graph_episode = current_ep
                except Exception as e:
                    print(f"Error creating graph: {e}")

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # Set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
