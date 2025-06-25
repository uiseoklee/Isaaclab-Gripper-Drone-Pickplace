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
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
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

        # 그리퍼 관련 인덱스 초기화
        self.hand_link_idx = self._robot.find_bodies("panda_hand")[0]
        self.left_finger_idx = self._robot.find_bodies("panda_leftfinger")[0]
        self.right_finger_idx = self._robot.find_bodies("panda_rightfinger")[0]
        
        # 큐브 잡기 상태 추적
        self._keep_gripper_closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cube_grabbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 그리퍼-큐브 관련 상태 변수 (중복 계산 방지)
        self._gripper_center = torch.zeros(self.num_envs, 3, device=self.device)
        self._cube_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gripper_to_cube = torch.zeros(self.num_envs, 3, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # 보상 그래프를 위한 데이터 저장소 추가
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
        # 모든 환경에 대한 인덱스 생성
        env_ids = torch.arange(self.num_envs, device=self.device)
        
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # 추력 제어
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        
        # 모멘트 제어
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]
        
        # 그리퍼 관절 제어
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = self._robot.data.joint_vel.clone()
        
        # 손 관절 제어
        hand_joint_idx = self._robot.find_joints("panda_hand_joint")[0]
        if isinstance(hand_joint_idx, (list, torch.Tensor)):
            hand_joint_idx = hand_joint_idx[0]
        
        # 큐브를 잡지 않은 환경에서만 회전 허용
        not_grabbed_mask = ~self._cube_grabbed
        if torch.any(not_grabbed_mask):
            # 잡지 않은 환경에만 회전 적용
            joint_pos[not_grabbed_mask, hand_joint_idx] += self._actions[not_grabbed_mask, 4] * 0.1
        
        # 큐브 잡은 상태에서는 현재 위치 유지 (추가 회전 없음)
        # 회전 방지를 위해 속도도 0으로 설정
        if torch.any(self._cube_grabbed):
            joint_vel[self._cube_grabbed, hand_joint_idx] = 0.0
        
        # 손가락 관절 제어 - 각 손가락 독립적으로 제어
        finger1_joint_idx = self._robot.find_joints("panda_finger_joint1")[0]
        finger2_joint_idx = self._robot.find_joints("panda_finger_joint2")[0]
        
        if isinstance(finger1_joint_idx, (list, torch.Tensor)):
            finger1_joint_idx = finger1_joint_idx[0]
        if isinstance(finger2_joint_idx, (list, torch.Tensor)):
            finger2_joint_idx = finger2_joint_idx[0]
        
        # 독립적 손가락 제어 - 평균을 사용하지 않고 각각 제어
        left_grip_action = self._actions[:, 5].unsqueeze(1)   # 왼쪽 손가락 액션
        right_grip_action = self._actions[:, 6].unsqueeze(1)  # 오른쪽 손가락 액션
        
        # 손가락 조인트 범위 더 제한 (안정성 향상)
        min_grip = 0.003  # 최소 그립 거리
        max_grip = 0.008  # 최대 그립 거리
        
        # 손가락 위치 계산 시 범위 제한
        left_grip_pos = 0.01 - 0.007 * (left_grip_action + 1.0) / 2.0
        right_grip_pos = 0.01 - 0.007 * (right_grip_action + 1.0) / 2.0
        
        # 현재 큐브와의 거리에 따라 적응형 그립 조절
        if hasattr(self, "_gripper_to_cube"):
            cube_distance = torch.norm(self._gripper_to_cube, dim=1)
            close_to_cube = (cube_distance < 0.03).unsqueeze(1)
            
            # 큐브 근처에서는 더 정밀한 그립 제어
            if torch.any(close_to_cube):
                # 큐브에 가까울 때 더 좁은 범위의 그립 허용
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
        
        # 각 손가락에 독립적으로 위치 적용
        joint_pos[:, finger1_joint_idx] = left_grip_pos.squeeze()
        joint_pos[:, finger2_joint_idx] = right_grip_pos.squeeze()
        
        # 관절 상태 업데이트
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """그리퍼와 큐브 간의 관계 계산"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 그리퍼 중심 위치 계산
        left_finger_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_idx]
        right_finger_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_idx]
        
        if len(left_finger_pos.shape) == 3:
            left_finger_pos = left_finger_pos.squeeze(1)
        if len(right_finger_pos.shape) == 3:
            right_finger_pos = right_finger_pos.squeeze(1)
        
        # 그리퍼 센터 계산
        if env_ids.shape[0] == self.num_envs:
            self._gripper_center = (left_finger_pos + right_finger_pos) / 2
        else:
            self._gripper_center[env_ids] = (left_finger_pos + right_finger_pos) / 2
        
        # 큐브 위치
        cube_pos_w = self._cube.data.root_state_w[env_ids, :3]
        
        # 그리퍼-큐브 상대 위치
        if env_ids.shape[0] == self.num_envs:
            self._gripper_to_cube = torch.abs(self._gripper_center - cube_pos_w)
        else:
            self._gripper_to_cube[env_ids] = torch.abs(self._gripper_center[env_ids] - cube_pos_w)
        
        # 큐브가 들림 여부 확인
        table_height = 0.05
        cube_height = cube_pos_w[:, 2] - table_height
        is_lifted = cube_height > 0.06
        
        # 전체 환경에 대해 업데이트
        if env_ids.shape[0] == self.num_envs:
            self._cube_lifted = is_lifted
        else:
            self._cube_lifted[env_ids] = is_lifted
        
        # 개선된 큐브 잡기 판단 로직
        finger_distance = torch.norm(right_finger_pos - left_finger_pos, dim=1)
        cube_distance = torch.norm(self._gripper_center[env_ids] - cube_pos_w, dim=1)
        
        # 큐브 크기 고려한 최적 그립 거리
        optimal_grip_distance = 0.0045  # 큐브 크기에 맞는 최적 그립 거리
        grip_tolerance = 0.003  # 허용 오차
        
        # 그립 품질 점수 계산 (1에 가까울수록 좋은 그립)
        grip_quality = 1.0 - torch.abs(finger_distance - optimal_grip_distance) / grip_tolerance
        grip_quality = torch.clamp(grip_quality, 0.0, 1.0)
        
        # 안정적인 그립 형성 여부 확인 (약간의 이력현상 포함)
        if not hasattr(self, "_prev_grabbed"):
            self._prev_grabbed = torch.zeros_like(is_lifted, dtype=torch.bool)
            self._grip_stability = torch.zeros_like(is_lifted, dtype=torch.float)
        
        # 기본 그립 조건
        basic_grip = (finger_distance < 0.008) & (finger_distance > 0.003) & (cube_distance < 0.012) & is_lifted
        
        # 이전에 잡고 있었던 상태 반영 (약간의 히스테리시스 추가)
        stable_grip = self._prev_grabbed & (finger_distance < 0.01) & (cube_distance < 0.018) & is_lifted
        
        # 두 조건 결합
        is_grabbed = basic_grip | stable_grip
        
        # 그립 안정성 점수 업데이트
        self._grip_stability = torch.where(
            is_grabbed,
            torch.clamp(self._grip_stability + 0.1, 0.0, 1.0),  # 잡고 있으면 안정성 증가
            torch.zeros_like(self._grip_stability)  # 놓치면 리셋
        )
        
        # 안정성 기간 기반 그립 보정
        is_grabbed = is_grabbed & (self._grip_stability > 0.2)  # 일정 시간 이상 유지된 그립만 인정
        
        # 상태 업데이트
        self._prev_grabbed = is_grabbed
        self._cube_grabbed[env_ids] = is_grabbed

    def _apply_action(self):
        # 기본 추력과 모멘트 적용
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        # 큐브를 잡고 있는 환경에 대해 그리퍼를 닫은 상태로 유지
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
            
            # 현재 손가락 위치 저장 (첫 잡았을 때의 상태 유지)
            if not hasattr(self, "_grip_positions"):
                self._grip_positions = torch.zeros((self.num_envs, 2), device=self.device)
                self._grip_positions[:, 0] = 0.0035  # 최적 그립 위치
                self._grip_positions[:, 1] = 0.0035  # 최적 그립 위치
                
            for idx in grabbed_env_ids:
                if not self._keep_gripper_closed[idx]:
                    optimal_grip = 0.0035
                    self._grip_positions[idx, 0] = optimal_grip
                    self._grip_positions[idx, 1] = optimal_grip
                    self._keep_gripper_closed[idx] = True
            
            # 각 환경마다 그립 위치 강제 적용 (더 강하게 잡도록)
            for idx in grabbed_env_ids:
                joint_pos[idx, finger1_joint_idx] = self._grip_positions[idx, 0] * 0.85  
                joint_pos[idx, finger2_joint_idx] = self._grip_positions[idx, 1] * 0.85
                joint_vel[idx, finger1_joint_idx] = 0.0
                joint_vel[idx, finger2_joint_idx] = 0.0
            
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None)

    def _get_observations(self) -> dict:
        # 먼저 중간값 계산하여 그리퍼-큐브 관계 업데이트
        self._compute_intermediate_values()
        
        # 기본 드론 관측값
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        
        # 큐브 정보 추가 (cube_pos_w와 cube_pos_b 계산은 유지하되, 관측값에는 포함하지 않음)
        cube_pos_w = self._cube.data.root_state_w[:, :3]
        # 아래 코드는 다른 계산에 필요할 수 있으므로 유지하지만, 관측값엔 포함하지 않음
        cube_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], cube_pos_w
        )

        # 관측값 결합 (cube_pos_b 제외)
        obs = torch.cat(
            [
                # 기본 드론 상태
                self._robot.data.root_lin_vel_b,       # 선속도 (3차원)
                self._robot.data.root_ang_vel_b,       # 각속도 (3차원) 
                self._robot.data.projected_gravity_b,  # 중력 방향 (3차원)
                desired_pos_b,                        # 목표 상대 위치 (3차원)
                
                # 큐브 관련 정보 (cube_pos_b 제거)
                self._gripper_to_cube,                # 그리퍼-큐브 상대 위치 (3차원)   
                self._cube_lifted.unsqueeze(1),       # 큐브 들림 여부 (1차원)
                self._cube_grabbed.unsqueeze(1),      # 큐브 잡힘 상태 (1차원)
            ],
            dim=-1,
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # observation 대신 직접 속성에서 값을 가져와 보상 계산
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        
        # 1. 그리퍼-큐브 거리 보상
        cube_gripper_distance = torch.norm(self._gripper_to_cube, dim=1)
        distance_std = 0.3
        cube_gripper_reward = 1 - torch.tanh(cube_gripper_distance / distance_std)
        
        # 2. 큐브 들림 보상
        lifted_mask = self._cube_lifted
        
        # 높이 추정 (정확한 높이는 observation에 없으므로 lifted 상태 활용)
        lift_reward = torch.zeros_like(cube_gripper_distance)
        lift_reward[lifted_mask] = 25.0  # 들린 경우 고정 보상 부여
        
        # 3. 큐브-목표 거리 보상 (로봇-목표 거리로 대체)
        distance_to_goal = torch.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        goal_std = 0.8
        goal_distance_reward = torch.zeros_like(distance_to_goal)
        
        # 큐브를 잡고 최소 높이 이상일 때만 목표 거리 보상 적용
        goal_reward_mask = self._cube_grabbed
        goal_distance_reward[goal_reward_mask] = (1.0 - torch.tanh(distance_to_goal[goal_reward_mask] / goal_std))
        goal_distance_reward = goal_distance_reward * self.cfg.distance_to_goal_reward_scale

        # 기본 보상 계산
        rewards = {
            "lin_vel": lin_vel * -50.0, # 선속도 패널티
            "ang_vel": ang_vel * -10.0, # 각속도 패널티
            "cube_gripper_distance": cube_gripper_reward * 15.0, # 큐브로의 접근
            "cube_lifted": lift_reward * 80, # 큐브 들어올리기
            "goal_distance": goal_distance_reward * 5000, # 큐브 든 상태로 목표 위치 접근
        }
        
        # 안전한 보상 합산 방식
        total_reward = torch.zeros_like(lin_vel)
        for value in rewards.values():
            total_reward += value
        
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
            else:
                self._episode_sums[key] = value

        # 보상 로깅
        self.log_reward_changes(rewards, total_reward)

        return total_reward * self.step_dt

    def log_reward_changes(self, rewards, total_reward=None):
        # 총 보상 계산
        if total_reward is None:
            reward_values = [value.mean().item() for value in rewards.values()]
            total_reward = sum(reward_values)
        else:
            if isinstance(total_reward, torch.Tensor):
                total_reward = total_reward.mean().item()
        
        # 절대값 합계 계산
        reward_abs_values = [abs(value.mean().item()) for value in rewards.values()]
        abs_sum = sum(reward_abs_values)
        
        # 0으로 나누기 방지
        eps = 1e-10
        safe_abs_sum = max(abs_sum, eps)
        
        # 초기 보상값 저장 또는 비교
        if not hasattr(self, "_initial_rewards") or self._initial_rewards is None:
            self._initial_rewards = {key: value.mean().item() for key, value in rewards.items()}
            initial_total = sum(self._initial_rewards.values())
            initial_abs_values = [abs(value) for value in self._initial_rewards.values()]
            initial_abs_sum = sum(initial_abs_values)
            safe_initial_abs_sum = max(initial_abs_sum, eps)
            
            print(f"초기 총 보상: {initial_total:.4f}")
            print("초기 보상값 저장:")
            for key, value in self._initial_rewards.items():
                percentage = (abs(value) / safe_initial_abs_sum) * 100
                sign_str = "" if value >= 0 else "-"
                print(f"초기 Reward {key}: {value:.4f} ({sign_str}{percentage:.2f}% 기여도)")
        else:
            print(f"\n총 보상: {total_reward:.4f}")
            print("현재 보상값 (기여도 %, 초기값 대비 변화량):")
            for key, value in rewards.items():
                current_value = value.mean().item()
                initial_value = self._initial_rewards.get(key, 0.0)
                
                # 기여도 계산
                abs_percentage = (abs(current_value) / safe_abs_sum) * 100
                sign_str = "" if current_value >= 0 else "-"
                
                # 변화율 계산
                if abs(initial_value) > eps:
                    percentage_change = ((current_value - initial_value) / abs(initial_value)) * 100
                    change_sign = "+" if percentage_change >= 0 else ""
                    print(f"Reward {key}: {current_value:.4f} ({sign_str}{abs_percentage:.2f}% 기여도, {change_sign}{percentage_change:.2f}% 변화)")
                else:
                    if abs(current_value) <= eps:
                        print(f"Reward {key}: {current_value:.4f} ({sign_str}{abs_percentage:.2f}% 기여도, 변화 없음)")
                    else:
                        direction = "증가" if current_value > 0 else "감소"
                        print(f"Reward {key}: {current_value:.4f} ({sign_str}{abs_percentage:.2f}% 기여도, 큰 폭 {direction})")

        # 에피소드 보상 데이터 기록 (기존 로깅 코드 뒤에 추가)
        if hasattr(self, "reward_history"):
            # 총 보상값 확정
            if total_reward is None:
                total_reward = sum([value.mean().item() for value in rewards.values()])
            elif isinstance(total_reward, torch.Tensor):
                total_reward = total_reward.mean().item()
            
            # 데이터 저장
            self.reward_history["episodes"].append(self.episode_counter)
            self.reward_history["total_reward"].append(total_reward)
            
            # 각 보상 요소 저장
            for key, value in rewards.items():
                if key in self.reward_history:
                    self.reward_history[key].append(value.mean().item())
            
            # 에피소드 카운터 증가
            self.episode_counter += 1                        

    def plot_reward_graph(self, save=True, show=False, filename=None):
        """보상 그래프를 생성하고 저장/표시합니다.
        
        Args:
            save (bool): 그래프 저장 여부
            show (bool): 그래프 표시 여부
            filename (str): 지정된 파일명 (None이면 자동 생성)
        """
        if len(self.reward_history["episodes"]) == 0:
            print("그래프를 그릴 보상 데이터가 없습니다.")
            return
            
        try:
            # 그래프 생성 - 행 수를 3에서 4로 늘려서 새 그래프 추가
            plt.figure(figsize=(20, 20))
            
            # 개별 보상 그래프
            reward_keys = ["lin_vel", "ang_vel", "cube_gripper_distance", "cube_lifted", "goal_distance"]
            
            # 1. 개별 보상 그래프
            for i, key in enumerate(reward_keys):
                if key in self.reward_history and len(self.reward_history[key]) > 0:
                    plt.subplot(4, 2, i+1)
                    plt.plot(self.reward_history["episodes"], self.reward_history[key], 'b-')
                    plt.title(f'Reward: {key}')
                    plt.xlabel('Episodes')
                    plt.ylabel('Reward Value')
                    plt.grid(True)
            
            # 2. 총 보상 그래프 (goal_distance를 포함하지 않도록 명시적 수정)
            plt.subplot(4, 2, 6)
            plt.plot(self.reward_history["episodes"], self.reward_history["total_reward"], 'r-')
            plt.title('Total Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.grid(True)
            
            # 3. 모든 보상을 한 그래프에 표시 (절대값 기준)
            plt.subplot(4, 2, 7)
            colors = ['b', 'g', 'r', 'c', 'm']
            for i, key in enumerate(reward_keys):
                if key in self.reward_history and len(self.reward_history[key]) > 0:
                    # 절대 스케일이 매우 다른 경우 선의 두께와 스타일로 구분
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
            
            # 4. NEW: 백분율(%) 기준 보상 기여도 그래프 추가
            plt.subplot(4, 2, 8)
            
            # 각 에피소드별 보상의 절대값 합계 계산
            total_abs_rewards = []
            for ep_idx in range(len(self.reward_history["episodes"])):
                ep_abs_sum = 0
                for key in reward_keys:
                    if key in self.reward_history and ep_idx < len(self.reward_history[key]):
                        ep_abs_sum += abs(self.reward_history[key][ep_idx])
                total_abs_rewards.append(max(ep_abs_sum, 1e-10))  # 0으로 나누기 방지
                
            # 백분율 데이터 생성 및 그래프 그리기
            for i, key in enumerate(reward_keys):
                if key in self.reward_history and len(self.reward_history[key]) > 0:
                    # 백분율 계산
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
            plt.ylim(0, 100)  # y축 범위 0~100%로 설정
            plt.grid(True)
            plt.legend(loc='best')
            
            plt.tight_layout()
            
            # 그래프 저장
            if save:
                # 저장 디렉토리 생성
                save_dir = os.path.expanduser("~/reward_graphs")
                os.makedirs(save_dir, exist_ok=True)
                
                # 파일명에 범위 표시
                if filename is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    start_ep = 0
                    end_ep = self.episode_counter
                    filename = f"reward_graph_{timestamp}_ep{start_ep}-{end_ep}.png"
                    
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath)
                print(f"보상 그래프가 저장되었습니다: {filepath}")
            
            # 그래프 표시
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"그래프 생성 중 오류가 발생했습니다: {e}")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 기존 종료 조건
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        
        # 목표 도달 성공 종료 조건 추가
        robot_pos = self._robot.data.root_pos_w
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - robot_pos, dim=1)
        
        # 성공 조건: 큐브가 들려있고, 목표 지점에 가까움
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

        # 추가: 큐브 관련 최종 통계
        if env_ids is not None and len(env_ids) > 0:
            cube_pos = self._cube.data.root_state_w[env_ids, :3]
            final_cube_height = (cube_pos[:, 2] - 0.05).mean()  # 테이블에서의 높이
            final_cube_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - cube_pos, dim=1).mean()
            
            # 추가 로깅
            if "Metrics" not in self.extras["log"]:
                self.extras["log"]["Metrics"] = {}
            self.extras["log"]["Metrics"]["final_cube_height"] = final_cube_height.item()
            self.extras["log"]["Metrics"]["final_cube_to_goal"] = final_cube_to_goal.item()        

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
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

        # 테이블 초기화
        table_state = torch.zeros((len(env_ids), 13), device=self.device)
        table_state[:, :3] = self._terrain.env_origins[env_ids]
        table_state[:, 2] += 0.05  # 테이블 높이 중심
        table_state[:, 3] = 1.0    # 쿼터니언 w
        self._table.write_root_state_to_sim(table_state, env_ids)
        
        # 큐브 초기화
        cube_state = torch.zeros((len(env_ids), 13), device=self.device)
        cube_state[:, :3] = self._terrain.env_origins[env_ids]
        cube_state[:, 2] += 0.075  # 테이블 위 큐브 높이
        cube_state[:, 3] = 1.0     # 쿼터니언 w
        self._cube.write_root_state_to_sim(cube_state, env_ids)

        # 그리퍼 상태 초기화
        self._keep_gripper_closed[env_ids] = False
        
        # 그래프 생성 조건 판단
        should_create_graph = False

        if self.episode_counter <= 1000:
            # 1000 에피소드 이전: 100 단위로 그래프 생성 (100, 200, ..., 900, 1000)
            should_create_graph = self.episode_counter % 100 == 0
            interval_text = "100"
        elif self.episode_counter <= 10000:
            # 1000~10000 에피소드: 1000 단위로 그래프 생성 (2000, 3000, ..., 9000, 10000)
            should_create_graph = self.episode_counter % 1000 == 0
            interval_text = "1000"
        else:
            # 10000 에피소드 이후
            # 정확한 10000 배수 확인
            is_exact_multiple = self.episode_counter % 10000 == 0
            
            # 이전 10000 배수 및 다음 10000 배수 계산
            prev_milestone = (self.episode_counter // 10000) * 10000
            next_milestone = prev_milestone + 10000
            
            # 1. 정확히 10000의 배수이거나 
            # 2. 아직 그래프가 생성되지 않았고 현재 에피소드가 다음 10000 배수를 지났을 경우
            if is_exact_multiple:
                should_create_graph = True
                milestone_ep = self.episode_counter
            elif hasattr(self, "last_graph_episode"):
                # 마지막 그래프가 이전 10000 배수에서 생성되었는지 확인
                if self.last_graph_episode < prev_milestone and self.episode_counter > prev_milestone:
                    should_create_graph = True
                    milestone_ep = prev_milestone
                    print(f"지난 마일스톤({prev_milestone})에서 그래프 생성 누락. 지금 생성합니다.")
                
            interval_text = "10000"

        # 그래프 생성
        if should_create_graph:
            # 이전 그래프 생성 시점을 저장하는 변수
            if not hasattr(self, "last_graph_episode") or self.last_graph_episode is None:
                self.last_graph_episode = 0
            
            # 마지막 생성 시점보다 에피소드가 진행된 경우에만 그래프 생성
            if self.episode_counter > self.last_graph_episode:  # 최소 100 에피소드 간격
                timestamp = datetime.datetime.now().strftime("%Y%m%d")
                current_ep = self.episode_counter
                
                # 파일명에 간격 정보 추가
                filename = f"reward_graph_{timestamp}_ep{current_ep}_interval{interval_text}.png"
                
                # 그래프 생성
                try:
                    self.plot_reward_graph(save=True, show=False, filename=filename)
                    print(f"에피소드 {current_ep}에서 그래프가 성공적으로 생성되었습니다. (간격: {interval_text})")
                    
                    # 마지막 그래프 생성 에피소드 업데이트
                    self.last_graph_episode = current_ep
                except Exception as e:
                    print(f"그래프 생성 중 오류 발생: {e}")

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)