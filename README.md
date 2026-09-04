# Isaac Lab(Gym) Quadcopter Object Manipulation

### A deep reinforcement learning framework for training quadcopters to perform object manipulation tasks in Isaac Sim.  
<img src="media/grisp_lift_f.gif" width="600"/>  
<img src="media/128env_training_f2.gif" width="1200"/>  
<img src="media/grisp_lift_togoal.gif" width="1200"/>  

## Publication  
This work has been published in the Proceedings of APISAT 2025 (Springer, Lecture Notes in Mechanical Engineering):

> Ui-Seok Lee, Hyun-Seok Kang, Ye-Chan Park, Tuan Anh Nguyen, Dug-Ki Min, Eun-Mi Choi, Jae-Woo Lee.
> **Sequential Task Reward Design for Complex Quadcopter Manipulation Using Massively Parallel Simulation.**
> In: *Proceedings of The 2025 Asia-Pacific International Symposium on Aerospace Technology Vol 6*, Lecture Notes in Mechanical Engineering, pp. 283–293. Springer, Singapore.
> https://doi.org/10.1007/978-981-92-1319-1_21

If you use this code in your research, please cite:

```bibtex
@inproceedings{lee2025sequential,
  title     = {Sequential Task Reward Design for Complex Quadcopter Manipulation Using Massively Parallel Simulation},
  author    = {Lee, Ui-Seok and Kang, Hyun-Seok and Park, Ye-Chan and Nguyen, Tuan Anh and Min, Dug-Ki and Choi, Eun-Mi and Lee, Jae-Woo},
  booktitle = {Proceedings of The 2025 Asia-Pacific International Symposium on Aerospace Technology Vol 6},
  series    = {Lecture Notes in Mechanical Engineering},
  pages     = {283--293},
  publisher = {Springer, Singapore},
  doi       = {10.1007/978-981-92-1319-1_21}
}
```

## Introduction  
**IsaacLab Quadcopter Manipulation** is a reinforcement learning framework designed to train drone-based robots to manipulate objects. The quadcopters are trained to identify, approach, grasp, and transport objects to designated locations, performing complex sequential tasks.

Built on IsaacSim, this environment provides realistic physics-based drone-object interactions and supports over 128 parallel environments for scalable distributed training.

## Key Features  
- **Multi-Stage Task Learning**: Trains drones in sequential steps—approaching, grasping, transporting, and placing objects  
- **High-Fidelity Physics Simulation**: Built on IsaacSim for precision physical interactions  
- **Multi-Reward System**: Fine-grained reward signals for velocity control, object approach, grasping, and task completion  
- **Detailed Analytics Tools**: Built-in visualization for tracking and analyzing reward contributions  
- **Massive Parallel Training**: Run hundreds of environments simultaneously to accelerate training  
- **Gripper Control Mechanism**: Precision control for stable grasping and object handling  

## Applications  
This framework can be used for:

- **Object Relocation Tasks**: Moving objects across surfaces like tables  
- **Drone Precision Control**: Stable hovering and flight in various conditions  
- **Drone-Object Interaction**: Performing airborne grasping and manipulation tasks   

## Environment Configuration  
- **Observation Space**: Includes robot states (linear/angular velocity, gravity vector), target position, gripper-object relation  
- **Action Space**: Thrust control, moment control, and gripper joint control (7 dimensions)  
- **Reward Components**:  
  - Velocity penalties to encourage stable flight  
  - Gripper-object distance for accurate targeting  
  - Object lift reward to promote successful grasp  
  - Target distance reward for task completion  

## Reward Design and Analysis  
For complex manipulation tasks, effective reward design is critical. The relative ratio between reward components is more important than their absolute values, as a balanced structure is key to guiding the agent's learning process.  

Our design philosophy follows two core principles:  

- **Sequential Guidance**: The reward system guides the agent through a sequence of sub-tasks (e.g., approach -> grisp&lift -> move to goal). Rewards for later stages become more dominant as the agent gains proficiency in earlier ones.  
- **Conditional Gating**: Rewards for later sub-tasks are gated by the successful completion of earlier ones. For example, the goal_distance reward is only active after the cube_lifted condition is true, enforcing the correct sequence.  

### Reward Formulation  
The overall reward function, designed according to this philosophy, is composed of the following components (Eq. 6):

$$
r_t = w_{vel} \cdot r_{vel} + w_{dist} \cdot r_{dist} + w_{lift} \cdot r_{lift} + \mathbb{I}_{\mathrm{lifted}} \cdot w_{goal} \cdot r_{goal} \quad (6)
$$

with the scalar weights ordered as

$$
w_{vel} < w_{dist} < w_{lift} < w_{goal} \quad (7)
$$

Each reward element is as follows:

| Term | Code name | Description |
|------|-----------|-------------|
| $r_{vel}$ | `lin_vel`, `ang_vel` | A speed penalty designed to encourage stable flight. It suppresses excessive linear and angular velocities to maintain flight stability. |
| $r_{dist}$ | `cube_gripper_distance` | A reward for minimizing the distance between the gripper and the target object. This encourages precise approach behavior. |
| $r_{lift}$ | `cube_lifted` | A sparse bonus triggered upon successfully lifting the object. It serves as a critical signal for successful grasp execution. |
| $r_{goal}$ | `goal_distance` | A reward for reducing the distance between the lifted object and the target location. This reward is only activated when the lift condition is satisfied. |
| $w_{vel},\ w_{dist},\ w_{lift},\ w_{goal}$ | reward scales in `quadcopter_env.py` | Scalar weights controlling reward importance. They are set such that $w_{vel} < w_{dist} < w_{lift} < w_{goal}$ (Eq. 7), so that larger weights encourage transition to the next behavior. This promotes sequential task completion from approach → grasp and lift (`lift`) → move to target (`goal`). |
| $\mathbb{I}_{\mathrm{lifted}}$ | `cube_lifted` condition | Indicator function (1 if the object has been lifted, 0 otherwise), used to activate the goal reward only after the grasp-and-lift task is completed. |

This sequential task reward design alleviates the difficulty of the sparse reward problem and helps the agent systematically master each step of a complex task.

### Reward Contribution Over Time  
The following graphs show how the reward composition shifts, validating our design.  

1. **Episode 0-10: Learning to Approach** / **key reward:** `cube_gripper_distance`
  
  <img src="media/100_episode_ff.png" width="600"/><br>
  
2. **Episode 150-300: Mastering the Lift** / **key reward:** `cube_lifted`
  
  <img src="media/500_episode_ff.png" width="600"/><br>
  
3. **Episode 400 and beyond: Focusing on the Goal** / **key reward:** `goal_distance`
  
  <img src="media/1000_episode_ff.png" width="600"/><br>
  <img src="media/10000_episode_ff.png" width="600"/><br>

## Customization  
The framework allows various customizations:

- Adjust reward components and weights  
- Modify observation space to tune task difficulty  
- Change environment setup (table/object/target positions)  
- Configure robot properties (thrust, gripper behavior)  

## Requirements  
- IsaacLab(v2.0) / IsaacSim(v4.5)  
- PyTorch  
- CUDA-enabled GPU (RTX 3090 GPU recommended)  
- Python 3.8+

## Installation
1. Copy from
  ```
  source\isaaclab_assets\isaaclab_assets\robots\quadcopter.py
  ```
2. Paste within your IsaacLab's folder to
  ```
  IsaacLab\source\isaaclab\assets\robots\
  ```
3. Copy from
  ```
  source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\quadcopter_env.py
  ```
4. Paste within your IsaacLab's folder to
  ```
  IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\
  ```
5. Copy from
  ```
  source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\agents\skrl_ppo_cfg.yaml
  ```
6. Paste within your IsaacLab's folder to
  ```
  IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\agents\
  ```

## Training
```
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Quadcopter-Direct-v0 --num_envs 128 --max_iterations 1000
```

## Playing
```
python scripts/reinforcement_learning/skrl/play.py --task Isaac-Quadcopter-Direct-v0 --checkpoint /home/dmsai3/IsaacLab/logs/skrl/quadcopter_direct/2025-07-04_15-55-35_ppo_torch/checkpoints/best_agent.pt --num_envs 1
```
