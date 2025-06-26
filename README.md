# quadcopter_gripper

## Installation
1. IsaacLab install
2. Copy from source\isaaclab_assets\isaaclab_assets\robots\quadcopter.py and Paste within your IsaacLab's folder to IsaacLab\source\isaaclab\assets\robots\
3. Copy from source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\quadcopter_env.py and Paste within your IsaacLab's folder to IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\

## Training
```
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Quadcopter-Direct-v0 --num_envs 128 --max_iterations 6300
```

## Playing
```
python scripts/reinforcement_learning/skrl/play.py --task Isaac-Quadcopter-Direct-v0 --checkpoint /home/dmsai3/IsaacLab/logs/skrl/quadcopter_direct/2025-06-25_19-05-18_ppo_torch_29s_100000/checkpoints/best_agent.pt --num_envs 1
```
