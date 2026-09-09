import numpy as np, gymnasium as gym, mujoco.viewer, so101_nexus.mujoco, imageio
from so101_nexus.config import PickConfig
from so101_nexus.observations import *
cfg = PickConfig(obs_mode="visual", observations=[JointPositions(), JointVelocities(), EndEffectorPose(), GraspState(), ObjectPose(), ObjectOffset(), WristCamera(), OverheadCamera()])
env = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array")
for label, val in [("gmin", -0.1745), ("gmax", 1.7453)]:
    obs, info = env.reset(seed=0)
    q = np.asarray(info["privileged_state"])[0:6].astype(np.float32)
    a = q.copy(); a[5] = val
    a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
    for _ in range(80):
        obs, r, t, tr, info = env.step(a)
    imageio.imwrite(f"gripper_{label}.png", obs["wrist_camera"])
    print("saved", label)
env.close()
