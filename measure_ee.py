import numpy as np, gymnasium as gym, mujoco, mujoco.viewer, so101_nexus.mujoco
from so101_nexus.config import PickConfig
from so101_nexus.observations import *
cfg = PickConfig(obs_mode="visual", observations=[JointPositions(), JointVelocities(), EndEffectorPose(), GraspState(), ObjectPose(), ObjectOffset(), WristCamera(), OverheadCamera()])
e = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array")
o, i = e.reset(seed=0)
u = e.unwrapped; m, d = u.model, u.data
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
print("site gripperframe xpos:", np.round(d.site_xpos[sid], 4))
for name in ["gripper", "moving_jaw_so101_v1", "wrist"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    print(f"body {name:22s} xpos: {np.round(d.xpos[bid],4)}")
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if bn in ("gripper", "moving_jaw_so101_v1"):
        print(f"  geom {g} on {bn}: pos {np.round(d.geom_xpos[g],4)}")
e.close()
