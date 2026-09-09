"""
Probe the MuJoCo PickLift environment for everything a scripted
controller needs to know.

Run:  python scripts\probe_env.py

Paste the whole output. Nothing here changes any files.
"""

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import so101_nexus.mujoco

from so101_nexus.config import PickConfig
from so101_nexus.observations import (
    JointPositions, JointVelocities, EndEffectorPose,
    GraspState, ObjectPose, ObjectOffset,
    WristCamera, OverheadCamera,
)

# Exactly the config we established today: 30-wide privileged state, no GazeState.
CFG = PickConfig(
    obs_mode="visual",
    observations=[
        JointPositions(),      # priv  0:6
        JointVelocities(),     # priv  6:12
        EndEffectorPose(),     # priv 12:19  (xyz + quaternion)
        GraspState(),          # priv 19
        ObjectPose(),          # priv 20:27  (xyz + quaternion)
        ObjectOffset(),        # priv 27:30
        WristCamera(),
        OverheadCamera(),
    ],
)


def main():
    env = gym.make("MuJoCoPickLift-v1", config=CFG, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    u = env.unwrapped

    print("=" * 60)
    print("1. ACTION SPACE")
    print("=" * 60)
    print("  shape:", env.action_space.shape)
    print("  low  :", np.round(env.action_space.low, 4))
    print("  high :", np.round(env.action_space.high, 4))
    print("  dtype:", env.action_space.dtype)

    print("\n" + "=" * 60)
    print("2. PRIVILEGED STATE LAYOUT")
    print("=" * 60)
    priv = np.asarray(u._privileged_state, dtype=np.float64)
    print("  total length:", priv.shape)
    print("  [ 0: 6] joint positions :", np.round(priv[0:6], 4))
    print("  [ 6:12] joint velocities:", np.round(priv[6:12], 4))
    print("  [12:19] ee pose         :", np.round(priv[12:19], 4))
    print("  [19]    grasp state     :", np.round(priv[19], 4))
    print("  [20:27] object pose     :", np.round(priv[20:27], 4))
    print("  [27:30] object offset   :", np.round(priv[27:30], 4))

    print("\n" + "=" * 60)
    print("3. MUJOCO MODEL NAMES")
    print("=" * 60)
    m = u.model

    print("  -- joints --")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    {i}: {name}")

    print("  -- sites --")
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"    {i}: {name}")

    print("  -- bodies --")
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"    {i}: {name}")

    print("  -- actuators --")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"    {i}: {name}")

    print(f"\n  nq={m.nq}  nv={m.nv}  nu={m.nu}")

    print("\n" + "=" * 60)
    print("4. JOINT LIMITS")
    print("=" * 60)
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        lim = m.jnt_range[i]
        limited = bool(m.jnt_limited[i])
        print(f"    {name:24s} limited={limited}  range={np.round(lim, 4)}")

    print("\n" + "=" * 60)
    print("5. WHAT ONE STEP DOES")
    print("=" * 60)
    qpos_before = u._get_current_qpos().copy()
    print("  qpos before  :", np.round(qpos_before, 4))

    # Command a small change on joint 0 only.
    action = qpos_before.copy().astype(np.float32)
    action[0] += 0.2
    action = np.clip(action, env.action_space.low, env.action_space.high)
    print("  action sent  :", np.round(action, 4))

    for _ in range(20):
        obs, reward, term, trunc, info = env.step(action)

    qpos_after = u._get_current_qpos().copy()
    print("  qpos after   :", np.round(qpos_after, 4))
    print("  delta        :", np.round(qpos_after - qpos_before, 4))
    print("\n  -> if joint 0 moved ~+0.2, actions are absolute joint targets in radians")
    print("  -> if nothing moved, actions are normalised or deltas")

    print("\n" + "=" * 60)
    print("6. INFO / REWARD")
    print("=" * 60)
    print("  reward:", reward)
    for k, v in info.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("7. OBJECT POSITION ACROSS SEEDS")
    print("=" * 60)
    for seed in range(5):
        o, i2 = env.reset(seed=seed)
        p = np.asarray(u._privileged_state, dtype=np.float64)
        print(f"    seed {seed}: object xyz = {np.round(p[20:23], 4)}  "
              f"ee xyz = {np.round(p[12:15], 4)}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()