"""
Scripted controller for MuJoCoPickLift-v1.

Solves the task with inverse kinematics and a phase state machine.
No learning involved - this GENERATES the demonstrations that a
policy will later learn from.

Usage:
    python scripts\\scripted_pick.py --test-gripper
        Work out which gripper value opens and which closes.
        RUN THIS FIRST.

    python scripts\\scripted_pick.py --demo --seed 0
        One episode in the 3D viewer so you can watch it.

    python scripts\\scripted_pick.py --batch --episodes 50
        Run many seeds headless, report success rate.

How it works
------------
The simulator tells us exactly where the cube is (privileged info a
real robot could never have). We convert "put the gripper HERE" into
joint angles using the Jacobian: it maps joint velocities to end
effector velocity, so inverting it maps a desired EE motion back to
the joint motion that achieves it. Damped least squares keeps that
inversion stable near singularities.

Four phases: hover above the cube, descend onto it, close, lift.
"""

import argparse
import time
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

# ---------------------------------------------------------------- config

CFG = PickConfig(
    obs_mode="visual",
    terminate_on_success=False,
    observations=[
        JointPositions(), JointVelocities(), EndEffectorPose(),
        GraspState(), ObjectPose(), ObjectOffset(),
        WristCamera(), OverheadCamera(),
    ],
)

EE_SITE = "gripperframe"
N_ARM = 5          # first five joints move the arm; the sixth is the gripper

# Privileged state slices, confirmed by probe_env.py
SL_QPOS = slice(0, 6)
SL_EE_XYZ = slice(12, 15)
SL_GRASP = 19
SL_OBJ_XYZ = slice(20, 23)

# Gripper values - VERIFY WITH --test-gripper AND SWAP IF NEEDED
GRIPPER_OPEN = 1.7
GRIPPER_CLOSE = 0.05

# Phase geometry (metres)
HOVER_HEIGHT = 0.10        # above the cube before descending
GRASP_HEIGHT = 0.002       # final height above cube centre
LIFT_HEIGHT = 0.18         # target height after grasping

PLACE_ROTATE_DEG = 35.0    # swing this far around the base to the drop spot
RELEASE_HEIGHT = 0.02      # height above table when opening the jaws
RETREAT_HEIGHT = 0.20      # how high to back off afterwards

# Control
IK_DAMPING = 0.08          # larger = more stable, slower
IK_ITERS = 6               # IK refinement steps per control step
MAX_STEP = 0.02
MAX_STEP_CARRY = 0.02     # slower once holding the cube            # max joint change per control step (rad)
POS_TOL = 0.006            # "arrived" tolerance (m)

PHASE_TIMEOUT = 500        # control steps before giving up on a phase

# ---------------------------------------------------------------- ik
def cube_yaw(quat_wxyz):
    """Yaw angle of the cube about the world z axis."""
    w, x, y, z = quat_wxyz
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap90(a):
    """A cube looks the same every 90 degrees - fold into [-45, +45]."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4

def ik_delta(model, data, qpos_arm, target_xyz, site_id):
    """Joint deltas that move the EE site toward target_xyz."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0
    q = np.array(qpos_arm, dtype=np.float64)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(IK_ITERS):
        scratch.qpos[:N_ARM] = q[:N_ARM]
        mujoco.mj_forward(model, scratch)

        cur = scratch.site_xpos[site_id].copy()
        err = target_xyz - cur
        if np.linalg.norm(err) < 1e-4:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = jacp[:, :N_ARM]                      # 3 x N_ARM

        # damped least squares: dq = J^T (J J^T + λ²I)^-1 err
        JJt = J @ J.T + (IK_DAMPING ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)

        q[:N_ARM] += np.clip(dq, -0.15, 0.15)

    return q[:N_ARM]


def clamp_action(action, env):
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


# ---------------------------------------------------------------- controller


def run_episode(env, seed, render=False, collect=False):
    """One scripted attempt. Returns (success, frames_or_None, transitions)."""
    obs, info = env.reset(seed=seed)
    u = env.unwrapped
    model, data = u.model, u.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    priv = np.asarray(info["privileged_state"], dtype=np.float64)
    obj_xyz = priv[SL_OBJ_XYZ].copy()
    obj_quat = priv[23:27].copy()
    obj_yaw = wrap90(cube_yaw(obj_quat))

    # Drop spot: swing around the base so it stays the same distance out
    ang = np.deg2rad(PLACE_ROTATE_DEG)
    ca, sa = np.cos(ang), np.sin(ang)
    place_xy = np.array([ca * obj_xyz[0] - sa * obj_xyz[1],
                         sa * obj_xyz[0] + ca * obj_xyz[1]])
    place = np.array([place_xy[0], place_xy[1], obj_xyz[2]])

    targets = {
        "HOVER":   obj_xyz + np.array([0.0, 0.0, HOVER_HEIGHT]),
        "DESCEND": obj_xyz + np.array([0.0, 0.0, GRASP_HEIGHT]),
        "CLOSE":   obj_xyz + np.array([0.0, 0.0, GRASP_HEIGHT]),
        "LIFT":    obj_xyz + np.array([0.0, 0.0, LIFT_HEIGHT]),
        "MOVE":    place  + np.array([0.0, 0.0, LIFT_HEIGHT]),
        "LOWER":   place  + np.array([0.0, 0.0, RELEASE_HEIGHT]),
        "RELEASE": place  + np.array([0.0, 0.0, RELEASE_HEIGHT]),
        "RETREAT": np.array([place[0] * 0.55, place[1] * 0.55,
                             place[2] + RETREAT_HEIGHT]),
        "SETTLE":  np.array([place[0] * 0.55, place[1] * 0.55,
                             place[2] + RETREAT_HEIGHT]),
    }
    order = ["HOVER", "DESCEND", "CLOSE", "LIFT",
             "MOVE", "LOWER", "RELEASE", "RETREAT", "SETTLE"]
    release_hold = 0
    settle_hold = 0

    phase_i = 0
    phase_steps = 0
    close_hold = 0
    frames = []
    transitions = []
    success = False
    lifted = False

    for step in range(3000):
        phase = order[phase_i]
        priv = np.asarray(info["privileged_state"], dtype=np.float64)
        q_now = priv[SL_QPOS].copy()
        ee = priv[SL_EE_XYZ].copy()

        target = targets[phase]
        q_arm = ik_delta(model, data, q_now, target, site_id)

        # limit how far the joints move in one control step
        step_cap = MAX_STEP_CARRY if phase in ("LIFT", "MOVE", "LOWER", "RELEASE") else MAX_STEP
        delta = np.clip(q_arm - q_now[:N_ARM], -step_cap, step_cap)
        action = np.empty(6, dtype=np.float32)
        action[:N_ARM] = q_now[:N_ARM] + delta
        action[5] = GRIPPER_CLOSE if phase in ("CLOSE", "LIFT", "MOVE", "LOWER") else GRIPPER_OPEN
        action[4] = wrap90(obj_yaw - action[0])
        reach = np.linalg.norm(target[:2])          # use the live target, not the cube
        t = np.clip((reach - 0.20) / 0.18, 0.0, 1.0)
        action[3] = np.clip(action[3] + 0.6 * (1.0 - t), env.action_space.low[3], env.action_space.high[3])
        action = clamp_action(action, env)

        prev_obs = obs
        obs, reward, term, trunc, info = env.step(action)

        if collect:
            transitions.append({
                "state": np.asarray(prev_obs["state"], dtype=np.float32).copy(),
                "wrist": prev_obs["wrist_camera"].copy(),
                "overhead": prev_obs["overhead_camera"].copy(),
                "env_state": np.asarray(info["privileged_state"], dtype=np.float32).copy(),
                "action": action.copy(),
            })

        if render:
            env.render()
            time.sleep(0.005)

        if info.get("success"):
            lifted = True          # note it, but keep going to place the cube

        dist = np.linalg.norm(ee - target)
        phase_steps += 1

        if phase == "CLOSE":
            close_hold += 1
            if close_hold > 60:
                phase_i += 1; phase_steps = 0
        elif phase == "RELEASE":
            release_hold += 1
            if release_hold > 50:
                phase_i += 1; phase_steps = 0
        elif phase == "SETTLE":
            settle_hold += 1
            if settle_hold > 150:
                break
        elif dist < POS_TOL or phase_steps > PHASE_TIMEOUT:
            if phase_i < len(order) - 1:
                phase_i += 1; phase_steps = 0
            else:
                break

        if term or trunc:
            success = bool(info.get("success", False))
            break

    priv = np.asarray(info["privileged_state"], dtype=np.float64)
    final_obj = priv[SL_OBJ_XYZ]
    ee_final = priv[SL_EE_XYZ]

    on_table = final_obj[2] < 0.04
    moved = np.linalg.norm(final_obj[:2] - obj_xyz[:2]) > 0.05
    clear = np.linalg.norm(ee_final - final_obj) > 0.12
    success = bool(lifted and on_table and moved and clear)

    print(f"    end: phase={order[phase_i]} step={step} lifted={lifted} "
          f"objz={final_obj[2]:.4f} "
          f"moved={np.linalg.norm(final_obj[:2]-obj_xyz[:2]):.3f} "
          f"clear={np.linalg.norm(ee_final-final_obj):.3f} success={success}")

    return success, frames, transitions


# ---------------------------------------------------------------- modes


def test_gripper():
    env = gym.make("MuJoCoPickLift-v1", config=CFG, render_mode="rgb_array", max_episode_steps=4000)
    print("Testing which gripper value opens vs closes.\n")

    for label, val in [("min (-0.1745)", -0.1745), ("max (1.7453)", 1.7453)]:
        obs, info = env.reset(seed=0)
        u = env.unwrapped
        priv = np.asarray(info["privileged_state"])
        q = priv[SL_QPOS].copy()
        action = q.astype(np.float32)
        action[5] = val
        action = clamp_action(action, env)
        for _ in range(60):
            obs, r, t, tr, info = env.step(action)
        jaw = u.data.qpos[5]
        print(f"  gripper cmd {label:16s} -> jaw qpos {jaw:+.4f}")

    print("\n  The value giving the LARGER jaw qpos is OPEN.")
    print("  Set GRIPPER_OPEN / GRIPPER_CLOSE at the top of this file accordingly.")
    env.close()


def demo(seed):
    env = gym.make("MuJoCoPickLift-v1", config=CFG, render_mode="human", max_episode_steps=4000)
    ok, _, _ = run_episode(env, seed, render=True)
    print(f"\n  seed {seed}: success = {ok}")
    env.close()


def batch(episodes):
    env = gym.make("MuJoCoPickLift-v1", config=CFG, render_mode="rgb_array", max_episode_steps=4000)
    wins = 0
    for s in range(episodes):
        ok, _, _ = run_episode(env, s)
        wins += int(ok)
        print(f"  seed {s:3d}: {'OK ' if ok else 'FAIL'}   running {wins}/{s+1}")
    print(f"\n=== {wins}/{episodes} = {100*wins/episodes:.1f}% success ===")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-gripper", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    a = p.parse_args()

    if a.test_gripper:
        test_gripper()
    elif a.demo:
        demo(a.seed)
    elif a.batch:
        batch(a.episodes)
    else:
        print("Pick one: --test-gripper | --demo | --batch")