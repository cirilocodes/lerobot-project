import time
import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco
import numpy as np

env = gym.make("MuJoCoTouch-v1", render_mode="human")
obs, info = env.reset(seed=0)

n = env.action_space.shape[0]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

print("Sweeping each joint in turn. Ctrl+C to stop.")

try:
    for step in range(1800):
        action = np.zeros(n, dtype=np.float32)
        joint = (step // 100) % n
        phase = (step % 100) / 100.0
        action[joint] = 0.6 * np.sin(phase * 2 * np.pi)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)

        if step % 100 == 0:
            name = JOINT_NAMES[joint] if joint < len(JOINT_NAMES) else f"joint {joint}"
            print(f"  step {step:5d}  moving: {name}")

        if terminated or truncated:
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nStopped.")

env.close()