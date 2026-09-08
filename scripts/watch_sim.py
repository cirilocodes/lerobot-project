import time
import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco

env = gym.make("MuJoCoPickLift-v1", render_mode="human")
obs, info = env.reset(seed=0)

print("Viewer open. Ctrl+C in this terminal to stop.")

try:
    for step in range(5000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.01)
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    print("\nStopped.")

env.close()