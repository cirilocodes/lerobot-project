import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco
import imageio

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)

frame = env.render()
imageio.imwrite("first_frame.png", frame)
print("Saved first_frame.png")

env.close()