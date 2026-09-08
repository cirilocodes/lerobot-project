import gymnasium as gym
import so101_nexus.mujoco  # this import is what registers the env IDs

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)

print("Environment loaded.")
print("Render modes available:", env.metadata.get("render_modes"))
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

frame = env.render()
print("Frame shape:", None if frame is None else frame.shape)

env.close()