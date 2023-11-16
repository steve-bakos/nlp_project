from stable_baselines3 import PPO
from env import FreezingEnv

# Create and wrap the environment
env = FreezingEnv()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# # Train the agent
# model.learn(total_timesteps=10000)

# Save the model
# model.save("ppo_custom_env_model")