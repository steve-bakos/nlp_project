import gym

# Number of layers for the XLM-R Base model
action_space = gym.spaces.MultiBinary(12)

# Placeholder for now
observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)