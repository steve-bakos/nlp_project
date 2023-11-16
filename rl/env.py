import gym
from spaces import action_space, observation_space
from reward import reward_function

import json
# Load configuration from JSON file
with open('rl/config.json', 'r') as config_file:
    config = json.load(config_file)

import importlib.util
def import_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

controlled_realignment = import_from_path("scripts/2023_acl/controlled_realignment.py")
controlled_realignment.rl_main(config)

class FreezingEnv(gym.Env):
    def __init__(self):
        super(FreezingEnv, self).__init__()

        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_function = reward_function

    def step(self, action):
        pass

    def reset(self):
        pass