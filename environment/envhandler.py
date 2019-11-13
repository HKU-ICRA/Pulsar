import numpy as np
import time

import glfw
from mujoco_py import const, MjViewer
from gym.spaces import Box, MultiDiscrete, Discrete


class EnvHandler():

    def __init__(self, env):
        self.env = env
        self.action_types = list(self.env.action_space.spaces.keys())
        self.num_action_types = len(self.env.action_space.spaces)
        self.num_action = self.num_actions(self.env.action_space)
        self.agent_mod_index = 0
        self.action_mod_index = 0
        self.action_type_mod_index = 0
        self.action = self.zero_action(self.env.action_space)

    def num_actions(self, ac_space):
        n_actions = []
        for k, tuple_space in ac_space.spaces.items():
            s = tuple_space.spaces[0]
            if isinstance(s, Box):
                n_actions.append(s.shape[0])
            elif isinstance(s, Discrete):
                n_actions.append(1)
            elif isinstance(s, MultiDiscrete):
                n_actions.append(s.nvec.shape[0])
            else:
                raise NotImplementedError(f"not NotImplementedError")

        return n_actions

    def zero_action(self, ac_space):
        ac = {}
        for k, space in ac_space.spaces.items():
            if isinstance(space.spaces[0], Box):
                ac[k] = np.zeros_like(space.sample())
            elif isinstance(space.spaces[0], Discrete):
                ac[k] = np.ones_like(space.sample()) * (space.spaces[0].n // 2)
            elif isinstance(space.spaces[0], MultiDiscrete):
                ac[k] = np.ones_like(space.sample(), dtype=int) * (space.spaces[0].nvec // 2)
            else:
                raise NotImplementedError("MultiDiscrete not NotImplementedError")
                # return action_space.nvec // 2  # assume middle element is "no action" action
        return ac

    def reset(self):
        obs = self.env.reset()
        return obs

    def render(self, mode):
        self.env.render(mode=mode)
    
    def close(self):
        self.env.close()

    def step(self, action):
        self.action = action
        obs, rew, done, info = self.env.step(self.action)
        return obs, rew, done, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec

    @property
    def n_actors(self):
        return self.env.metadata['n_actors']
