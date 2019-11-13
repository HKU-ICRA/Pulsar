import sys
import numpy as np
from copy import deepcopy
from itertools import compress

import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple

from mujoco_worldgen.util.rotation import mat2quat
from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs, joint_qpos_idxs, body_names_from_joint_prefix
from environment.wrappers.util_w import update_obs_space
from environment.utils.vision import insight, in_cone2d


class HealthWrapper(gym.Wrapper):
    '''
        Adds health mechanics to agents.
        Args:
            starting_health (float): number of times a food item can be eaten
                                   before it disappears
    '''
    def __init__(self, env, starting_health=100.0):
        super().__init__(env)
        self.starting_health = starting_health
        self.n_agents = self.metadata['n_agents']

        # Reset obs space
        self.observation_space = update_obs_space(self.env, {'agents_health': (self.n_agents, 1)})


    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        # Reset agents' healths
        self.agents_health = np.full((self.n_agents, 1), self.starting_health)
        self.metadata['agents_health'] = self.agents_health

        return self.observation(obs)

    def observation(self, obs):
        # Add agents' healths to obersvations
        obs['agents_health'] = self.metadata['agents_health']
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        health_rew = np.array([0 for a in range(self.n_agents)])
        agents_hp = self.metadata['agents_health']

        assert np.array(agents_hp).shape != np.array(rew).shape, f"Shape of agent's health does not match reward's shape"

        for i, ah in enumerate(agents_hp):
            if (ah[0] <= 0):
                health_rew[i] += -1000
                health_rew[1 - i] += 1000
                done = True
        
        rew += health_rew
        return self.observation(obs), rew, done, info
