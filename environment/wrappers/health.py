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
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.metadata['n_agents']

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get new agents' info
        self.agent_infos = self.metadata['agent_infos']
        # Record additional hp given by buff
        self.extra_hps = [0 for _ in range(self.n_agents)]
        return self.observation(obs)

    def observation(self, obs):
        obs['agents_health'] = self.env.get_hp()
        return obs        

    def get_extra_hps(self):
        return self.extra_hps

    def health_buff(self, obs):
        for i in range(self.n_agents):
            restore_hp = obs['Agent:buff'][i][0][0]
            if restore_hp:
                self.env.set_buff_status(agent_idx=i, buff_idx=0, status=0)
                for j in range(self.n_agents):
                    if self.agent_infos[j]['team'] == self.agent_infos[i]['team']:
                        self.env.add_hp(j, 200)
                        self.extra_hps[j] += 200

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.health_buff(obs)
        return self.observation(obs), rew, done, info
