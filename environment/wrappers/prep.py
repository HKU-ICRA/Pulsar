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


class PrepWrapper(gym.Wrapper):
    '''
        Add variables and mechanisms needed before any wrapper.
    '''
    def __init__(self, env):
        super().__init__(env)
        # Set starting hp
        self.starting_health = 2000.0
        self.metadata['starting_health'] = self.starting_health
        self.n_agents = self.metadata['n_agents']
        # Reset obs space
        self.observation_space = update_obs_space(self.env, {'agents_health': [self.n_agents, 1]})
        self.observation_space = update_obs_space(self.env, {'agent_teams': [self.n_agents, 1]})

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Reset agents' healths
        self.agents_health = np.array([[self.starting_health] for _ in range(self.n_agents)])
        # Store agents' qpos
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        return self.observation(obs)

    def observation(self, obs):
        obs['agent_teams'] = np.array([[self.metadata['agent_infos'][i]['team']] for i in range(self.n_agents)])
        return obs        

    def minus_hp(self, agent_idx, hp):
        """
            Args:
                agent_idx: id of agent
                hp: health of agent to minus
        """
        self.agents_health[agent_idx][0] -= hp
        self.agents_health[agent_idx][0] = max(self.agents_health[agent_idx][0], 0.0)
    
    def add_hp(self, agent_idx, hp):
        """
            Args:
                agent_idx: id of agent
                hp: health of agent to add
        """
        self.agents_health[agent_idx][0] += hp
    
    def get_hp(self):
        return self.agents_health

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        sim = self.unwrapped.sim
        for aqidx in self.agent_qpos_idxs:
            agent_qpos = sim.data.qpos[aqidx]
            if agent_qpos[0] < 0 or agent_qpos[0] > 8540.50 or agent_qpos[1] < 0 or agent_qpos[1] > 5540.50:
                done = True
        rew = np.array([0.0 for _ in range(2)])
        info['lasting_rew'] = [0.0 for _ in range(2)]
        return self.observation(obs), rew, done, info
