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


class SimpleWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.n_agents = env.metadata['n_agents']
        # Reset obs space
        #self.observation_space = update_obs_space(self.env, {'target_pos': (2, 1)})


    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        #self.random_pos = np.array([[2.0], [-2.0]])#np.array(np.random.uniform(low=-3.0, high=3.0, size=(2, 1)))
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        self.agent_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'agent{i}')
                                        for i in range(self.n_agents)])

        return self.observation(obs)

    def observation(self, obs):
        #obs['target_pos'] = self.random_pos
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        simple_rew = np.array([0.0 for a in range(1)])

        sim = self.unwrapped.sim
        agent_qpos = sim.data.qpos[self.agent_qpos_idxs]

        #if abs(agent0_body[0] - self.random_pos[0][0]) <= 0.1:
        #    done = True
        #else:
        dist0 = np.linalg.norm(np.array(agent_qpos[0][0:2]) - np.array((5.0,  5.0)))
        if (dist0 < 0.1):
            done = True
        elif agent_qpos[0][0] < 0.0 or agent_qpos[0][1] < 0.0:
            simple_rew[0] += -1000.0
            done = True
        elif agent_qpos[0][0] > 10.0 or agent_qpos[0][1] > 10.0:
            simple_rew[0] += -1000.0
            done = True
        else:
            simple_rew[0] += -dist0
        #elif (abs(agent_qpos[1][0] - 4.2) < 0.1 and abs(agent_qpos[1][1] - 4.2) < 0.1):
        #    simple_rew[1] += 5
        
        #dist1 = np.linalg.norm(np.array(agent_qpos[1][0:2]) - np.array((4.2,  4.2)))
        #simple_rew[1] += -dist1 * 0.01
        
        rew += simple_rew
        return self.observation(obs), rew, done, info
