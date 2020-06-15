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


class CollisionWrapper(gym.Wrapper):
    '''
        Adds collision detection and punishment.
        Also provides info for agents' distance to obstacle.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = n_agents = self.metadata['n_agents']
        # Collision force threshold
        self.threshold = 0.0
        self.observation_space = update_obs_space(self.env, {'colli_dmg': [self.n_agents, 1]})
        
    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get indexes of qvel for movement debuff
        self.agent_qvel_idxs = [qvel_idxs_from_joint_prefix(sim, f'agent{i}') for i in range(self.n_agents)]
        self.movement_debuffed = [False for _ in range(self.n_agents)]
        # Get indexes of qpos of agents
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        # Get indexes of obstacles pos
        self.obstacles_id = [
            sim.model.body_name2id("B1"),
            sim.model.body_name2id("B2"),
            sim.model.body_name2id("B3"),
            sim.model.body_name2id("B4"),
            sim.model.body_name2id("B5"),
            sim.model.body_name2id("B6"),
            sim.model.body_name2id("B7"),
            sim.model.body_name2id("B8"),
            sim.model.body_name2id("B9")
        ]
        # Damage recorder
        self.colli_dmg_taken = [[0] for _ in range(self.n_agents)]
        # Timer to account for the interval between armor collision deduction
        self.agent_collision_timer = [-1000000 for _ in range(self.n_agents)]
        return self.observation(obs)

    def observation(self, obs):
        obs['colli_dmg'] = np.array(self.colli_dmg_taken)
        return obs

    def wall_collision(self, info):
        sim = self.unwrapped.sim
        self.colli_dmg_taken = [[0] for _ in range(self.n_agents)]
        for idx in range(self.n_agents):
            if self.env.get_ts() - self.agent_collision_timer[idx] < self.env.secs_to_steps(0.05):
                continue
            else:
                self.agent_collision_timer[idx] = self.env.get_ts()
            agent_collisions = {
                'wheel_fl_touch': sim.data.get_sensor(f'agent{idx}:wheel_fl_touch'),
                'wheel_fr_touch': sim.data.get_sensor(f'agent{idx}:wheel_fr_touch'),
                'wheel_bl_touch': sim.data.get_sensor(f'agent{idx}:wheel_bl_touch'),
                'wheel_br_touch': sim.data.get_sensor(f'agent{idx}:wheel_br_touch'),
                'armor_r_touch': sim.data.get_sensor(f'agent{idx}:armor_r_touch'),
                'armor_b_touch': sim.data.get_sensor(f'agent{idx}:armor_b_touch'),
                'armor_f_touch': sim.data.get_sensor(f'agent{idx}:armor_f_touch'),
                'armor_l_touch': sim.data.get_sensor(f'agent{idx}:armor_l_touch'),
                'bar_f_touch': sim.data.get_sensor(f'agent{idx}:bar_f_touch'),
                'bar_b_touch': sim.data.get_sensor(f'agent{idx}:bar_b_touch'),
                'body_touch': sim.data.get_sensor(f'agent{idx}:body_touch')
            }
            for v in agent_collisions.values():
                if v  > self.threshold:
                    self.env.minus_hp(idx, 10)
                    self.colli_dmg_taken[idx][0] += 10
                    if idx == 0 or idx == 1:
                        info['lasting_rew'][idx] += -1
        return info

    def movement_debuff(self, action):
        for i in range(self.n_agents):
            if self.movement_debuffed[i]:
                action['action_movement'][i] = np.array([0, 0, 0])
        return action
    
    def movement_debuff_check(self, obs):
        for i in range(self.n_agents):
            no_move = obs['Agent:buff'][i][3][0]
            if no_move:
                self.movement_debuffed[i] = True
            else:
                self.movement_debuffed[i] = False

    def step(self, action):
        action = self.movement_debuff(action)
        obs, rew, done, info = self.env.step(action)
        self.movement_debuff_check(obs)
        info = self.wall_collision(info)
        return self.observation(obs), rew, done, info
