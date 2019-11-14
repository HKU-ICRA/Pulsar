import sys
import random
from copy import deepcopy
from itertools import compress
import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple
from mujoco_worldgen.util.rotation import mat2quat
from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs, joint_qpos_idxs, body_names_from_joint_prefix
from environment.wrappers.util_w import update_obs_space
from environment.utils.vision import insight, in_cone2d

class ProjectileManager:
    '''
        Manages the projectile hit-chance
        Args:
            n_agents: number of agents
            projectile_speed: speed of projectile (in whatever units)
            tolerance: distance from target in which counts as a hit
            dodge_tolerance: distance of the new armor position from the old armor position in which counts as a hit
    '''
    def __init__(self, n_agents, projectile_speed, tolerance, dodge_tolerance):
        self.n_agents = n_agents
        self.projectile_speed = projectile_speed
        self.tolerance = tolerance
        self.dodge_tolerance = dodge_tolerance
        self.projectile_buffer = []
    
    def add_2_buffer(self, sim, armor_qpos, barrel_pos, agent_id):
        '''
            Args:
                armor_qpos: qpos range for the armor
                barrel_pos: actual position (xyz) of the barrel
                agent_id: actual index of the armor's agent
        '''
        armor_pos = sim.data.geom_xpos[armor_qpos]
        dist = np.linalg.norm(np.array(armor_pos) - np.array(barrel_pos))
        self.projectile_buffer.append([dist, armor_qpos, armor_pos, agent_id])
    
    def query(self, sim):
        hits = [0 for i in range(self.n_agents)]
        for i, proj in enumerate(self.projectile_buffer):
            proj[0] -= self.projectile_speed
            if (proj[0] <= self.tolerance):
                dist = np.linalg.norm(np.array(sim.data.geom_xpos[proj[1]]) - np.array(proj[2]))
                if dist <= self.dodge_tolerance:
                    hits[proj[3]] += 1
                self.projectile_buffer.pop(i)
        return hits
    
    def reset(self):
        self.projectile_buffer = []


class ProjectileWrapper(gym.Wrapper):
    '''
        Allows agent to shoot a projectile towards an armor whenever the opponent and the armor is visible
        Args:
            armors: list of armors that the agent can see
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.projmang = ProjectileManager(self.n_agents, 0.1, 0.01, 0.1 * 2.0)
    
    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        
        self.projmang.reset()
                
        # Cache geom ids
        self.barrel_index = [i for i in range(self.n_agents)]
        self.barrel_body_idxs = np.array([sim.model.body_name2id(f"agent{i}:barrel_head") for i in range(self.n_agents)])
        self.agent_barrel_idxs = np.array([sim.model.geom_name2id(f"agent{i}:barrel_head_geom") for i in range(self.n_agents)])

        self.armor_index = [i for i in range(self.n_agents)]
        self.agent_armors = [sim.model.geom_name2id(f"agent{i}:armor1") for i in range(self.n_agents)]

        self.armor_index += [i for i in range(self.n_agents)]
        self.agent_armors += [sim.model.geom_name2id(f"agent{i}:armor2") for i in range(self.n_agents)]

        self.armor_index += [i for i in range(self.n_agents)]
        self.agent_armors += [sim.model.geom_name2id(f"agent{i}:armor3") for i in range(self.n_agents)]

        self.armor_index += [i for i in range(self.n_agents)]
        self.agent_armors += [sim.model.geom_name2id(f"agent{i}:armor4") for i in range(self.n_agents)]

        self.agent_armors = np.array(self.agent_armors)

        return obs

    def shoot_projectile(self, obs, projectile_rew):
        sim = self.unwrapped.sim
        for ib, b in enumerate(self.agent_barrel_idxs):
            barrel_pos = sim.data.geom_xpos[b]

            armors_is = [(insight(sim, b, geom2_id=g2, pt2=None, dist_thresh=100.0, check_body=False)) for g2 in self.agent_armors]

            barrel_pos_2d = np.array([[barrel_pos[0], barrel_pos[1]]])
            armors_pos_2d = [np.array([sim.data.geom_xpos[g2][0], sim.data.geom_xpos[g2][1]]) for g2 in self.agent_armors] 
            angle_pos_2d = sim.data.qpos[joint_qpos_idxs(sim, f"agent" + str(ib) + ":rz")]

            armors_incone = in_cone2d(barrel_pos_2d,
                                  np.array(angle_pos_2d),
                                  0.785,
                                  np.array(armors_pos_2d))

            armors_is = (np.array(armors_is) & np.array(armors_incone))[0]
            armor_to_shoot, armor_ts_idx, agent_id = np.inf, None, None

            for i, armor in enumerate(self.agent_armors):
                if (armors_is[i]) and self.barrel_index[ib] != self.armor_index[i]:
                    armor_pos = sim.data.geom_xpos[armor]
                    diff = np.linalg.norm(np.array(armor_pos) - np.array(barrel_pos))
                    if diff < armor_to_shoot:
                        armor_to_shoot = diff
                        armor_ts_idx = armor
                        agent_id = self.armor_index[i]
            
            if armor_ts_idx != None:
                armor_pos = sim.data.geom_xpos[armor_ts_idx]
                self.projmang.add_2_buffer(sim, armor_ts_idx, barrel_pos, agent_id)
            
        hits = self.projmang.query(sim)
        for i, h in enumerate(hits):
            self.metadata['agents_health'][i] -= h * 1.5
            #projectile_rew[i] += -1.0
            #projectile_rew[1 - i] += 1.0
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        projectile_rew = np.array([0 for a in range(self.n_agents)], dtype=np.float64)
        self.shoot_projectile(obs, projectile_rew)
        #rew += projectile_rew
        #sim = self.unwrapped.sim
        #print(self.metadata['agents_health'])
        #rew += projectile_rew
        #print(self.metadata['agents_health'])
        return obs, rew, done, info
