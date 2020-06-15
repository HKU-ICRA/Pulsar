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


class BuffWrapper(gym.Wrapper):
    '''
        Adds buff/debuff mechanics to buff/debuff zones.
        Args:
            env: simulator environment
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = n_agents = self.metadata['n_agents']
        # Setup obs space
        self.observation_space = update_obs_space(self.env, {'F1': [4, 1],
                                                             'F2': [4, 1],
                                                             'F3': [4, 1],
                                                             'F4': [4, 1],
                                                             'F5': [4, 1],
                                                             'F6': [4, 1]
                                                                         })
        # Setup buff status of each agent
        self.observation_space = update_obs_space(self.env, {'Agent:buff': [n_agents, 4, 1]})
        self.agents_buff_status = np.array([[[0],[0],[0],[0]] for _  in range(self.n_agents)])

    def buff_randomize(self):
        # Randomize all the buffs/debuffs
        f1 = np.zeros((4, 1))
        f1[np.random.randint(0, 4)][0] = 1
        self.fzones['F1'] = f1
        f2 = np.zeros((4, 1))
        f2[np.random.randint(0, 4)][0] = 1
        self.fzones['F2'] = f2
        f3 = np.zeros((4, 1))
        f3[np.random.randint(0, 4)][0] = 1
        self.fzones['F3'] = f3
        f4 = np.zeros((4, 1))
        f4[np.random.randint(0, 4)][0] = 1
        self.fzones['F4'] = f4
        f5 = np.zeros((4, 1))
        f5[np.random.randint(0, 4)][0] = 1
        self.fzones['F5'] = f5
        f6 = np.zeros((4, 1))
        f6[np.random.randint(0, 4)][0] = 1
        self.fzones['F6'] = f6

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get new agents' info
        self.agent_infos = self.metadata['agent_infos']
        # Reset buff states
        self.reset_buffs()
        # Get indexes of qpos of agents
        self.agent_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        # Get buff/debuff zone indexes
        self.fzone_idxs = [
            np.array(sim.model.site_name2id(f"F1")),
            np.array(sim.model.site_name2id(f"F2")),
            np.array(sim.model.site_name2id(f"F3")),
            np.array(sim.model.site_name2id(f"F4")),
            np.array(sim.model.site_name2id(f"F5")),
            np.array(sim.model.site_name2id(f"F6"))
        ]
        # Reset agent buff status. 1st = restoration. 2nd = Projectile. 3rd = No shoot. 4th = No move.
        self.agents_buff_status = [[[0],[0],[0],[0]] for _  in range(self.n_agents)]
        self.buff_randomize()
        # Reset buff duration counter
        self.debuff_dur = []
        return self.observation(obs)

    def observation(self, obs):
        for k, v in self.fzones.items():
            obs[k] = np.array(v, dtype=np.float32)
        obs['Agent:buff'] = np.array(self.agents_buff_status)
        return obs

    def reset_buffs(self):
        self.fzones = {
            'F1': [[0], [0], [0], [0]],
            'F2': [[0], [0], [0], [0]],
            'F3': [[0], [0], [0], [0]],
            'F4': [[0], [0], [0], [0]],
            'F5': [[0], [0], [0], [0]],
            'F6': [[0], [0], [0], [0]]
        }

    def buff_detection_debug(self, team, buff_idx, current_buff_idx):
        print(f"A buff has been activated by {team} team.")
        if current_buff_idx == 0:
            buff = "Restoration"
        elif current_buff_idx == 1:
            buff = "Projectile"
        elif current_buff_idx == 2:
            buff = "No shoot"
        elif current_buff_idx == 3:
            buff = "No move"
        print(f"It corresponds to zone F{buff_idx} which had buff: " + buff)

    def buff_detection(self, obs, info):
        sim = self.unwrapped.sim
        for idx in range(6):
            # Try to find index with value "1"
            current_buff_idx = np.argmax(self.fzones[f'F{idx+1}'])
            if self.fzones[f'F{idx+1}'][current_buff_idx][0] == 1:
                # Buff for this zone is activated so check if any agents on it
                mindist, agent = np.inf, 0
                for aidx, aq in enumerate(self.agent_idxs):
                    agent_qpos = sim.data.qpos[aq][0:2]
                    vector = np.array(agent_qpos) - np.array(sim.data.site_xpos[self.fzone_idxs[idx]])[0:2]
                    dist = np.linalg.norm(vector)
                    if dist < mindist:
                        mindist = dist
                        agent = aidx
                if mindist <= 250.0:
                    # Activate zone
                    #self.buff_detection_debug(self.agent_infos[agent]['team'], idx, current_buff_idx)
                    self.fzones[f'F{idx+1}'][current_buff_idx][0] = 0
                    self.agents_buff_status[agent][current_buff_idx][0] = 1
                    if current_buff_idx > 1:
                        # Is a debuff
                        self.debuff_dur.append([agent, current_buff_idx, self.env.get_ts()])
                    if agent == 0 or agent == 1:
                        if current_buff_idx > 1:
                            info['lasting_rew'][agent] += -200.0
                        else:
                            info['lasting_rew'][agent] += 200.0
        return info

    def debuff_removal(self, obs):
        # First check for over-laps
        nonoverlapped = []
        for idx1, debuff1 in enumerate(self.debuff_dur):
            a_idx1, f_idx1, init_ts1 = debuff1[0], debuff1[1], debuff1[2]
            overlapped = False
            for idx2, debuff2 in enumerate(self.debuff_dur):
                a_idx2, f_idx2, init_ts2 = debuff2[0], debuff2[1], debuff2[2]
                if idx1 != idx2:
                    if a_idx1 == a_idx2 and f_idx1 == f_idx2:
                        if init_ts1 < init_ts2:
                            overlapped = True
                        elif init_ts1 == init_ts2:
                            self.debuff_dur[idx2][2] = -1
            if not overlapped:
                nonoverlapped.append(self.debuff_dur[idx1])
        self.debuff_dur = nonoverlapped
        # Now do removal
        for idx, debuff in enumerate(self.debuff_dur):
            a_idx, f_idx, init_ts = debuff[0], debuff[1], debuff[2]
            if self.env.get_ts() - init_ts >= self.env.secs_to_steps(10):
                self.agents_buff_status[a_idx][f_idx][0] = 0
                self.debuff_dur.pop(idx)
    
    def set_buff_status(self, agent_idx, buff_idx, status):
        """
            Args:
                agent_idx: the agent id
                buff_idx: the buff id
                status: status of buff to set to (0 or 1)
        """
        self.agents_buff_status[agent_idx][buff_idx][0] = status

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.debuff_removal(obs)
        if (self.env.get_ts() != 0 and self.env.get_ts() == self.env.secs_to_steps(60)):
            # Once one minute has passed, randomize the buffs again and reset buffs
            self.reset_buffs()
            self.buff_randomize()
        elif (self.env.get_ts() != 0 and self.env.get_ts() == self.env.secs_to_steps(120)):
            # Once two minutes has passed, randomize the buffs again and reset buffs
            self.reset_buffs()
            self.buff_randomize()
        info = self.buff_detection(obs, info)
        #print(self.unwrapped.sim.model.ntex)
        return self.observation(obs), rew, done, info
