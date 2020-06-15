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


class OutcomeWrapper(gym.Wrapper):
    '''
        Adds reward according to outcome of match.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.metadata['n_agents']

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get new agents' info
        self.agent_infos = self.metadata['agent_infos']
        # Agent dead checker
        self.agent_status = [True for _ in range(self.n_agents)]
        return self.observation(obs)

    def observation(self, obs):
        return obs        

    def get_outcome(self, obs):
        """
            Returns:
                boolean for 'red' and 'blue. True if team lost, else False.
        """
        hps = self.env.get_hp()
        # Check 1st win-con: Team with all destroyed robot loses
        outcome = {'red': True, 'blue': True}
        for i in range(self.n_agents):
            if hps[i][0] > 0:
                outcome[self.agent_infos[i]['team']] = False
        # Check 2nd win-con: When match-time is up, team with higher damage output wins
        if self.env.get_ts() >= self.env.get_horizon():
            # First check if we have a winner/loser
            for k in outcome.keys():
                if outcome[k]:
                    return outcome
            # Otherwise check 2nd win-con
            outcome = {'red': True, 'blue': True}
            extra_hps = self.env.get_extra_hps()
            redt_dd, bluet_dd = 0, 0
            dtaken = {'red': 0, 'blue': 0}
            for i in range(self.n_agents):
                agent_hp = self.metadata['starting_health'] + extra_hps[i]
                dtaken[self.agent_infos[i]['team']] += agent_hp - max(0, hps[i][0])
            if dtaken['red'] < dtaken['blue']:
                outcome['red']  = False
            elif dtaken['red'] > dtaken['blue']:
                outcome['blue']  = False
            else:
                outcome['red'] = False
                outcome['blue'] = False
        return outcome

    def get_outcome_rewards(self, obs, action, info):
        outcome = self.get_outcome(obs)
        main_agent_team = self.agent_infos[0]['team']
        opponent_team = [k for k in outcome.keys() if k != main_agent_team][0]     
        # Reward for winning / losing and True reward (win/lose), or general reward if no outcome yet
        reward_4_winning = 10000
        if outcome[main_agent_team]:
            true_rew = 'lose'
            done = True
            rew = np.array([-reward_4_winning, -reward_4_winning])
        elif outcome[opponent_team]:
            true_rew = 'win'
            done = True
            rew = np.array([reward_4_winning, reward_4_winning])
        else:        
            true_rew = 'draw'
            done = False
            # Reward for damage taken and dealt
            """
            hps = self.env.get_hp()
            extra_hps = self.env.get_extra_hps()
            redt_dd, bluet_dd = 0, 0
            dtaken = {'red': 0, 'blue': 0}
            for i in range(self.n_agents):
                agent_hp = self.metadata['starting_health'] + extra_hps[i]
                dtaken[self.agent_infos[i]['team']] += agent_hp - max(0, hps[i][0])
            if self.agent_infos[0]['team'] == 'red':
                dmg_taken_dealt_rew = np.array([(dtaken['blue'] - dtaken['red']) for _ in range(2)])
            else:
                dmg_taken_dealt_rew = np.array([dtaken['red'] - dtaken['blue'] for _ in range(2)])
            rew = dmg_taken_dealt_rew * 0.5
            """
            hps = self.env.get_hp()
            for i in range(self.n_agents):
                if hps[i][0] <= 0 and self.agent_status[i]:
                    if self.agent_infos[0]['team'] != self.agent_infos[i]['team']:
                        info['lasting_rew'][0] += 1000
                        info['lasting_rew'][1] += 1000
                    self.agent_status[i] = False
                elif not self.agent_status[i] and hps[i][0] > 0:
                    self.agent_status[i] = True
            team_avg_hp = {'red': 0, 'blue': 0}
            rew = np.array([0.0 for _ in range(2)])
            for i in range(self.n_agents):
                team_avg_hp[self.agent_infos[i]['team']] += hps[i][0]
            team_avg_hp['red'] /= 2.0
            team_avg_hp['blue'] /= 2.0
            for i in range(2):
                if self.agent_infos[i]['team'] == 'red':
                    rew[i] += hps[i][0] - team_avg_hp['blue']
                else:
                    rew[i] += hps[i][0] - team_avg_hp['red']
            rew = rew * 0.5
            # Reward for action penalty
            for i in range(2):
                rew[i] += -(abs(action['action_movement'][i, 0]) + abs(action['action_movement'][i, 1]) + abs(action['action_movement'][i, 2])) / 3000.0
        return rew, true_rew, done, info
            
    def step(self, action):
        # Prevent movements from downed robots
        agent_hps = self.env.get_hp()
        for i in range(self.n_agents):
            if agent_hps[i][0] <= 0:
                action['action_movement'][i] = np.array([0, 0, 0])
        # Handle outcomes
        obs, rew, done, info = self.env.step(action)
        outcome_rew, true_rew, outcome_done, info = self.get_outcome_rewards(obs, action, info)
        rew += outcome_rew
        info['true_rew'] = true_rew
        done = done | outcome_done
        return self.observation(obs), rew, done, info
