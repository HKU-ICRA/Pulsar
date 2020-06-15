import numpy as np
import sys
from copy import deepcopy

from architecture.entity_encoder.entity_formatter import Entity_formatter


class InfoMasker:

    def __init__(self, n_agents, mjco_ts, n_substeps):
        self.n_agents = n_agents
        self.mjco_ts = mjco_ts
        self.n_substeps = n_substeps
        self.entity_formatter = Entity_formatter()
        # Frequency parameters
        self.my_qpos_freq = 0.005
        self.my_qvel_freq = 0.005
        self.local_qvel_freq = 0.005
        self.teammate_info_freq = 0.020
        self.opponent_info_freq = 0.060
        self.rrsystem_info_freq = 0.050
        self.my_qpos_std = 0.0001
        self.my_qvel_std = 0.0001
        self.local_qvel_std = 0.0001
        self.teammate_info_std = 0.001
        self.opponent_info_std = 0.001
        self.rrsystem_info_std = 0.0001

    def generate_info_trajectory(self, t_per_eval, t_per_eval_std):
        # Create sampling timesteps
        info_sample_ts = [self.secs_to_steps(np.random.normal(t_per_eval, t_per_eval_std))]
        max_sample_ts = 0
        while max_sample_ts < self.secs_to_steps(3 * 60):
            info_sample_ts.append(
                    self.secs_to_steps(np.random.normal(t_per_eval, t_per_eval_std)) + info_sample_ts[-1]
            )
            max_sample_ts = max(max_sample_ts, info_sample_ts[-1])
        return info_sample_ts
    
    def reset_masker(self):
        # Trajectories for obs
        self.obs_trajectory = [{
            "my_qpos":                          [self.generate_info_trajectory(self.my_qpos_freq, self.my_qpos_std), 0],
            "my_qvel":                          [self.generate_info_trajectory(self.my_qvel_freq, self.my_qvel_std), 0],
            "local_qvel":                       [self.generate_info_trajectory(self.local_qvel_freq, self.local_qvel_std), 0],
            "teammate_qpos":                    [self.generate_info_trajectory(self.teammate_info_freq, self.teammate_info_std), 0],
            "opponent1_qpos":                   [self.generate_info_trajectory(self.opponent_info_freq, self.opponent_info_std), 0],
            "opponent2_qpos":                   [self.generate_info_trajectory(self.opponent_info_freq, self.opponent_info_std), 0],
            "my_hp":                            [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0],
            "teammate_hp":                      [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0],
            "opponent1_hp":                     [self.generate_info_trajectory(self.opponent_info_freq, self.opponent_info_std), 0],
            "opponent2_hp":                     [self.generate_info_trajectory(self.opponent_info_freq, self.opponent_info_std), 0],
            "my_projs":                         [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0],
            "teammate_projs":                   [self.generate_info_trajectory(self.rrsystem_info_freq, self.teammate_info_std), 0],
            "opponent1_projs":                  [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0],
            "opponent2_projs":                  [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0],
            "my_armors":                        [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0],
            "teammate_armors":                  [self.generate_info_trajectory(self.rrsystem_info_freq, self.teammate_info_std), 0]
        } for _ in range(self.n_agents)]
        for ai in range(self.n_agents):
            hp_deduct_traj = [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0]
            self.obs_trajectory[ai]["my_hp_deduct"] = deepcopy(hp_deduct_traj)
            self.obs_trajectory[ai]["my_hp_deduct_res"] = deepcopy(hp_deduct_traj)
            zone_traj = [self.generate_info_trajectory(self.rrsystem_info_freq, self.rrsystem_info_std), 0]
            self.obs_trajectory[ai]["zone_1"] = deepcopy(zone_traj)
            self.obs_trajectory[ai]["zone_2"] = deepcopy(zone_traj)
            self.obs_trajectory[ai]["zone_3"] = deepcopy(zone_traj)
            self.obs_trajectory[ai]["zone_4"] = deepcopy(zone_traj)
            self.obs_trajectory[ai]["zone_5"] = deepcopy(zone_traj)
            self.obs_trajectory[ai]["zone_6"] = deepcopy(zone_traj)
        # Obs for agents
        self.agent_obs = self.entity_formatter.get_empty_obs_with_shapes(1, self.n_agents)
        self.agent_obs_mask = [{k: 0 for k in self.agent_obs.keys()} for _ in range(self.n_agents)]
        self.buffered_agent_obs = deepcopy(self.agent_obs)

    def step(self, env_ts, entities):     
        for ai in range(self.n_agents):
            for k in self.obs_trajectory[ai].keys():
                if env_ts == 0:
                    self.buffered_agent_obs[k][0, ai] = deepcopy(entities[ai][k])
                elif self.obs_trajectory[ai][k][1] < len(self.obs_trajectory[ai][k][0]) and self.obs_trajectory[ai][k][0][self.obs_trajectory[ai][k][1]] >= env_ts:
                    self.agent_obs[k][0, ai] = deepcopy(self.buffered_agent_obs[k][0, ai])
                    self.agent_obs_mask[ai][k] = 1
                    self.obs_trajectory[ai][k][1] += 1
                    self.buffered_agent_obs[k][0, ai] = deepcopy(entities[ai][k])
        
    def get_masked_entities(self, agent_no):
        masks_of_obs = deepcopy(self.agent_obs_mask[agent_no])
        masked_obs = dict()
        for k, observable in masks_of_obs.items():
            if observable:
                masked_obs[k] = self.agent_obs[k][:, agent_no:agent_no+1]
            else:
                masked_obs[k] = np.zeros(self.agent_obs[k][:, agent_no:agent_no+1].shape)
            self.agent_obs_mask[agent_no][k] = 0
        return masks_of_obs, masked_obs

    def secs_to_steps(self, secs):
        return int(secs / (self.mjco_ts * self.n_substeps))
