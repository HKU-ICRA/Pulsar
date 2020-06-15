import numpy as np


class DomainRandomization:

    def __init__(self, n_agents):
        self.n_agents = n_agents
        # noise randomization parameters (uniform distribution)
        self.noise = {
            "my_qpos":                  [0.00, 0.003],
            "my_qvel":                  [0.00, 0.003],
            "teammate_qpos":            [0.00, 0.003],
            "opponent1_qpos":           [0.00, 0.003],
            "opponent2_qpos":           [0.00, 0.003]
        }
        # info masker randomization parameters (uniform distribution)
        self.info_masker = {
            "my_qpos_freq": [0.003, 0.007],
            "my_qpos_std": [0.0001, 0.0003],
            "my_qvel_freq": [0.003, 0.007],
            "my_qvel_std": [0.0001, 0.0003],
            "local_qvel_freq": [0.003, 0.007],
            "local_qvel_std": [0.0001, 0.0003],
            "teammate_info_freq": [0.05, 0.08],
            "teammate_info_std": [0.001, 0.005],
            "opponent_info_freq": [0.1, 0.15],
            "opponent_info_std": [0.01, 0.03],
            "rrsystem_info_freq": [0.03, 0.07],
            "rrsystem_info_std": [0.0001, 0.0003],
        }
        # Kv randomization (uniform distribution)
        self.kv = {
            "motortx_kv": [10, 100],
            "motorty_kv": [10, 100],
            "motorrz_kv": [10, 100]
        }
        # Barrel_sight randomization (normal distribution)
        self.barrel_sight = {
            "barrel_sight_x": [0, 1],
            "barrel_sight_y": [-11, 3],
            "barrel_sight_z": [138, 3]
        }

    def noise_randomization(self, noise):
        noise.my_qpos_std =             np.random.uniform(self.noise["my_qpos"][0], self.noise["my_qpos"][1])
        noise.my_qvel_std =             np.random.uniform(self.noise["my_qvel"][0], self.noise["my_qvel"][1])
        noise.teammate_qpos_std =       np.random.uniform(self.noise["teammate_qpos"][0], self.noise["teammate_qpos"][1])
        noise.opponent1_qpos_std =      np.random.uniform(self.noise["opponent1_qpos"][0], self.noise["opponent1_qpos"][1])
        noise.opponent2_qpos_std =      np.random.uniform(self.noise["opponent2_qpos"][0], self.noise["opponent2_qpos"][1])
        return noise
    
    def info_masker_randomization(self, info_masker):
        info_masker.my_qpos_freq =          np.random.uniform(self.info_masker["my_qpos_freq"][0], self.info_masker["my_qpos_freq"][1])
        info_masker.my_qvel_freq =          np.random.uniform(self.info_masker["my_qvel_freq"][0], self.info_masker["my_qvel_freq"][1])
        info_masker.local_qvel_freq =       np.random.uniform(self.info_masker["local_qvel_freq"][0], self.info_masker["local_qvel_freq"][1])
        info_masker.teammate_info_freq =    np.random.uniform(self.info_masker["teammate_info_freq"][0], self.info_masker["teammate_info_freq"][1])
        info_masker.opponent_info_freq =    np.random.uniform(self.info_masker["opponent_info_freq"][0], self.info_masker["opponent_info_freq"][1])
        info_masker.rrsystem_info_freq =    np.random.uniform(self.info_masker["rrsystem_info_freq"][0], self.info_masker["rrsystem_info_freq"][1])
        info_masker.my_qpos_std =           np.random.uniform(self.info_masker["my_qpos_std"][0], self.info_masker["my_qpos_std"][1])
        info_masker.my_qvel_std =           np.random.uniform(self.info_masker["my_qvel_std"][0], self.info_masker["my_qvel_std"][1])
        info_masker.local_qvel_std =        np.random.uniform(self.info_masker["local_qvel_std"][0], self.info_masker["local_qvel_std"][1])
        info_masker.teammate_info_std =     np.random.uniform(self.info_masker["teammate_info_std"][0], self.info_masker["teammate_info_std"][1])
        info_masker.opponent_info_std =     np.random.uniform(self.info_masker["opponent_info_std"][0], self.info_masker["opponent_info_std"][1])
        info_masker.rrsystem_info_std =     np.random.uniform(self.info_masker["rrsystem_info_std"][0], self.info_masker["rrsystem_info_std"][1])
        return info_masker

    def kv_randomization(self, env):
        for ai in range(self.n_agents):
            # motortx
            motortx_kv = np.random.uniform(self.kv["motortx_kv"][0], self.kv["motortx_kv"][1])
            env.unwrapped.sim.model.actuator_gainprm[ai * 3][0] = motortx_kv
            env.unwrapped.sim.model.actuator_biasprm[ai * 3][2] = -motortx_kv
            # motorty
            motorty_kv = np.random.uniform(self.kv["motorty_kv"][0], self.kv["motorty_kv"][1])
            env.unwrapped.sim.model.actuator_gainprm[ai * 3 + 1][0] = motorty_kv
            env.unwrapped.sim.model.actuator_biasprm[ai * 3 + 1][2] = -motorty_kv
            # motorrz
            motorrz_kv = np.random.uniform(self.kv["motorrz_kv"][0], self.kv["motorrz_kv"][1])
            env.unwrapped.sim.model.actuator_gainprm[ai * 3 + 2][0] = motorrz_kv
            env.unwrapped.sim.model.actuator_biasprm[ai * 3 + 2][2] = -motorrz_kv

    def barrel_sight_randomization(self, env):
        for ai in range(self.n_agents):
            barrel_sight_idx = env.unwrapped.sim.model.site_name2id(f"agent{ai}:barrel_sight")
            bs_x = np.random.normal(self.barrel_sight["barrel_sight_x"][0], self.barrel_sight["barrel_sight_x"][1])
            bs_y = np.random.normal(self.barrel_sight["barrel_sight_y"][0], self.barrel_sight["barrel_sight_y"][1])
            bs_z = np.random.normal(self.barrel_sight["barrel_sight_z"][0], self.barrel_sight["barrel_sight_z"][1])
            env.unwrapped.sim.model.site_pos[barrel_sight_idx] = np.array([bs_x, bs_y, bs_z])
