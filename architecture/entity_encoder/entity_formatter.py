import numpy as np


class Entity_formatter():

    def __init__(self):
        pass
    
    def get_empty_obs(self):
        """
            Returns:
                dict with the name of all obs with empty lists
        """
        obs = {
            "my_qpos": [],
            "my_qvel": [],
            "local_qvel": [],
            "teammate_qpos": [],
            "opponent1_qpos": [],
            "opponent2_qpos": [],
            "my_hp": [],
            "teammate_hp": [],
            "opponent1_hp": [],
            "opponent2_hp": [],
            "my_projs": [],
            "teammate_projs": [],
            "opponent1_projs": [],
            "opponent2_projs": [],
            "my_armors": [],
            "teammate_armors": [],
            "my_hp_deduct": [],
            "my_hp_deduct_res": [],
            "zone_1": [],
            "zone_2": [],
            "zone_3": [],
            "zone_4": [],
            "zone_5": [],
            "zone_6": [],
        }
        return obs
    
    def get_empty_obs_with_shapes(self, nsteps, n_agents):
        obs = {
            "my_qpos": np.zeros([nsteps, n_agents, 3]),
            "my_qvel": np.zeros([nsteps, n_agents, 3]),
            "local_qvel": np.zeros([nsteps, n_agents, 2]),
            "teammate_qpos": np.zeros([nsteps, n_agents, 3]),
            "opponent1_qpos": np.zeros([nsteps, n_agents, 3]),
            "opponent2_qpos": np.zeros([nsteps, n_agents, 3]),
            "my_hp": np.zeros([nsteps, n_agents, 1]),
            "teammate_hp": np.zeros([nsteps, n_agents, 1]),
            "opponent1_hp": np.zeros([nsteps, n_agents, 1]),
            "opponent2_hp": np.zeros([nsteps, n_agents, 1]),
            "my_projs": np.zeros([nsteps, n_agents, 1]),
            "teammate_projs": np.zeros([nsteps, n_agents, 1]),
            "opponent1_projs": np.zeros([nsteps, n_agents, 1]),
            "opponent2_projs": np.zeros([nsteps, n_agents, 1]),
            "my_armors": np.zeros([nsteps, n_agents, 4]),
            "teammate_armors": np.zeros([nsteps, n_agents, 4]),
            "my_hp_deduct": np.zeros([nsteps, n_agents, 2]),
            "my_hp_deduct_res": np.zeros([nsteps, n_agents, 2]),
            "zone_1": np.zeros([nsteps, n_agents, 4]),
            "zone_2": np.zeros([nsteps, n_agents, 4]),
            "zone_3": np.zeros([nsteps, n_agents, 4]),
            "zone_4": np.zeros([nsteps, n_agents, 4]),
            "zone_5": np.zeros([nsteps, n_agents, 4]),
            "zone_6": np.zeros([nsteps, n_agents, 4])
        }
        return obs

    def concat_encoded_entity_obs(self, no_of_agents, agent_idx, all_obs):
        """
        Returns all encoded entities as a concaternation.
        Args:
            no_of_agents: the number of agents in total
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
            env_ts: the current timestep of the environment
        Returns:
            An array of masks, False if the obs is not observed, else True
            One hot encoding with shape [batch_size, n_entities, feature_size]
        """
        my_mask = []
        my_obs = dict()
        def append_mask_obs(mask, obs, obs_name):
            my_mask.append(mask)
            my_obs[obs_name] = obs.astype(dtype=np.float32)
        append_mask_obs(*self.encode_qpos(agent_idx, all_obs), "my_qpos")
        append_mask_obs(*self.encode_qvel(agent_idx, all_obs), "my_qvel")
        append_mask_obs(*self.encode_local_qvel(agent_idx, all_obs), "local_qvel")
        append_mask_obs(*self.encode_teammate_qpos(no_of_agents, agent_idx, all_obs), "teammate_qpos")
        append_mask_obs(*self.encode_opponent_qpos(no_of_agents, agent_idx, all_obs, 1), "opponent1_qpos")
        append_mask_obs(*self.encode_opponent_qpos(no_of_agents, agent_idx, all_obs, 2), "opponent2_qpos")
        append_mask_obs(*self.encode_robot_hp(agent_idx, all_obs), "my_hp")
        append_mask_obs(*self.encode_teammate_hp(no_of_agents, agent_idx, all_obs), "teammate_hp")
        append_mask_obs(*self.encode_opponent_hp(no_of_agents, agent_idx, all_obs, 1), "opponent1_hp")
        append_mask_obs(*self.encode_opponent_hp(no_of_agents, agent_idx, all_obs, 2), "opponent2_hp")
        append_mask_obs(*self.encode_launchable_projectiles(agent_idx, all_obs), "my_projs")
        append_mask_obs(*self.encode_teammate_projectiles(no_of_agents, agent_idx, all_obs), "teammate_projs")
        append_mask_obs(*self.encode_opponent_projectiles(no_of_agents, agent_idx, all_obs, 1), "opponent1_projs")
        append_mask_obs(*self.encode_opponent_projectiles(no_of_agents, agent_idx, all_obs, 2), "opponent2_projs")
        append_mask_obs(*self.encode_attacked_armors(agent_idx, all_obs), "my_armors")
        append_mask_obs(*self.encode_teammates_armors(no_of_agents, agent_idx, all_obs), "teammate_armors")
        append_mask_obs(*self.encode_hp_deduction(agent_idx, all_obs), "my_hp_deduct")
        append_mask_obs(*self.encode_hp_deduct_reason(agent_idx, all_obs), "my_hp_deduct_res")
        # Encode zones
        zone_mask, zone_obs = self.encode_zones(agent_idx, all_obs)
        my_mask += [zone_mask for _ in range(6)]
        for idx, zobs in enumerate(zone_obs):
            my_obs[f"zone_{idx+1}"] = zobs.astype(dtype=np.float32)
        return np.array(my_mask).astype(dtype=np.float32), my_obs
    
    def get_baseline(self, no_of_agents, agent_idx, all_obs, env_ts):
        """
            Baseline is only useful for agent 0 (the agent we're training)
            Args:
                no_of_agents: the number of agents in total
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
                env_ts: the current timestep of the environment
            Returns:
                All observations flattened
        """
        baseline = []
        def append_common_obs(common_obs_agent_idx):
            baseline.append(self.encode_robot_hp(common_obs_agent_idx, all_obs)[1].flatten())
            baseline.append(self.encode_launchable_projectiles(common_obs_agent_idx, all_obs)[1].flatten())
            baseline.append(self.encode_hp_deduction(common_obs_agent_idx, all_obs)[1].flatten())
            baseline.append(self.encode_hp_deduct_reason(common_obs_agent_idx, all_obs)[1].flatten())
            baseline.append(self.encode_attacked_armors(common_obs_agent_idx, all_obs)[1].flatten())
        def append_zero_obs():
            baseline.append(np.zeros([1, 1, 3]).flatten())
            baseline.append(np.zeros([1, 1, 3]).flatten())
            baseline.append(np.zeros([1, 1, 1]).flatten())
            baseline.append(np.zeros([1, 1, 1]).flatten())
            baseline.append(np.zeros([1, 1, 2]).flatten())
            baseline.append(np.zeros([1, 1, 2]).flatten())
            baseline.append(np.zeros([1, 1, 4]).flatten())
        # Self
        baseline.append(self.encode_qpos(agent_idx, all_obs)[1].flatten())
        baseline.append(self.encode_qvel(agent_idx, all_obs)[1].flatten())
        append_common_obs(agent_idx)
        # Teammate
        baseline.append(self.encode_teammate_qpos(no_of_agents, agent_idx, all_obs)[1].flatten())
        baseline.append(self.encode_teammate_qvel(no_of_agents, agent_idx, all_obs)[1].flatten())
        append_common_obs(1 - agent_idx)
        # Opponent 1
        baseline.append(self.encode_opponent_qpos(no_of_agents, agent_idx, all_obs, 1)[1].flatten())
        baseline.append(self.encode_opponent_qvel(no_of_agents, agent_idx, all_obs, 1)[1].flatten())
        append_common_obs(2)
        if no_of_agents > 3:
            # Opponent 2
            baseline.append(self.encode_opponent_qpos(no_of_agents, agent_idx, all_obs, 2)[1].flatten())
            baseline.append(self.encode_opponent_qvel(no_of_agents, agent_idx, all_obs, 2)[1].flatten())
            append_common_obs(3)
        else:
            append_zero_obs()
        # Zones
        encoded_zones = self.encode_zones(agent_idx, all_obs)[1]
        for encoded_zone in encoded_zones:
            baseline.append(encoded_zone.flatten())
        # Make baseline 2D
        baseline = np.concatenate(baseline, axis=0)
        return np.array([baseline]).astype(dtype=np.float32)

    def coord_transform(self, team, coords):
        """
            Args:
                team (string): agent's team
                coords: (1, 1, 3) of x,y,angle coords
            Returns:
                Coords but with respect to the agent's team
        """
        if team == 'blue':
            return coords
        else:
            angle = coords[0, 0, 2]
            if angle == 0:
                angle = -np.pi
            else:
                angle = -np.sign(angle) * (np.pi - abs(angle))
            return np.array([ [ [1.0 - coords[0, 0, 0], 1.0 - coords[0, 0, 1], angle] ] ]).astype(dtype=coords.dtype)
    
    def vel_transform(self, team, vels):
        """
            Args:
                team (string): agent's team
                vels: (1, 1, 3) of x,y,angle vels
            Returns:
                Vel but with respect to the agent's team
        """
        if team == 'blue':
            return vels
        else:
            return np.array([ [ [-vels[0, 0, 0], -vels[0, 0, 1], vels[0, 0, 2]] ] ]).astype(dtype=vels.dtype)
            
    def encode_qpos(self, agent_idx, all_obs):
        """
            Args:
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
            Returns:
                Mask, 1 if observed else 0
                qpos of agent
        """
        qpos = np.expand_dims(np.array(all_obs['observation_self'][agent_idx:agent_idx+1, 0:3], dtype=np.float32), axis=0)
        qpos = self.coord_transform(all_obs['agent_teams'][agent_idx][0], qpos)
        return 1, qpos
    
    def encode_qvel(self, agent_idx, all_obs):
        """
            Args:
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
            Returns:
                Mask, 1 if observed else 0
                qvel of agent
        """
        qvel = np.expand_dims(np.array(all_obs['observation_self'][agent_idx:agent_idx+1, 3:6], dtype=np.float32), axis=0)
        qvel = self.vel_transform(all_obs['agent_teams'][agent_idx][0], qvel)
        return 1, qvel
    
    def encode_teammate_qpos(self, no_of_agents, agent_idx, all_obs):
        """
            Args:
                no_of_agents: the number of agents in total
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
            Returns:
                Mask, 1 if observed else 0
                qpos of teammate
        """
        for i in range(no_of_agents):
            if i != agent_idx and all_obs['agent_teams'][i] == all_obs['agent_teams'][agent_idx]:
                qpos = np.expand_dims(np.array(all_obs['observation_self'][i:i+1, 0:3], dtype=np.float32), axis=0)
                qpos = self.coord_transform(all_obs['agent_teams'][agent_idx][0], qpos)
                return 1, qpos
        return 0, np.zeros([1, 1, 3])
    
    def encode_teammate_qvel(self, no_of_agents, agent_idx, all_obs):
        """
            Args:
                no_of_agents: the number of agents in total
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
            Returns:
                Mask, 1 if observed else 0
                qvel of teammate
        """
        for i in range(no_of_agents):
            if i != agent_idx and all_obs['agent_teams'][i] == all_obs['agent_teams'][agent_idx]:
                qvel = np.expand_dims(np.array(all_obs['observation_self'][i:i+1, 3:6], dtype=np.float32), axis=0)
                qvel = self.vel_transform(all_obs['agent_teams'][agent_idx][0], qvel)
                return 1, qvel
        return 0, np.zeros([1, 1, 3])

    def encode_opponent_qpos(self, no_of_agents, agent_idx, all_obs, opponent_no):
        """
            Args:
                no_of_agents: the number of agents in total
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
                opponent_no: the no. of opponent
            Returns:
                Mask, 1 if observed else 0
                qpos of opponent
        """
        curr_opponent_no = 0
        for i in range(no_of_agents):
            if all_obs['agent_teams'][i] != all_obs['agent_teams'][agent_idx]:
                curr_opponent_no += 1
                if curr_opponent_no == opponent_no:
                    qpos = np.expand_dims(np.array(all_obs['observation_self'][i:i+1, 0:3], dtype=np.float32), axis=0)
                    qpos = self.coord_transform(all_obs['agent_teams'][agent_idx][0], qpos)
                    return 1, qpos
        return 0, np.zeros([1, 1, 3])
    
    def encode_opponent_qvel(self, no_of_agents, agent_idx, all_obs, opponent_no):
        """
            Args:
                no_of_agents: the number of agents in total
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
                opponent_no: the no. of opponent
            Returns:
                Mask, 1 if observed else 0
                qvel of teammate
        """
        curr_opponent_no = 0
        for i in range(no_of_agents):
            if all_obs['agent_teams'][i] != all_obs['agent_teams'][agent_idx]:
                curr_opponent_no += 1
                if curr_opponent_no == opponent_no:
                    qvel = np.expand_dims(np.array(all_obs['observation_self'][i:i+1, 3:6], dtype=np.float32), axis=0)
                    qvel = self.vel_transform(all_obs['agent_teams'][agent_idx][0], qvel)
                    return 1, qvel
        return 0, np.zeros([1, 1, 3])

    def encode_local_qvel(self, agent_idx, all_obs):
        """
            Args:
                agent_idx: the id of the agent getting the obs
                all_obs: all observations
            Returns:
                Mask, 1 if observed else 0
                local qvel of agent
        """
        local_qvel = np.expand_dims(np.array(all_obs['agent_local_qvel'][agent_idx:agent_idx+1], dtype=np.float32), axis=0)
        return 1, local_qvel

    def encode_zones(self, agent_idx, all_obs):
        """
        Encodes a buff/debuff zone with one-hot encoding.
        As there are 4 types of buff,
            1. Restoration Zone
            2. Projectile Supplier Zone
            3. No Shooting Zone
            4. No Moving Zone
        the buffs would be encoded into a size 4 vector, where "1" at
        index "i" means buff "i + 1" is active. "0" otherwise.
        Args:
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 6, 4]
        """
        if all_obs['agent_teams'][agent_idx][0] == 'blue':
            f1 = np.array([[all_obs['F1'][:, 0]]])
            f2 = np.array([[all_obs['F2'][:, 0]]])
            f3 = np.array([[all_obs['F3'][:, 0]]])
            f4 = np.array([[all_obs['F4'][:, 0]]])
            f5 = np.array([[all_obs['F5'][:, 0]]])
            f6 = np.array([[all_obs['F6'][:, 0]]])
        else:
            f1 = np.array([[all_obs['F4'][:, 0]]])
            f2 = np.array([[all_obs['F5'][:, 0]]])
            f3 = np.array([[all_obs['F6'][:, 0]]])
            f4 = np.array([[all_obs['F1'][:, 0]]])
            f5 = np.array([[all_obs['F2'][:, 0]]])
            f6 = np.array([[all_obs['F3'][:, 0]]])
        return 1, [f1, f2, f3, f4, f5, f6]

    def encode_robot_hp(self, agent_idx, all_obs):
        """
        Encodes the robot's health points as float in range [0, 1].
        Args:
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        health = np.expand_dims(np.array([[all_obs['agents_health'][agent_idx, 0]]], dtype=np.float32), axis=1) / 2400.0
        return 1, health

    def encode_teammate_hp(self, no_of_agents, agent_idx, all_obs):
        """
        Encodes the robot's health points as float in range [0, 1].
        Args:
            no_of_agents: the number of agents in total
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        for i in range(no_of_agents):
            if i != agent_idx and all_obs['agent_teams'][i] == all_obs['agent_teams'][agent_idx]:
                health = np.expand_dims(np.array([[all_obs['agents_health'][i, 0]]], dtype=np.float32), axis=1) / 2400.0
                return 1, health
        return 0, np.zeros([1, 1, 1])

    def encode_opponent_hp(self, no_of_agents, agent_idx, all_obs, opponent_no):
        """
        Encodes the robot's health points as float in range [0, 1].
        Args:
            no_of_agents: the number of agents in total
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
            opponent_no: the no. of opponent
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        curr_opponent_no = 0
        for i in range(no_of_agents):
            if all_obs['agent_teams'][i] != all_obs['agent_teams'][agent_idx]:
                curr_opponent_no += 1
                if curr_opponent_no == opponent_no:
                    health = np.expand_dims(np.array([[all_obs['agents_health'][i, 0]]], dtype=np.float32), axis=1) / 2400.0
                    return 1, health
        return 0, np.zeros([1, 1, 1])

    def encode_launchable_projectiles(self, agent_idx, all_obs):
        """
        Encodes the the number of launchable projectiles as float in range [0, 1].
        Args:
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        nprojs = np.expand_dims(np.array([[all_obs['nprojectiles'][agent_idx, 0]]], dtype=np.float32), axis=1) / 250.0
        return 1, nprojs

    def encode_teammate_projectiles(self, no_of_agents, agent_idx, all_obs):
        """
        Encodes the the number of launchable projectiles as float in range [0, 1].
        Args:
            no_of_agents: the number of agents in total
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        for i in range(no_of_agents):
            if i != agent_idx and all_obs['agent_teams'][i] == all_obs['agent_teams'][agent_idx]:
                nprojs = np.expand_dims(np.array([[all_obs['nprojectiles'][i, 0]]], dtype=np.float32), axis=1) / 250.0
                return 1, nprojs
        return 0, np.zeros([1, 1, 1])
    
    def encode_opponent_projectiles(self, no_of_agents, agent_idx, all_obs, opponent_no):
        """
        Encodes the the number of launchable projectiles as float in range [0, 1].
        Args:
            no_of_agents: the number of agents in total
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
            opponent_no: the no. of opponent
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        curr_opponent_no = 0
        for i in range(no_of_agents):
            if all_obs['agent_teams'][i] != all_obs['agent_teams'][agent_idx]:
                curr_opponent_no += 1
                if curr_opponent_no == opponent_no:
                    nprojs = np.expand_dims(np.array([[all_obs['nprojectiles'][i, 0]]], dtype=np.float32), axis=1) / 250.0
                    return 1, nprojs
        return 0, np.zeros([1, 1, 1])

    def encode_hp_deduction(self, agent_idx, all_obs):
        """
        Encodes the the amount of hp deducted as float in range [0, 1].
        Args:
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 1]
        """
        hpdeduct = np.expand_dims(np.array([[all_obs['colli_dmg'][agent_idx, 0], float(all_obs['proj_dmg'][agent_idx, 0])]], dtype=np.float32), axis=0)
        return 1, hpdeduct

    def encode_hp_deduct_reason(self, agent_idx, all_obs):
        """
        Encodes the the reason for hp deduction
        There are 2 possible reasons we consider:
            1. Getting shot at (projectile hits)
            2. Hitting an obstacle / robot
        the reason would be encoded into a size 2 vector, where "1" at
        index "i" means reason "i + 1" is the cause. "0" otherwise.
        Args:
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 2]
        """
        hpdres = np.expand_dims(np.array([[all_obs['colli_dmg'][agent_idx, 0] > 0, float(all_obs['proj_dmg'][agent_idx, 0]) > 0]], dtype=np.float32), axis=0)
        return 1, hpdres

    def encode_attacked_armors(self, agent_idx, all_obs):
        """
        Encodes the the no. of armor being attacked.
        There are in total 4 armors:
            1. Front
            2. Rear
            3. Right
            4. Left
        the attacked armors would be encoded into a size 4 vector, where "1" at
        index "i" means armor "i + 1" is being attacked. "0" otherwise.
        Args:
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 4]
        """
        attkd_armors = np.array([[[0, 0, 0, 0]]], dtype=np.float32)
        attkd_armors[0, 0, 0] = (all_obs['proj_dmg'][agent_idx, 1]  == 'f')
        attkd_armors[0, 0, 1] = (all_obs['proj_dmg'][agent_idx, 1]  == 'b')
        attkd_armors[0, 0, 2] = (all_obs['proj_dmg'][agent_idx, 1]  == 'r')
        attkd_armors[0, 0, 3] = (all_obs['proj_dmg'][agent_idx, 1]  == 'l')
        return 1, attkd_armors

    def encode_teammates_armors(self, no_of_agents, agent_idx, all_obs):
        """
        Encodes the the no. of armor being attacked.
        There are in total 4 armors:
            1. Front
            2. Rear
            3. Right
            4. Left
        the attacked armors would be encoded into a size 4 vector, where "1" at
        index "i" means armor "i + 1" is being attacked. "0" otherwise.
        Args:
            no_of_agents: the number of agents in total
            agent_idx: the id of the agent getting the obs
            all_obs: all observations
        Returns:
            Mask, 1 if observed else 0
            One hot encoding with shape [batch_size, 1, 4]
        """
        for i in range(no_of_agents):
            if i != agent_idx and all_obs['agent_teams'][i] == all_obs['agent_teams'][agent_idx]:
                attkd_armors = np.array([[[0, 0, 0, 0]]], dtype=np.float32)
                attkd_armors[0, 0, 0] = (all_obs['proj_dmg'][i, 1]  == 'f')
                attkd_armors[0, 0, 1] = (all_obs['proj_dmg'][i, 1]  == 'b')
                attkd_armors[0, 0, 2] = (all_obs['proj_dmg'][i, 1]  == 'r')
                attkd_armors[0, 0, 3] = (all_obs['proj_dmg'][i, 1]  == 'l')
                return 1, attkd_armors
        return 0, np.zeros([1, 1, 4])
    