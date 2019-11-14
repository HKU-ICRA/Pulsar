import numpy as np


class Entity_encoder():

    def __init__(self):
        pass
    
    def concat_encoded_entity_obs(self, all_obs):
        """
        Returns all encoded entities as a concaternation.

        Args:
            all_obs: pass

        Returns:
            One hot encoding with shape [batch_size, n_entities, feature_size]
        """
        qpos = np.expand_dims(np.array(all_obs['observation_self'][0:1, 0:4], dtype=np.float32), axis=1)
        qvel = np.expand_dims(np.array(all_obs['observation_self'][0:1, 4:8], dtype=np.float32), axis=1)
        health = np.expand_dims(np.array([[all_obs['observation_self'][0, 8], 0, 0, 0]], dtype=np.float32), axis=1)
        opp_qpos = np.array(all_obs['agent_qpos_qvel'][0:1, :, 0:4], dtype=np.float32)
        opp_qvel = np.array(all_obs['agent_qpos_qvel'][0:1, :, 4:8], dtype=np.float32)

        padding = np.array([[1, 1, 1, 1, 1]], dtype=np.float32)

        return np.concatenate([qpos, qvel, health, opp_qpos, opp_qvel], axis=1), padding

    def get_baseline(self, all_obs):
        return np.concatenate([np.array(all_obs['observation_self'][0:1], dtype=np.float32), np.array(all_obs['observation_self'][1:2], dtype=np.float32)], axis=1)

    def concat_opp_encoded_entity_obs(self, all_obs):
        """
        Returns all of opponent's encoded entities as a concaternation.

        Args:
            all_obs: pass

        Returns:
            One hot encoding with shape [batch_size, n_entities, feature_size]
        """
        qpos = np.expand_dims(np.array(all_obs['observation_self'][1:2, 0:4], dtype=np.float32), axis=1)
        qvel = np.expand_dims(np.array(all_obs['observation_self'][1:2, 4:8], dtype=np.float32), axis=1)
        health = np.expand_dims(np.array([[all_obs['observation_self'][1, 8], 0, 0, 0]], dtype=np.float32), axis=1)
        opp_qpos = np.array(all_obs['agent_qpos_qvel'][1:2, :, 0:4], dtype=np.float32)
        opp_qvel = np.array(all_obs['agent_qpos_qvel'][1:2, :, 4:8], dtype=np.float32)

        padding = np.array([[1, 1, 1, 1, 1]], dtype=np.float32)

        return np.concatenate([qpos, qvel, health, opp_qpos, opp_qvel], axis=1), padding

    def get_opp_baseline(self, all_obs):
        return np.concatenate([np.array(all_obs['observation_self'][1:2], dtype=np.float32), np.array(all_obs['observation_self'][0:1], dtype=np.float32)], axis=1)

    def encode_zone(self, f):
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
            f: pass

        Returns:
            One hot encoding with shape [batch_size, 1, 4]
        """
        pass

    def encode_zone_activation(self, f):
        """
        Encodes whether the zone is active or not.
        True for index 0 means the zone has been activated.
        True for index 1 means the zone has not been activated.
        False otherwise.

        Args:
            f: pass
        
        Returns:
            One hot encoding with shape [batch_size, 1, 2]
        """
        pass

    def encode_robot_hp(self, hp):
        """
        Encodes the robot's health points as float in range [0, 1].

        Args:
            hp: hp of the robot
        
        Returns:
            One hot encoding with shape [batch_size, 1, 1]
        """
        pass

    def encode_launchable_projectiles(self, nlpg):
        """
        Encodes the the number of launchable projectiles as float in range [0, 1].

        Args:
            nlpg: number of launchable projectiles
        
        Returns:
            One hot encoding with shape [batch_size, 1, 1]
        """
        pass

    def encode_hp_deduction(self, hpdeduct):
        """
        Encodes the the amount of hp deducted as float in range [0, 1].

        Args:
            hpdeduct: the amount of hp deducted from the robot
        
        Returns:
            One hot encoding with shape [batch_size, 1, 1]
        """
        pass

    def encode_hp_deduct_reason(self, reason):
        """
        Encodes the the reason for hp deduction
        There are 2 possible reasons we consider:
            1. Getting shot at (projectile hits)
            2. Hitting an obstacle / robot
        the reason would be encoded into a size 2 vector, where "1" at
        index "i" means reason "i + 1" is the cause. "0" otherwise.

        Args:
            reason: the reason for hp deduction from the robot
        
        Returns:
            One hot encoding with shape [batch_size, 1, 2]
        """
        pass

    def encode_attacked_armors(self, armors):
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
            armors: the armors being attacked / not attacked.
        
        Returns:
            One hot encoding with shape [batch_size, 1, 4]
        """
        pass

    def encode_projectiles_launched(self, lprojs):
        """
        Encodes the the number of projectiles launched.
       
        Args:
            lprojs: the number of projectiles launched
        
        Returns:
            One hot encoding with shape [batch_size, 1, 1]
        """
        pass
