import numpy as np
from copy import deepcopy


class Noise:

    def __init__(self):
        self.my_qpos_std = 0.0001
        self.my_qvel_std = 0.0001
        self.teammate_qpos_std = 0.0001
        self.opponent1_qpos_std = 0.0001
        self.opponent2_qpos_std = 0.0001
    
    def add_noise(self, entity_masks, entities):
        new_entities = deepcopy(entities)
        if entity_masks["my_qpos"]:
            new_entities["my_qpos"] = np.random.normal(entities["my_qpos"], self.my_qpos_std)
        if entity_masks["my_qvel"]:
            new_entities["my_qvel"] = np.random.normal(entities["my_qvel"], self.my_qvel_std)
        if entity_masks["teammate_qpos"]:
            new_entities["teammate_qpos"] = np.random.normal(entities["teammate_qpos"], self.teammate_qpos_std)
        if entity_masks["opponent1_qpos"]:
            new_entities["opponent1_qpos"] = np.random.normal(entities["opponent1_qpos"], self.opponent1_qpos_std)
        if entity_masks["opponent2_qpos"]:
            new_entities["opponent2_qpos"] = np.random.normal(entities["opponent2_qpos"], self.opponent2_qpos_std)
        return new_entities
