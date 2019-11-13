import os
import sys
import numpy as np
import functools

from environment.envs.base import Base
from environment.module.simple_agent import SimpleAgent
from environment.module.util import uniform_placement, center_placement, custom_placement
from environment.wrappers.simple_wrapper import SimpleWrapper
from environment.wrappers.util_w import DiscardMujocoExceptionEpisodes, DiscretizeActionWrapper, AddConstantObservationsWrapper, ConcatenateObsWrapper
from environment.wrappers.multi_agent import SplitMultiAgentActions, SplitObservations, SelectKeysWrapper
from environment.wrappers.line_of_sight import AgentAgentObsMask2D


def make_env(n_substeps=5, horizon=250, deterministic_mode=False, env_no=0):
    env = Base(n_agents=1, n_substeps=n_substeps, horizon=horizon,
               floor_size=5, grid_size=5, deterministic_mode=deterministic_mode, env_no=env_no,
               action_lims=(-250.0, 250.0),
               meshdir=os.path.join(os.getcwd(), 'environment', 'assets', 'stls'),
               texturedir=os.path.join(os.getcwd(), 'environment', 'assets', 'texture'))
    
    # Add Walls
    #env.add_module(RandomWalls(grid_size=5, num_rooms=2, min_room_size=5, door_size=5, low_outside_walls=True, outside_wall_rgba="1 1 1 0.1"))

    # Add Agents
    first_agent_placement = functools.partial(custom_placement, pos=np.array([0, 1]))
    #second_agent_placement = uniform_placement#functools.partial(custom_placement, pos=np.array([0, 0]))
    agent_placement_fn = [first_agent_placement]#, second_agent_placement]
    env.add_module(SimpleAgent(1, placement_fn=agent_placement_fn))
    
    env.reset()

    keys_self = ['agent_qpos_qvel']
    keys_mask_self = []#['mask_aa_obs']
    keys_external = []#['agent_qpos_qvel']
    keys_mask_external = []
    keys_copy = []

    env = AddConstantObservationsWrapper(env, new_obs={'target_pos': np.full((1, 2), (5.0, 5.0))})
    keys_self += ['target_pos']
    env = SimpleWrapper(env)

    env = SplitMultiAgentActions(env)
    #env = DiscretizeActionWrapper(env, 'action_movement', nbuckets=21)
    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy)
    env = DiscardMujocoExceptionEpisodes(env)

    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)
    
    return env
