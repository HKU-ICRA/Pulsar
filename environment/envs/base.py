import os, sys
import numpy as np
import logging

from mujoco_worldgen import Floor

from environment.worldgen.builder import WorldBuilder
from environment.worldgen.core import WorldParams
from environment.worldgen.env import Env
from environment.module.walls import RandomWalls
from environment.module.agents import Agents, agent_set_action, ctrl_set_action_gimbalVer
from environment.module.util import uniform_placement, center_placement, custom_placement
from environment.wrappers.util_w import DiscardMujocoExceptionEpisodes, DiscretizeActionWrapper, AddConstantObservationsWrapper, ConcatenateObsWrapper
from environment.wrappers.lidar import Lidar
from environment.wrappers.multi_agent import SplitMultiAgentActions, SplitObservations, SelectKeysWrapper
from environment.wrappers.line_of_sight import AgentAgentObsMask2D
#from environment.wrappers.bullets import Bullets
from environment.wrappers.health import HealthWrapper
from environment.wrappers.projectile import ProjectileWrapper
from environment.objects.lidarsites import LidarSites


class Base(Env):
    '''
        Base environment.
        Args:
            horizon (int): Number of steps agent gets to act
            n_substeps (int): Number of mujoco simulation steps per outer environment time-step
            n_agents (int): number of agents in the environment
            floor_size (float): size of the floor
            grid_size (int): size of the grid that we'll use to place objects on the floor
            action_lims (float tuple): lower and upper limit of mujoco actions
            deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.
    '''
    def __init__(self, horizon=250, n_substeps=5, n_agents=2, floor_size=6.,
                 grid_size=30, action_lims=(-200.0, 200.0), deterministic_mode=False, meshdir="assets/stls", texturedir="assets/texture",
                 env_no=1, **kwargs):
        super().__init__(get_sim=self._get_sim,
                         get_obs=self._get_obs,
                         action_space=tuple(action_lims),
                         horizon=horizon,
                         #set_action=ctrl_set_action_gimbalVer,
                         deterministic_mode=deterministic_mode)
        self.env_no = env_no
        self.n_agents = n_agents
        self.metadata['n_actors'] = n_agents
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.floor_size = floor_size
        self.grid_size = grid_size
        self.kwargs = kwargs
        self.placement_grid = np.zeros((grid_size, grid_size))
        self.modules = []
        self.meshdir = meshdir
        self.texturedir = texturedir
    

    def add_module(self, module):
        self.modules.append(module)


    def _get_obs(self, sim):
        '''
            Loops through modules, calls their observation_step functions, and
                adds the result to the observation dictionary.
        '''
        obs = {}
        for module in self.modules:
            obs.update(module.observation_step(self, self.sim))
        return obs


    def _get_sim(self, seed):
        '''
            Calls build_world_step and then modify_sim_step for each module. If
                a build_world_step failed, then restarts.
        '''
        world_params = WorldParams(size=(self.floor_size, self.floor_size, 2.5),
                                   num_substeps=self.n_substeps)
        successful_placement = False
        failures = 0
        while not successful_placement:
            if (failures + 1) % 10 == 0:
                logging.warning(f"Failed {failures} times in creating environment")
            builder = WorldBuilder(world_params, self.meshdir, self.texturedir, seed, env_no=self.env_no)
            floor = Floor()

            builder.append(floor)

            self.placement_grid = np.zeros((self.grid_size, self.grid_size))

            successful_placement = np.all([module.build_world_step(self, floor, self.floor_size)
                                           for module in self.modules])
            failures += 1

        sim = builder.get_sim()

        for module in self.modules:
            module.modify_sim_step(self, sim)

        return sim


def make_env(n_substeps=3, horizon=250, deterministic_mode=False, n_agents=1, env_no=1):
    
    env = Base(n_agents=n_agents, n_substeps=n_substeps, horizon=horizon,
               floor_size=10, grid_size=50, deterministic_mode=deterministic_mode, env_no=env_no)
    
    # Add Walls
    #env.add_module(RandomWalls(grid_size=5, num_rooms=2, min_room_size=5, door_size=5, low_outside_walls=True, outside_wall_rgba="1 1 1 0.1"))

    # Add Agents
    first_agent_placement = uniform_placement
    second_agent_placement = uniform_placement
    agent_placement_fn = [first_agent_placement] + [second_agent_placement]
    env.add_module(Agents(n_agents, placement_fn=agent_placement_fn))
    
    # Add LidarSites
    n_lidar_per_agent = 1
    visualize_lidar = False
    compress_lidar_scale = None
    if visualize_lidar:
        env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))

    env.reset()

    keys_self = ['agent_qpos_qvel']
    keys_mask_self = []#['mask_aa_obs']
    keys_external = []#['agent_qpos_qvel']
    keys_mask_external = []
    keys_copy = []

    env = AddConstantObservationsWrapper(env, new_obs={'agents_health': np.full((n_agents, 1), 100.0)})
    keys_self += ['agents_health']
    env = ProjectileWrapper(env)
    #env = HealthWrapper(env)

    env = SplitMultiAgentActions(env)
    env = DiscretizeActionWrapper(env, 'action_movement')
    env = AgentAgentObsMask2D(env)

    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']

    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy)
    env = DiscardMujocoExceptionEpisodes(env)

    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)
    
    return env
