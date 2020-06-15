import os, sys
import numpy as np
import logging
from copy import deepcopy

from mujoco_worldgen import Floor
from mujoco_worldgen import ObjFromXML
from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix

from environment.worldgen.battlefield import Battlefield
from environment.worldgen.builder import WorldBuilder
from environment.worldgen.core import WorldParams
from environment.worldgen.env import Env
from environment.module.agents import Agents
from environment.wrappers.util_w import DiscardMujocoExceptionEpisodes, DiscretizeActionWrapper, AddConstantObservationsWrapper, ConcatenateObsWrapper
from environment.wrappers.lidar import Lidar
from environment.wrappers.multi_agent import SplitMultiAgentActions, SplitObservations, SelectKeysWrapper
from environment.wrappers.line_of_sight import AgentAgentObsMask2D
from environment.wrappers.buff import BuffWrapper
from environment.wrappers.collision import CollisionWrapper
from environment.wrappers.health import HealthWrapper
from environment.wrappers.prep import PrepWrapper
from environment.wrappers.projectile import ProjectileWrapper
from environment.wrappers.outcome import OutcomeWrapper
from environment.wrappers.no_enter_zone import NoEnterZoneWrapper
from environment.objects.lidarsites import LidarSites


class IcraBase(Env):
    '''
        Icra base environment.
        Args:
            horizon (int): Number of steps agent gets to act
            n_substeps (int): Number of mujoco simulation steps per outer environment time-step
            n_agents (int): number of agents in the environment
            mjco_ts (float): seconds for one mujoco simulation step
            action_lims (float tuple): lower and upper limit of mujoco actions
            deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.
            meshdir (string): directory for meshes
            texturedir (string): directory for textures
            set_action (function): function for setting actions
            env_no (int): number for environment file
    '''
    def __init__(self, horizon=250, n_substeps=3, n_agents=2, mjco_ts=0.002,
                 action_lims=(-200.0, 200.0), deterministic_mode=False,
                 meshdir="assets/stls", texturedir="assets/texture",
                 set_action=None,
                 env_no=1, **kwargs):
        super().__init__(get_sim=self._get_sim,
                         get_obs=self._get_obs,
                         action_space=tuple(action_lims),
                         horizon=horizon,
                         set_action=set_action,
                         deterministic_mode=deterministic_mode)
        self.env_no = env_no
        self.mjco_ts = mjco_ts
        self.n_agents = n_agents
        self.metadata['n_actors'] = n_agents
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.kwargs = kwargs
        self.modules = []
        self.meshdir = meshdir
        self.texturedir = texturedir
        self.placement_size = (8080, 4480)
    
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
        world_params = WorldParams(size=(self.placement_size[0], self.placement_size[1], 100),
                                   num_substeps=self.n_substeps)
        successful_placement = False
        failures = 0
        while not successful_placement:
            if (failures + 1) % 10 == 0:
                logging.warning(f"Failed {failures} times in creating environment")
            builder = WorldBuilder(world_params, self.meshdir, self.texturedir, seed, env_no=self.env_no)
            battlefield = Battlefield()
            builder.append(battlefield)
            self.placement_grid = np.zeros((self.placement_size[0], self.placement_size[1]))
            successful_placement = np.all([module.build_world_step(self, battlefield, self.placement_size)
                                           for module in self.modules])
            failures += 1
        sim = builder.get_sim()
        for module in self.modules:
            module.modify_sim_step(self, sim)
        return sim

    def get_ts(self):
        return self.t
    
    def get_horizon(self):
        return self.horizon
    
    def secs_to_steps(self, secs):
        return int(secs / (self.mjco_ts * self.n_substeps))


def make_env(deterministic_mode=False, n_agents=4, env_no=1, add_bullets_visual=False):
    '''
    Response time = 0.02 seconds
    Game time = 180 seconds
    Decisions = 180 / 0.02 = 9000
    Total steps = 9000
    Seconds per simulated step =  0.002 seconds
    Seconds for each run = 9000 * 0.002 = 18 seconds
    '''
    mjco_ts = 0.002
    n_substeps = 1
    horizon = 90000
    # Setup action functions
    motor_trans_max, motor_forw_max, motor_z_max = 2000.0, 3000.0, 47123.9
    action_scale = (motor_trans_max, motor_forw_max, motor_z_max)
    action_lims = (-1.0, 1.0)
    def icra_ctrl_set_action(sim, action):
        """
        For velocity actuators it copies the action into mujoco ctrl field.
        """
        if sim.model.nmocap > 0:
            _, action = np.split(action, (sim.model.nmocap * 7, ))
        if sim.data.ctrl is not None:
            for a_idx in range(n_agents):
                for as_idx in range(3):
                    sim.data.ctrl[a_idx*3 + as_idx] = action[a_idx*3 + as_idx] * action_scale[as_idx]
    # Create base environment for battlefield
    env = IcraBase(n_agents=n_agents,
                   n_substeps=n_substeps,
                   horizon=horizon,
                   mjco_ts=mjco_ts,
                   action_lims=action_lims,
                   deterministic_mode=deterministic_mode,
                   env_no=env_no,
                   set_action=icra_ctrl_set_action,
                   meshdir=os.path.join(os.getcwd(), "environment", "assets", "stls"),
                   texturedir=os.path.join(os.getcwd(), "environment", "assets", "textures"))
    # Add bullets just for visualization
    nbullets = 25
    env.add_module(Agents(n_agents, action_scale=action_scale, add_bullets_visual=add_bullets_visual, nbullets=nbullets))        
    env.reset()
    # PrepWrapper must always be on-top
    env = PrepWrapper(env)
    env = BuffWrapper(env)
    env = CollisionWrapper(env)
    env = ProjectileWrapper(env, add_bullets_visual, nbullets)
    env = NoEnterZoneWrapper(env)
    # OutcomeWrapper must always be lowest, after HealthWrapper
    env = HealthWrapper(env)
    env = OutcomeWrapper(env)

    keys_self = ['agent_qpos_qvel']
    global_obs = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'Agent:buff', 'colli_dmg',
                     'proj_dmg', 'nprojectiles', 'agents_health', 'agent_teams',
                     'agent_local_qvel']
    keys_external = deepcopy(global_obs)
    keys_copy = deepcopy(global_obs)
    
    keys_mask_self = []
    keys_mask_external = []

    env = SplitMultiAgentActions(env)
    #env = DiscretizeActionWrapper(env, 'action_movement')
    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy)
    env = DiscardMujocoExceptionEpisodes(env)
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)
    return env
