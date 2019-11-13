import os
import sys
import numpy as np

from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs)
from mujoco_worldgen.util.rotation import normalize_angles
from mujoco_worldgen import ObjFromXML
from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen import Geom
from environment.worldgen.transforms import set_geom_attr_transform, add_weld_equality_constraint_transform, set_joint_damping_transform
from environment.module.module import EnvModule
from environment.module.util import rejection_placement, get_size_from_xml


class AgentObjFromXML(ObjFromXML):
    '''
        Path to Agent's XML.
    '''
    def _get_xml_dir_path(self, *args):
        '''
        If you want to use custom XMLs, subclass this class and overwrite this
        method to return the path to your 'xmls' folder
        '''
        return os.path.join(os.getcwd(), 'environment', 'assets', 'xmls', args[0])

class SimpleAgent(EnvModule):
    '''
        Add Agents to the environment.
        Args:
            n_agents (int): number of agents
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for
                spec. If list of functions, then it is assumed there is one function given
                per agent
            color (tuple or list of tuples): rgba for agent. If list of tuples, then it is
                assumed there is one color given per agent
            friction (float): agent friction
            damp_z (bool): if False, reduce z damping to 1
            polar_obs (bool): Give observations about rotation in polar coordinates
    '''
    @store_args
    def __init__(self, n_agents, placement_fn=None, color=None, friction=None,
                 damp_z=False, polar_obs=True):
        pass

    def build_world_step(self, env, floor, floor_size):
        env.metadata['n_agents'] = self.n_agents
        successful_placement = True

        for i in range(self.n_agents):
            env.metadata.pop(f"agent{i}_initpos", None)

        for i in range(self.n_agents):
            obj = AgentObjFromXML("simpleagent", name=f"agent{i}")
            if self.friction is not None:
                obj.add_transform(set_geom_attr_transform('friction', self.friction))
            if self.color is not None:
                _color = (self.color[i]
                          if isinstance(self.color[0], (list, tuple, np.ndarray))
                          else self.color)
                obj.add_transform(set_geom_attr_transform('rgba', _color))
            if not self.damp_z:
                obj.add_transform(set_joint_damping_transform(1, 'tz'))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                obj_size = get_size_from_xml(obj)
                #pos, pos_grid = rejection_placement(env, _placement_fn, floor_size, obj_size)
                pos = np.array([0.2, 0.2])
                if pos is not None:
                    floor.append(obj, placement_xy=pos)
                    # store spawn position in metadata. This allows sampling subsequent agents
                    # close to previous agents
                    #env.metadata[f"agent{i}_initpos"] = pos_grid
                else:
                    successful_placement = False
            else:
                floor.append(obj)
        return successful_placement

    def modify_sim_step(self, env, sim):
        # Cache qpos, qvel idxs
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        self.agent_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'agent{i}')
                                        for i in range(self.n_agents)])
        env.metadata['agent_geom_idxs'] = [sim.model.geom_name2id(f'agent{i}:agent')
                                           for i in range(self.n_agents)]

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        agent_qpos = qpos[self.agent_qpos_idxs][:, 0:2]
        agent_qvel = qvel[self.agent_qvel_idxs][:, 0:2]
        agent_qpos_qvel = np.concatenate([agent_qpos, agent_qvel], -1)
        obs = {'agent_qpos_qvel': agent_qpos_qvel}
   
        return obs
