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
from environment.module.util import uniform_placement, rejection_placement, get_size_from_xml, clip_angle_range


class AgentObjFromXML(ObjFromXML):
    '''
        Path to Agent's XML.
    '''
    def _get_xml_dir_path(self, *args):
        '''
        If you want to use custom XMLs, subclass this class and overwrite this
        method to return the path to your 'xmls' folder
        '''
        return os.path.join(os.getcwd(), "environment", "assets", "xmls", args[0])


class Agents(EnvModule):
    '''
        Add Agents to the environment.
    '''
    @store_args
    def __init__(self, n_agents, action_scale, add_bullets_visual=False, nbullets=0, polar_obs=True):
        self.placement_fn = [uniform_placement for _ in range(n_agents)]
        self.placement_size = (8080, 4480)
        self.action_scale = action_scale

    def build_world_step(self, env, floor, floor_size):
        # Add Agents. First agent must always be one of the main agents
        teams = ['red', 'blue']
        agent_infos = [dict() for _ in range(self.n_agents)]
        main_agent_team = np.random.randint(0, 2)
        agent_infos[0]['team'] = teams[main_agent_team]
        agent_infos[1]['team'] = teams[main_agent_team]
        agent_infos[2]['team'] = teams[1 - main_agent_team]
        if self.n_agents == 4:
            agent_infos[3]['team'] = teams[1 - main_agent_team]
        # Set metadata
        env.metadata['n_agents'] = self.n_agents
        self.agent_infos = agent_infos
        env.metadata['agent_infos'] = agent_infos
        successful_placement = True
        for i in range(self.n_agents):
            obj = AgentObjFromXML(agent_infos[i]['team'] + "_infantry", name=f"agent{i}")
            if self.add_bullets_visual:
                for bidx in range(self.nbullets):
                    if agent_infos[i]['team'] == 'red':
                        obj.mark(f"agent{i}:bullet{bidx}", absolute_xyz=(-100, -100, -100), rgba=(1, 0, 0, 1), size=np.array([10, 10, 10]))
                    else:
                        obj.mark(f"agent{i}:bullet{bidx}", absolute_xyz=(-100, -100, -100), rgba=(0, 0, 1, 1), size=np.array([10, 10, 10]))
            agent_info = agent_infos[i]
            _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
            obj_size = get_size_from_xml(obj)
            pos, pos_grid = rejection_placement(env, _placement_fn, floor_size, obj_size)
            if pos is not None:
                floor.append(obj, placement_xy=pos)
                # store spawn position in metadata. This allows sampling subsequent agents
                # close to previous agents
                env.metadata[f"agent{i}_initpos"] = pos_grid
            else:
                successful_placement = False
        return successful_placement

    def modify_sim_step(self, env, sim):
        # Cache qpos, qvel idxs
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        self.agent_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'agent{i}')
                                        for i in range(self.n_agents)])
        env.metadata['agent_geom_idxs'] = [sim.model.site_name2id(f'agent{i}:agent')
                                           for i in range(self.n_agents)]
        # Randomize starting orientation of agents
        for i in range(self.n_agents):
            rqpos = qpos_idxs_from_joint_prefix(sim, f'agent{i}:rz')
            sim.data.qpos[rqpos] = np.random.uniform(low=0, high=6.28319)

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()        
        # Agent
        agent_qpos = qpos[self.agent_qpos_idxs]
        agent_qvel = qvel[self.agent_qvel_idxs]
        # Normalize angles in range 0 to 6.28
        agent_angle = (clip_angle_range(agent_qpos[:, [-1]]) + 4.71239) % 6.28319
        agent_qpos[:, [-1]] = agent_angle = normalize_angles(agent_angle)
        # Normalize positions in range 0 to 1
        agent_qpos[:, [0]] = agent_qpos[:, [0]] / self.placement_size[0]
        agent_qpos[:, [1]] = agent_qpos[:, [1]] / self.placement_size[1]
        # Normalize qvel in range -1 to 1
        max_speed = np.sqrt(self.action_scale[0]**2 + self.action_scale[1]**2)
        agent_qvel[:, [0]] = agent_qvel[:, [0]] / max_speed
        agent_qvel[:, [1]] = agent_qvel[:, [1]] / max_speed
        agent_qvel[:, [3]] = agent_qvel[:, [3]] / 4.71239
        # Form observation, remove z-pos from qpos and qvel
        agent_qpos = np.array([[aqpos[0], aqpos[1], aqpos[3]] for aqpos in agent_qpos])
        agent_qvel = np.array([[aqvel[0], aqvel[1], aqvel[3]] for aqvel in agent_qvel])
        agent_qpos_qvel = np.concatenate([agent_qpos, agent_qvel], -1)
        # Velocimeter info
        velocimeter_info = []
        for i in range(self.n_agents):
            velocimeter_idx = sim.model.sensor_name2id(f"agent{i}:velocimeter")
            velocimeter_data = sim.data.sensordata[velocimeter_idx:velocimeter_idx+2]
            velocimeter_data[0] /= self.action_scale[0]
            velocimeter_data[1] /= self.action_scale[1]
            velocimeter_info.append(velocimeter_data)
        agent_local_qvel = np.array(velocimeter_info)
        obs = {
            'agent_qpos_qvel': agent_qpos_qvel,
            'agent_angle': agent_angle,
            'agent_pos': agent_qpos[:, :3],
            'agent_local_qvel': agent_local_qvel
        }
        return obs
