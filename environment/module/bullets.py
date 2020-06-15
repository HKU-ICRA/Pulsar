import os
import sys
import numpy as np

from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs)
from mujoco_worldgen.util.rotation import normalize_angles
from mujoco_worldgen import ObjFromXML
from mujoco_worldgen.util.path import worldgen_path
from environment.worldgen.transforms import set_geom_attr_transform, add_weld_equality_constraint_transform, set_joint_damping_transform
from environment.module.module import EnvModule
from environment.module.util import rejection_placement, get_size_from_xml


class BulletObjFromXML(ObjFromXML):
    '''
        Path to Bullet's XML.
    '''
    def _get_xml_dir_path(self, *args):
        '''
        If you want to use custom XMLs, subclass this class and overwrite this
        method to return the path to your 'xmls' folder
        '''
        return os.path.join(os.getcwd(), "environment", "assets", "xmls", args[0])

class Bullets(EnvModule):
    '''
        Add bullets to the environment.
        Args:
            n_bullets (int): number of bullets
    '''
    @store_args
    def __init__(self, n_bullets):
        pass

    def build_world_step(self, env, floor, floor_size):
        env.metadata['n_bullets'] = self.n_bullets
        
        for i in range(self.n_bullets):
            obj = BulletObjFromXML("bullet", name=f"bullet{i}")
            floor.append(obj)

        return True
