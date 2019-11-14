import sys
import logging
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from environment.envs.base import make_env
from environment.envhandler import EnvHandler
from environment.viewer.env_viewer import EnvViewer
from mujoco_worldgen.util.envs import examine_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments


logger = logging.getLogger(__name__)

def env_test():
    env = EnvHandler(make_env())
    obs = env.reset()
    print(obs)

def main():

    core_dir = "./environment"
    envs_dir = './environment/envs',
    xmls_dir = './environment/assets/xmls',

    examine_env("./environment/envs/base.py", {},
                core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                env_viewer=EnvViewer)

if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    #main()
    env_test()
