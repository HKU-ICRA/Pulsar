import sys
import logging
import time
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from architecture.entity_encoder.entity_formatter import Entity_formatter
from environment.envs.icra import make_env
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


def icra_main():
    """
    core_dir = "./environment"
    envs_dir = './environment/envs',
    xmls_dir = './environment/assets/xmls',

    examine_env("./environment/envs/icra.py", {},
                core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                env_viewer=EnvViewer)
    """
    n_agents = 4
    env = EnvHandler(make_env(n_agents=n_agents))
    act = {'action_movement': np.array([ [1.0, 1.0, 1.0] for _ in range(n_agents)]),
           'opponent': np.array([ [1] for _ in range(n_agents)]),
           'armor': np.array([ [0] for _ in range(n_agents)])}
    obs = env.reset()
    entity_formatter = Entity_formatter()
    done = False
    #entities = entity_formatter.concat_encoded_entity_obs(n_agents, 0, obs)
    start_time = time.time()
    for _ in range(104):
        obs, rew, done, info = env.step(act)
    #while not done:
    #    obs, rew, done, info = env.step(act)
        #if done:
        #    print(obs)
        #    print(info['true_rew'])
        #entities = entity_formatter.concat_encoded_entity_obs(n_agents, 0, obs, env.t)
        #print(obs)
        #env.render(mode='human')
        #print(rew)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    icra_main()
    #main()
    #env_test()
