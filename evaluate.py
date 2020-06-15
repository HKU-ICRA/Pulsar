import os
import sys
import time
import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime
from copy import deepcopy

from environment.envs.icra import make_env
from environment.envhandler import EnvHandler
from environment.viewer.monitor import Monitor
from architecture.pulsar import Pulsar
from sim2real.time_warper import TimeWarper


def load_main_player(mp_file):
    if os.path.isfile(mp_file):
        with open(mp_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise Exception('Main player file does not exist')


def manual_evaluate():
    # Build network architecture
    n_agents = 4
    pulsars = [Pulsar(1, 1, training=False) for _ in range(n_agents)]
    for pulsar in pulsars:
        pulsar.call_build()
    main_player_file = os.path.join(".", "data", "main_player")
    if os.path.exists(main_player_file):
        print("Restoring network")
        main_player = load_main_player(main_player_file)
        agent = main_player.get_agent()
        for pulsar in pulsars:
            pulsar.set_all_weights(main_player.get_agent().get_weights())
        del(main_player)
    # Setup environment and vid monitor
    video_dir = os.path.join(os.getcwd(), "data", "vids", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #eval_env = Monitor(make_env(env_no=100), video_dir, video_callable=lambda episode_id:True, force=True)
    eval_env = make_env(env_no=100, add_bullets_visual=True)
    eval_env = EnvHandler(eval_env)
    nsteps = 104
    eval_env = TimeWarper(eval_env, pulsars, None, n_agents, eval_env.mjco_ts, eval_env.n_substeps, nsteps)
    eval_env.reset_env()
    eval_env.reset()
    eval_env.set_agent(agent)
    #time_start = time.time()
    #eval_env.collect()
    #print(time.time() - time_start)
    #return
    # Visualize either the agent's POV or real-time POV
    nsteps = 1000000
    RT_POV = True
    # Load main agent
    for _ in range(nsteps):
        eval_env.step(RT_POV)
        if not RT_POV:
            eval_env.env.render(mode="human")


manual_evaluate()
