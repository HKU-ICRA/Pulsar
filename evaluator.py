import os
import sys
from copy import deepcopy
import numpy as np
import time, datetime
from mpi4py import MPI

from environment.viewer.monitor import Monitor

from environment.envs.base import make_env
from environment.envhandler import EnvHandler
from architecture.pulsar import Pulsar
from architecture.entity_encoder.entity_encoder import Entity_encoder


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
agent_index = int(sys.argv[1])

# Build network architecture
pulsar = Pulsar(training=False)
pulsar.call_build()
entity_encoder = Entity_encoder()
# Setup environment and vid monitor
video_dir = os.path.join(os.getcwd(), "data", "vids", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
eval_env = Monitor(make_env(env_no=100 + agent_index * rank + rank), video_dir, video_callable=lambda episode_id:True, force=True)
eval_env = EnvHandler(eval_env)
n_agents = eval_env.n_actors

while True:
    # Reset environment and states
    obs = eval_env.reset()
    states = None
    opponent_states = None
    dones = False

    # Receive main agent for eval where main agent play against main agent
    agent = comm.recv()
    pulsar.set_weights(agent.get_weights())

    while not dones:
        # Given observations, get action value and neglopacs
        # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
        agent_actions = []
        # Get actions of training agent
        entities, entity_masks = entity_encoder.concat_encoded_entity_obs(obs)
        entities = np.repeat(np.expand_dims(entities, axis=1), repeats=1, axis=1)
        scalar_features = {'match_time': np.array([[float(eval_env.t)]])}
        scalar_features['bptt_match_time'] = np.expand_dims(scalar_features['match_time'], axis=1)
        baseline = entity_encoder.get_baseline(obs)
        actions, neglogp, entropy, mean, value, states, prev_state = pulsar(scalar_features, entities, entity_masks, baseline, states)
        agent_actions.append({'action_movement': [np.array(actions['xyvel'])[0][0], np.array(actions['xyvel'])[0][1], np.array(actions['yaw'])[0][0]]})
        # Get actions of opponent agent
        entities, entity_masks = entity_encoder.concat_opp_encoded_entity_obs(obs)
        entities = np.repeat(np.expand_dims(entities, axis=1), repeats=1, axis=1)
        baseline = entity_encoder.get_opp_baseline(obs)
        opp_actions, _, _, _, _, opponent_states, _ = pulsar(scalar_features, entities, entity_masks, baseline, opponent_states)
        agent_actions.append({'action_movement': [np.array(opp_actions['xyvel'])[0][0], np.array(opp_actions['xyvel'])[0][1], np.array(opp_actions['yaw'])[0][0]]})
        # Take actions in env and look the results
        per_act = {k: [] for k in agent_actions[0].keys()}
        for z in range(n_agents):
            for k, v in agent_actions[z].items():
                per_act[k].append(v)
        for k, v in per_act.items():
            per_act[k] = np.array(v)

        obs, rewards, dones, infos = eval_env.step(per_act)
