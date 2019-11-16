import os
import sys
from copy import deepcopy
import numpy as np
import time
from mpi4py import MPI

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
opponent_pulsar = Pulsar(training=False)
opponent_pulsar.call_build()
entity_encoder = Entity_encoder()
# Setup environment
env = EnvHandler(make_env(env_no=100 + agent_index * rank + rank))
n_agents = env.n_actors
states = None
opponent_states = None
dones = False

while True:
    agent = MPI.COMM_WORLD.recv()
    opponent = comm.recv()
    
    pulsar.set_weights(agent.get_weights())
    opponent_pulsar.set_weights(opponent.get_weights())

    # Reset environment and states
    obs = env.reset()
    states = None
    opponent_states = None

    while not dones:
        # Given observations, get action value and neglopacs
        # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
        agent_actions = []
        # Get actions of training agent
        obs_dc = deepcopy(obs)
        entities, entity_masks = entity_encoder.concat_encoded_entity_obs(obs_dc)
        entities = np.repeat(np.expand_dims(entities, axis=1), repeats=1, axis=1)
        scalar_features = {'match_time': np.array([[time.time() - start_time]])}
        scalar_features['bptt_match_time'] = np.expand_dims(scalar_features['match_time'], axis=1)
        baseline = entity_encoder.get_baseline(obs_dc)
        actions, neglogp, entropy, mean, value, states = pulsar(scalar_features, entities, entity_masks, baseline, states)
        mb_values.append(value)
        mb_neglogpacs_xy += list(neglogp['xyvel'])
        mb_neglogpacs_yaw += list(neglogp['yaw'])
        mb_scalar_features['match_time'] += list(scalar_features['match_time'])
        mb_scalar_features['bptt_match_time'] += list(scalar_features['bptt_match_time'])
        mb_entities += list(entities)
        mb_entity_masks += list(entity_masks)
        mb_baselines += list(baseline)
        mb_actions_xy += list(actions['xyvel'])
        mb_actions_yaw += list(actions['yaw'])
        agent_actions.append({'action_movement': [np.array(actions['xyvel'])[0][0], np.array(actions['xyvel'])[0][1], np.array(actions['yaw'])[0][0]]})
        # Get actions of opponent agent
        obs_dc = deepcopy(obs)
        entities, entity_masks = entity_encoder.concat_opp_encoded_entity_obs(obs_dc)
        entities = np.repeat(np.expand_dims(entities, axis=1), repeats=1, axis=1)
        baseline = entity_encoder.get_opp_baseline(obs_dc)
        opp_actions, _, _, _, _, opponent_states = opponent_pulsar(scalar_features, entities, entity_masks, baseline, opponent_states)
        agent_actions.append({'action_movement': [np.array(opp_actions['xyvel'])[0][0], np.array(opp_actions['xyvel'])[0][1], np.array(opp_actions['yaw'])[0][0]]})
        # Take actions in env and look the results
        per_act = {k: [] for k in agent_actions[0].keys()}
        for z in range(n_agents):
            for k, v in agent_actions[z].items():
                per_act[k].append(v)
        for k, v in per_act.items():
            per_act[k] = np.array(v)

        obs, rewards, dones, infos = env.step(per_act)

        agent_rewards += rewards[0]
        opponent_rewards += rewards[1]
        mb_rewards.append(deepcopy(rewards)[0])
        mb_dones.append(dones)