import os
import sys
from copy import deepcopy
import numpy as np
import time
import pickle
from datetime import datetime
from mpi4py import MPI

from environment.envs.base import make_env
from environment.envhandler import EnvHandler
from architecture.pulsar import Pulsar
from architecture.entity_encoder.entity_encoder import Entity_encoder


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
agent_index = int(sys.argv[1])

# GAE hyper-parameters
lam = float(sys.argv[2])
gamma = float(sys.argv[3])
# Training parameters
training_duration = 10 # seconds
learner_bound = int(sys.argv[4])
# Build network architecture
pulsar = Pulsar(training=False)
pulsar.call_build()
opponent_pulsar = Pulsar(training=False)
opponent_pulsar.call_build()
entity_encoder = Entity_encoder()
# Setup environment
env = EnvHandler(make_env(env_no=agent_index * rank + rank))
obs = env.reset()
n_agents = env.n_actors
states = None
opponent_states = None
dones = False
# Function to store trajectories
def save_trajectory(trajectory):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    trajectory_file = os.path.join(os.getcwd(), "data", "trajectories", dt_string)
    with open(trajectory_file, 'wb') as f:
        pickle.dump(trajectory, f)

while True:
    agent = MPI.COMM_WORLD.recv()
    opponent = comm.recv()
    
    pulsar.set_weights(agent.get_weights())
    opponent_pulsar.set_weights(opponent.get_weights())

    mb_rewards = []
    mb_values = []
    mb_neglogpacs_xy = []
    mb_neglogpacs_yaw = []
    mb_dones = []
    mb_scalar_features = {'match_time': [], 'bptt_match_time': []}
    mb_entities = []
    mb_entity_masks = []
    mb_baselines = []
    mb_actions_xy = []
    mb_actions_yaw = []
    mb_states = []

    agent_rewards = 0
    opponent_rewards = 0

    actual_start_time = time.time()
    while time.time() - actual_start_time <  training_duration:
        # Reset model lstm states if done
        if dones:
            states = None
        # Given observations, get action value and neglopacs
        # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
        agent_actions = []
        # Get actions of training agent
        obs_dc = deepcopy(obs)
        entities, entity_masks = entity_encoder.concat_encoded_entity_obs(obs_dc)
        entities = np.repeat(np.expand_dims(entities, axis=1), repeats=1, axis=1)
        scalar_features = {'match_time': np.array([[float(env.t)]])}
        scalar_features['bptt_match_time'] = np.expand_dims(scalar_features['match_time'], axis=1)
        baseline = entity_encoder.get_baseline(obs_dc)
        actions, neglogp, entropy, mean, value, states, prev_state = pulsar(scalar_features, entities, entity_masks, baseline, states)
        mb_values.append(value)
        mb_states.append(prev_state)
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
        opp_actions, _, _, _, _, opponent_states, _ = opponent_pulsar(scalar_features, entities, entity_masks, baseline, opponent_states)
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
        
    #batch of steps to batch of rollouts
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32) #np.squeeze(np.asarray(mb_rewards[i], dtype=np.float32), -1)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_values = np.squeeze(np.squeeze(mb_values, -1), -1)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = 0 #np.squeeze(deepcopy(model.value(sa_obs)), -1)
    # discount/bootstrap off value fn
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    # perform GAE calculation
    nsteps = mb_rewards.shape[0]
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextnonterminal = 1.0 - dones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_dones[t+1]
            nextvalues = mb_values[t+1]
        delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
        mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    mb_returns = mb_advs + mb_values

    for k, v in mb_scalar_features.items():
        mb_scalar_features[k] = np.asarray(mb_scalar_features[k])

    # Send match outcome to coordinator
    if agent_rewards > opponent_rewards:
        match_outcome = "win"
    elif agent_rewards == opponent_rewards:
        match_outcome = "draw"
    else:
        match_outcome = "lose"
    traj_outcome = {'outcome': match_outcome}
    comm.send(traj_outcome, dest=0)
    # Send trajectory to learner
    trajectory = {'mb_scalar_features': mb_scalar_features,
                  'mb_entities': np.asarray(mb_entities, dtype=np.float32),
                  'mb_entity_masks': np.asarray(mb_entity_masks, dtype=np.float32),
                  'mb_baselines': np.asarray(mb_baselines, dtype=np.float32),
                  'mb_actions_xy': np.asarray(mb_actions_xy, dtype=np.float32),
                  'mb_actions_yaw': np.asarray(mb_actions_yaw, dtype=np.float32),
                  'mb_returns': mb_returns,
                  'mb_dones': mb_dones,
                  'mb_values': mb_values,
                  'mb_neglogpacs_xy': np.asarray(mb_neglogpacs_xy, dtype=np.float32),
                  'mb_neglogpacs_yaw': np.asarray(mb_neglogpacs_yaw, dtype=np.float32),
                  'mb_states': np.asarray(mb_states, dtype=np.float32)}
    save_trajectory(trajectory)

    MPI.COMM_WORLD.send(trajectory, dest=learner_bound)
