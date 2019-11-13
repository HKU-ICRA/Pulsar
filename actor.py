import os
import sys
from copy import deepcopy
import numpy as np
import time
from mpi4py import MPI

from environment.envs.simple import make_env
from environment.envhandler import EnvHandler


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
agent_index = int(sys.argv[1])

# GAE hyper-parameters
lam = float(sys.argv[2])
gamma = float(sys.argv[3])
# Training parameters
training_duration = 5 # seconds

env = EnvHandler(make_env(env_no=0))
n_agents = env.n_actors
nenv = 1
obs = env.reset()
states = None
dones = [False for _ in range(nenv)]

while True:
    player = MPI.COMM_WORLD.recv()
    model = player.get_model()
    if states == None:
        states = {k: np.repeat(v, nenv, axis=0) for k, v in model.initial_state.items()}
    opponent = player.get_match()

    mb_rewards = [[] for _ in range(n_agents)]
    mb_values = [[] for _ in range(n_agents)]
    mb_neglogpacs = [[] for _ in range(n_agents)]
    mb_obs = [{k: [] for k in env.observation_space.spaces.keys()} for _ in range(n_agents)]
    mb_actions = [{k: [] for k in env.action_space.spaces.keys()} for _ in range(n_agents)]
    mb_dones = [[] for _ in range(n_agents)]
    mb_states = states
    model.set_state(states)

    start_time = time.time()
    while time.time() - start_time <  training_duration:
        # Reset model lstm states if done
        for i in range(nenv): 
            if dones[i]:
                for k, v in model.get_zero_states().items():
                    states[k][i] = v
            
        # Given observations, get action value and neglopacs
        # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
        agent_actions = []

        for i in range(n_agents):
            obs_dc = deepcopy(obs)
            sa_obs = {k: v[:, i:i+1, :] for k, v in obs_dc.items()}
            actions, info = model.step(sa_obs)
            actions_dc = deepcopy(actions)
            info_dc = deepcopy(info)
            agent_actions.append(actions_dc)
            mb_values[i].append(info_dc['vpred'])
            mb_neglogpacs[i].append(info_dc['ac_neglogp'])
            states = info_dc['state']
            mb_dones[i].append(deepcopy(dones))  
            for k, v in actions_dc.items():
                mb_actions[i][k].append(v)
            for k, v in sa_obs.items():
                mb_obs[i][k].append(v)
                
        # Take actions in env and look the results
        per_act = {k: [] for k in agent_actions[0].keys()}
        for z in range(n_agents):
            for k, v in agent_actions[z].items():
                per_act[k].append(v[e])
        for k, v in per_act.items():
            per_act[k] = np.array(v)

        obs, rewards, dones, infos = env.step([per_act])

        for i in range(n_agents):
            mb_rewards[i].append(deepcopy(rewards)[:, i:i+1])
        
    #batch of steps to batch of rollouts
    last_values = [[] for _ in range(n_agents)]
    for i in range(n_agents):
        mb_rewards[i] = np.squeeze(np.asarray(mb_rewards[i], dtype=np.float32), -1)
        mb_values[i] = np.asarray(mb_values[i], dtype=np.float32)
        mb_neglogpacs[i] = np.asarray(mb_neglogpacs[i], dtype=np.float32)
        mb_dones[i] = np.asarray(mb_dones[i], dtype=np.bool)
        sa_obs = {k: v[:, i:i+1, :] for k, v in deepcopy(obs).items()}
        last_values[i] = np.squeeze(deepcopy(model.value(sa_obs)), -1)

    # discount/bootstrap off value fn
    mb_returns = [np.zeros_like(mb_rewards[i]) for i in range(n_agents)]
    mb_advs = [np.zeros_like(mb_rewards[i]) for i in range(n_agents)]

    lastgaelam = [0 for _ in range(n_agents)]
    nsteps = len(mb_obs[0])
    for i in range(n_agents):
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values[i]
            else:
                nextnonterminal = 1.0 - mb_dones[i][t+1]
                nextvalues = mb_values[i][t+1]
            delta = mb_rewards[i][t] + gamma * nextvalues * nextnonterminal - mb_values[i][t]
            mb_advs[i][t] = lastgaelam[i] = delta + gamma * lam * nextnonterminal * lastgaelam[i]
        mb_returns[i] = mb_advs[i] + mb_values[i]

    mb_obs[i], mb_actions[i] = dsf01(mb_obs[i]), dsf01(mb_actions[i])
    mb_returns[i], mb_dones[i], mb_values[i], mb_neglogpacs[i] = sf01(mb_returns[i]), sf01(mb_dones[i]), sf01(mb_values[i]), sf01(mb_neglogpacs[i])
        
    mb_obs = dconcat(np.asarray(mb_obs))
    mb_actions = dconcat(np.asarray(mb_actions))
    mb_returns = f01(np.asarray(mb_returns))
    mb_dones = f01(np.asarray(mb_dones))
    mb_values = f01(np.asarray(mb_values))
    mb_neglogpacs = f01(np.asarray(mb_neglogpacs))
        
    mb_states_final = np.asarray([{k: v[i:i+1] for k, v in deepcopy(states).items()} for i in range(nenv)])
    
    trajectory = {'mb_obs': mb_obs, 'mb_actions': mb_actions, 'mb_returns': mb_returns,
                  'mb_dones': mb_dones, 'mb_values': mb_values, 'mb_neglogpacs': mb_neglogpacs,
                  'mb_states_final': mb_states_final}

    c = MPI.COMM_WORLD.send(trajectory, dest=1)
    

'''
b = MPI.COMM_WORLD.recv()
print("Received msg:", b)
c = MPI.COMM_WORLD.send("From actor", dest=1)
'''
