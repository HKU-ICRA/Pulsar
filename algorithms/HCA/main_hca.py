import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_hca import Network


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


actors = [[0,1,2,3,4,5]]
learners = [6]

 
if rank >= 0 and rank <= 5:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
elif rank == 6:
    pass
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf


# Training parameters
batch_size = nsteps = 128
no_of_agents_per_env = 2


def actor(n_agents=2):
    # GAE hyper-parameters
    lam = 0.95
    gamma = 0.99
    # Setup environment
    env = Game(n_agents=n_agents)
    obs = env.reset()
    global_obs, global_rew = env.get_global_state_reward()
    dones = False
    # Build network architecture
    network = Network(1)
    network(np.array([obs[0]]))
    network.get_worker_v(np.array([obs[0]]))
    network.get_manager_v(np.array([global_obs]))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_worker_obs = np.zeros([nsteps, n_agents, 4 * n_agents], dtype=np.float32)
        mb_manager_obs = np.zeros([nsteps, 16], dtype=np.float32)
        mb_worker_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_manager_rewards = np.zeros([nsteps, 1], dtype=np.float32)
        mb_worker_values = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_manager_values = np.zeros([nsteps, 1], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions = []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                mb_worker_values[step, ai] = network.get_worker_v(np.array([obs_dc[ai]]))
                mb_neglogpacs[step, ai] = neglogp
                mb_dones[step, ai] = dones
                mb_actions[step, ai] = actions
                mb_worker_obs[step, ai] = obs_dc[ai]
                agent_actions.append(actions[0])
            mb_manager_values[step] = network.get_manager_v(np.array([global_obs]))
            mb_manager_obs[step] = global_obs
            agent_actions = np.array(agent_actions)
            obs, rewards, dones = env.step(agent_actions)
            global_obs, global_rew = env.get_global_state_reward()
            # Handle rewards
            mb_worker_rewards[step] = np.expand_dims(np.array(rewards), axis=-1)
            mb_manager_rewards[step] = global_rew
            if dones:
                obs = env.reset()
        # Get last worker values
        last_worker_values = []
        for ai in range(n_agents):
            obs_dc = deepcopy(obs)
            last_worker_value = network.get_worker_v(np.array([obs_dc[ai]]))
            last_worker_values.append(last_worker_value)
        # Get last manager values
        last_manager_value = network.get_manager_v(np.array([global_obs]))
        # discount/bootstrap off value fn
        mb_values = np.zeros_like(mb_worker_rewards)
        mb_worker_returns = np.zeros_like(mb_worker_rewards)
        mb_manager_returns = np.zeros_like(mb_manager_rewards)
        mb_advs = np.zeros_like(mb_worker_rewards)
        mb_training_advs = np.zeros_like(mb_worker_rewards)
        mb_manager_advs = np.zeros_like(mb_manager_rewards)
        lastgaelam, training_lastgaelam = 0, 0
        # perform GAE calculation
        for ai in range(n_agents):
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    nextnonterminal = 1.0 - dones
                    next_training_values = np.maximum(last_worker_values[ai], last_manager_value)
                    nextvalues = last_worker_values[ai]
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1, ai]
                    next_training_values = np.maximum(mb_worker_values[t+1, ai], mb_manager_values[t+1])
                    nextvalues = mb_worker_values[t+1, ai]
                # Actual
                delta = mb_worker_rewards[t, ai] + gamma * nextvalues * nextnonterminal - mb_worker_values[t, ai]
                mb_advs[t, ai] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                # Training
                max_current_value = np.maximum(mb_worker_values[t, ai], mb_manager_values[t])
                training_delta = mb_worker_rewards[t, ai] + gamma * next_training_values * nextnonterminal - max_current_value
                mb_training_advs[t, ai] = training_lastgaelam = training_delta + gamma * lam * nextnonterminal * training_lastgaelam
            mb_worker_returns[:, ai] = mb_advs[:, ai] + mb_worker_values[:, ai]
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_manager_value
            else:
                nextnonterminal = 1.0 - mb_dones[t+1, ai]
                nextvalues = mb_manager_values[t+1]
            delta = mb_manager_rewards[t] + gamma * nextvalues * nextnonterminal - mb_manager_values[t]
            mb_manager_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_manager_returns = mb_manager_advs + mb_manager_values
        # Send trajectory to learner
        mb_worker_rewards = np.squeeze(mb_worker_rewards, axis=-1)
        mb_manager_rewards = np.squeeze(mb_manager_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_worker_returns = np.squeeze(mb_worker_returns, axis=-1)
        mb_manager_returns = np.squeeze(mb_manager_returns, axis=-1)
        mb_advs = np.squeeze(mb_advs, axis=-1)
        mb_worker_values = np.squeeze(mb_worker_values, axis=-1)
        mb_manager_values = np.squeeze(mb_manager_values, axis=-1)
        mb_values = np.squeeze(mb_values, axis=-1)
        mb_training_advs = np.squeeze(mb_training_advs, axis=-1)
        trajectory = {
                    'mb_worker_obs': mb_worker_obs,
                    'mb_manager_obs': mb_manager_obs,
                    'mb_actions': mb_actions,
                    'mb_dones': mb_dones,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_worker_rewards': mb_worker_rewards,
                    'mb_manager_rewards': mb_manager_rewards,
                    'mb_advs': mb_advs,
                    'mb_worker_returns': mb_worker_returns,
                    'mb_manager_returns': mb_manager_returns,
                    'mb_worker_values': mb_worker_values,
                    'mb_manager_values': mb_manager_values,
                    'mb_values': mb_values,
                    'mb_training_advs': mb_training_advs
                    }
        comm.send(trajectory, dest=learners[agent_type])


def learner():
    # Learner hyperparameters
    ent_coef = 0.01
    vf_coef = 0.5
    q_coef = 1.0
    CLIPRANGE = 0.2
    max_grad_norm = 5
    noptepochs = 2
    bptt_ts = 2 #16
    n_actors = len(actors[0])
    nbatch = batch_size * n_actors
    nbatch_steps = nbatch // 2
    n_agents = no_of_agents_per_env
    # Build network architecture
    network = Network(nbatch_steps)
    env = Game(n_agents=n_agents)
    obs = env.reset()
    global_obs, global_rew = env.get_global_state_reward()
    network(np.array([obs[0]]))
    network.get_worker_v(np.array([obs[0]]))
    network.get_manager_v(np.array([global_obs]))
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_neglogpacs, b_rewards, b_worker_values, b_values, b_training_advs):
        # Stochastic selection
        inds = tf.range(nbatch)
        # Buffers for recording
        losses_total = []
        approxkls = []
        entropies = []
        # Start SGD
        for _ in range(noptepochs):
            inds = tf.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_steps):
                end = start + nbatch_steps
                # Gather mini-batch
                mbinds = inds[start:end]
                mb_obs = tf.gather(b_obs, mbinds)
                mb_actions = tf.gather(b_actions, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_rewards = tf.gather(b_rewards, mbinds)
                mb_worker_values = tf.gather(b_worker_values, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                advs = tf.gather(b_training_advs, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    vpred = network.get_worker_v(mb_obs)
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    # Calculate neglogpac
                    neglogpac = taken_action_neglogp
                    entropy = tf.reduce_mean(p_entropy)
                    vpredclipped = mb_worker_values + tf.clip_by_value(vpred - mb_worker_values, -CLIPRANGE, CLIPRANGE)
                    # Unclipped value
                    vf_losses1 = tf.square(vpred - mb_returns)
                    # Clipped value
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    pg_losses = -advs * ratio
                    pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
                var = network.trainable_variables
                grads = tape.gradient(loss, var)
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                grads_and_var = zip(grads, var)
                network.optimizer.apply_gradients(grads_and_var)
                losses_total.append(loss)
                approxkls.append(approxkl)
                entropies.append(entropy)
        losses_total = tf.reduce_mean(losses_total)
        approxkls = tf.reduce_mean(approxkls)
        entropies = tf.reduce_mean(entropies)
        return losses_total, approxkls, entropies
    # Manager value function optimizer
    @tf.function
    def manager_loss(b_obs, b_returns, b_values):
        # Stochastic selection
        inds = tf.range(nbatch)
        # Buffers for recording
        losses_total = []
        # Start SGD
        for _ in range(noptepochs):
            inds = tf.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_steps):
                end = start + nbatch_steps
                # Gather mini-batch
                mbinds = inds[start:end]
                mb_obs = tf.gather(b_obs, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                with tf.GradientTape() as tape:
                    vpred = network.get_manager_v(mb_obs)
                    vpredclipped = mb_values + tf.clip_by_value(vpred - mb_values, -CLIPRANGE, CLIPRANGE)
                    vf_losses1 = tf.square(vpred - mb_returns)
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    vf_loss = vf_loss * vf_coef
                var = network.trainable_variables
                grads = tape.gradient(vf_loss, var)
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                grads_and_var = zip(grads, var)
                network.optimizer.apply_gradients(grads_and_var)
                losses_total.append(vf_loss)
        losses_total = tf.reduce_mean(losses_total)
        return losses_total
    # Start learner process loop
    while True:
        trajectory = {
                    'mb_worker_obs':                            np.empty((nbatch, n_agents, n_agents * 4),     dtype=np.float32),
                    'mb_manager_obs':                            np.empty((nbatch, 16),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_worker_rewards':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_manager_rewards':                        np.empty((nbatch,),                   dtype=np.float32),
                    'mb_advs':                           np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_worker_returns':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_manager_returns':                        np.empty((nbatch,),                   dtype=np.float32),
                    'mb_worker_values':                         np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_manager_values':                         np.empty((nbatch,),                   dtype=np.float32),
                    'mb_values':                                np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_training_advs':                   np.empty((nbatch, n_agents),                   dtype=np.float32)
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        while cur_traj_size < nbatch:
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_worker_rewards'].shape[0]
            for k, v in trajectory.items():
                trajectory[k][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-cur_traj_size)]
            cur_traj_size += min(a_traj_size, nbatch-cur_traj_size)
        # Collect trajectories
        for ai in range(n_agents):
            b_worker_obs = trajectory['mb_worker_obs'][:, ai]
            b_manager_obs = trajectory['mb_manager_obs']
            b_actions = trajectory['mb_actions'][:, ai]
            b_dones = trajectory['mb_dones'][:, ai]
            b_neglogpacs = trajectory['mb_neglogpacs'][:, ai]
            b_worker_rewards = trajectory['mb_worker_rewards'][:, ai]
            b_manager_rewards = trajectory['mb_manager_rewards']
            b_worker_returns = trajectory['mb_worker_returns'][:, ai]
            b_manager_returns = trajectory['mb_manager_returns']
            b_worker_values = trajectory['mb_worker_values'][:, ai]
            b_manager_values = trajectory['mb_manager_values']
            b_values = trajectory['mb_values'][:, ai]
            b_training_advs = trajectory['mb_training_advs'][:, ai]
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss(b_worker_obs, b_actions,
                                                    b_worker_returns, b_dones,
                                                    b_neglogpacs,
                                                    b_worker_rewards, b_worker_values,
                                                    b_values, b_training_advs)
            vf_losses = manager_loss(b_manager_obs, b_manager_returns, b_manager_values)
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Send updated agent and summaries to coordinator
        print("PPO NAV:")
        print("approxkl: ", np.array(approxkls))
        print("loss: ", np.array(losses))
        print("entropy: ", np.array(entropies))
        print("Worker rewards: ", np.mean(b_worker_rewards, axis=0))
        print("Manager rewards: ", np.mean(b_manager_rewards, axis=0))
        print("")
        sys.stdout.flush()


if rank >= 0 and rank <= 5:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 6 and rank <= 6:
    learner()
