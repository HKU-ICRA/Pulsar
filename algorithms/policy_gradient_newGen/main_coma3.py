import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_coma2 import Network


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
    dones = False
    # Build network architecture
    network = Network(1)
    network(np.array([obs[0]]))
    network.get_q(np.array([obs[0]]), np.array([0]))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_obs = np.zeros([nsteps, n_agents, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_q = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_qs = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_other_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_probs = np.zeros([nsteps, n_agents, 5], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions = []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                mb_neglogpacs[step, ai] = neglogp
                mb_dones[step, ai] = dones
                mb_actions[step, ai] = actions
                mb_obs[step, ai] = obs_dc[ai]
                mb_probs[step, ai] = probs
                agent_actions.append(actions[0])
            agent_actions = np.array(agent_actions)
            for ai in range(n_agents):
                other_ai = 1 - ai
                mb_other_actions[step, ai] = agent_actions[other_ai]
                mb_qs[step, ai] = network.get_q(np.array([obs_dc[ai]]), np.array([agent_actions[other_ai]]))
                mb_q[step, ai] = mb_qs[step, ai, agent_actions[ai]]
            obs, rewards, dones = env.step(agent_actions)
            # Handle rewards
            mb_rewards[step] = np.array([[rewards] for _ in range(n_agents)])
            if dones:
                obs = env.reset()
        # Get last worker values
        last_actions, last_qs, last_single_qs, last_probs = [], [], [], []
        for ai in range(n_agents):
            obs_dc = deepcopy(obs)
            actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
            last_actions.append(actions[0])
            last_probs.append(probs)
        for ai in range(n_agents):
            obs_dc = deepcopy(obs)
            other_ai = 1 - ai
            last_q = network.get_q(np.array([obs_dc[ai]]), np.array([last_actions[other_ai]]))
            last_qs.append(last_q)
            last_single_qs.append(last_q[0, last_actions[ai]])
        # discount/bootstrap off value fn
        mb_values = np.zeros_like(mb_rewards)
        mb_returns = np.zeros_like(mb_rewards)
        mb_qtargets = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        # perform GAE calculation
        for ai in range(n_agents):
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    nextnonterminal = 1.0 - dones
                    nextvalues = 0
                    for idx in range(5):
                        nextvalues += last_qs[ai][0, idx] * last_probs[ai][0, idx]
                    target_q = last_single_qs[ai]
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1, ai]
                    nextvalues = 0
                    for idx in range(5):
                        nextvalues += mb_qs[t+1, ai, idx] * mb_probs[t+1, ai, idx]
                    target_q = mb_q[t+1, ai]
                # Actual
                for idx in range(5):
                    mb_values[t, ai] += mb_qs[t, ai, idx] * mb_probs[t, ai, idx]
                delta = mb_rewards[t, ai] + gamma * nextvalues * nextnonterminal - mb_values[t, ai]
                mb_advs[t, ai] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                mb_qtargets[t, ai] = mb_rewards[t, ai] + nextnonterminal * gamma * target_q
            mb_returns[:, ai] = mb_advs[:, ai] + mb_values[:, ai]
        # Send trajectory to learner
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_qtargets = np.squeeze(mb_qtargets, axis=-1)
        mb_advs = np.squeeze(mb_advs, axis=-1)
        mb_values = np.squeeze(mb_values, axis=-1)
        mb_q = np.squeeze(mb_q, axis=-1)
        trajectory = {
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
                    'mb_other_actions': mb_other_actions,
                    'mb_dones': mb_dones,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_rewards': mb_rewards,
                    'mb_advs': mb_advs,
                    'mb_returns': mb_returns,
                    'mb_values': mb_values,
                    'mb_probs': mb_probs,
                    'mb_qtargets': mb_qtargets,
                    'mb_q': mb_q
                    }
        comm.send(trajectory, dest=learners[agent_type])


def learner():
    # Learner hyperparameters
    ent_coef = 0.01
    vf_coef = 0.5
    q_coef = 1.0
    CLIPRANGE = 0.2
    max_grad_norm = 5
    noptepochs = 4
    bptt_ts = 2 #16
    n_actors = len(actors[0])
    nbatch = batch_size * n_actors
    nbatch_steps = nbatch // 4
    n_agents = no_of_agents_per_env
    # Build network architecture
    network = Network(nbatch_steps)
    env = Game(n_agents=n_agents)
    obs = env.reset()
    network(np.array([obs[0]]))
    network.get_q(np.array([obs[0]]), np.array([0]))
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_other_actions, b_returns, b_dones, b_neglogpacs, b_rewards, b_values, b_probs, b_qtargets, b_q):
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
                mb_other_actions = tf.gather(b_other_actions, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_rewards = tf.gather(b_rewards, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                mb_probs = tf.gather(b_probs, mbinds)
                mb_qtargets = tf.gather(b_qtargets, mbinds)
                mb_qvalues = tf.gather(b_q, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    qpred = network.get_q(mb_obs, mb_other_actions)
                    #batch_idx = tf.range(mb_obs.shape[0])
                    #batch_idx = tf.expand_dims(batch_idx, axis=-1)
                    #gather_actions = tf.concat([batch_idx, tf.expand_dims(mb_actions, axis=-1)], axis=-1)
                    #qpred = tf.gather_nd(qpred, gather_actions)
                    vpred = qpred * mb_probs
                    vpred = tf.reduce_sum(vpred, axis=-1)
                    #qpredclipped = mb_qvalues + tf.clip_by_value(qpred - mb_qvalues, -CLIPRANGE, CLIPRANGE)
                    #q_losses1 = tf.square(qpred - mb_qtargets)
                    #q_losses2 = tf.square(qpredclipped - mb_qtargets)
                    #q_loss = .5 * tf.reduce_mean(tf.maximum(q_losses1, q_losses2))
                    # Batch normalize the advantages
                    advs = mb_returns - mb_values
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    # Calculate neglogpac
                    neglogpac = taken_action_neglogp
                    entropy = tf.reduce_mean(p_entropy)
                    vpredclipped = mb_values + tf.clip_by_value(vpred - mb_values, -CLIPRANGE, CLIPRANGE)
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
    # Start learner process loop
    while True:
        trajectory = {
                    'mb_obs':                            np.empty((nbatch, n_agents, n_agents * 4),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_other_actions':                  np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_advs':                           np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_returns':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_values':                                np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_probs':                          np.empty((nbatch, n_agents, 5),                   dtype=np.float32),
                    'mb_qtargets':                       np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_q':                              np.empty((nbatch, n_agents),                   dtype=np.float32)
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        while cur_traj_size < nbatch:
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_rewards'].shape[0]
            for k, v in trajectory.items():
                trajectory[k][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-cur_traj_size)]
            cur_traj_size += min(a_traj_size, nbatch-cur_traj_size)
        # Collect trajectories
        for ai in range(n_agents):
            b_obs = trajectory['mb_obs'][:, ai]
            b_actions = trajectory['mb_actions'][:, ai]
            b_other_actions = trajectory['mb_other_actions'][:, ai]
            b_dones = trajectory['mb_dones'][:, ai]
            b_neglogpacs = trajectory['mb_neglogpacs'][:, ai]
            b_rewards = trajectory['mb_rewards'][:, ai]
            b_returns = trajectory['mb_returns'][:, ai]
            b_values = trajectory['mb_values'][:, ai]
            b_probs = trajectory['mb_probs'][:, ai]
            b_qtargets = trajectory['mb_qtargets'][:, ai]
            b_q = trajectory['mb_q'][:, ai]
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss(b_obs, b_actions, b_other_actions,
                                                    b_returns, b_dones,
                                                    b_neglogpacs,
                                                    b_rewards,
                                                    b_values, b_probs,
                                                    b_qtargets, b_q)
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Send updated agent and summaries to coordinator
        print("PPO NAV:")
        print("approxkl: ", np.array(approxkls))
        print("loss: ", np.array(losses))
        print("entropy: ", np.array(entropies))
        print("rewards: ", np.mean(b_rewards, axis=0))
        print("")
        sys.stdout.flush()


if rank >= 0 and rank <= 5:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 6 and rank <= 6:
    learner()
