import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_cm3 import Network


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
batch_size = nsteps = 512
state_switch_steps = int(1e6)
no_of_agents_per_env = 2


def actor(n_agents=2):
    # GAE hyper-parameters
    lam = 0.95
    gamma = 0.99
    # Setup environment
    training_state = 1
    total_steps = 0
    env_1 = Game(n_agents=1)
    obs_1 = env_1.reset()
    dones_1 = False
    env = Game(n_agents=n_agents)
    obs = env.reset()
    dones = False
    # Build network architecture
    network = Network(1)
    network(np.array([obs[0]]), np.array([obs[0]]), stage=2)
    network.get_global_critic(np.array([obs[0]]))
    network.get_credit_critic(np.array([obs[0]]), np.array([obs[0]]), np.array([0]))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        if training_state == 1:
            mb_obs = np.zeros([nsteps, 6], dtype=np.float32)
            mb_rewards = np.zeros([nsteps, 1], dtype=np.float32)
            mb_q = np.zeros([nsteps, 1], dtype=np.float32)
            mb_qs = np.zeros([nsteps, 5], dtype=np.float32)
            mb_neglogpacs = np.zeros([nsteps, 1], dtype=np.float32)
            mb_dones = np.zeros([nsteps,], dtype=np.float32)
            mb_actions = np.zeros([nsteps, 1], dtype=np.float32)
            mb_probs = np.zeros([nsteps, 5], dtype=np.float32)
            for step in range(nsteps):
                # Get actions of training agent
                obs_1_dc = deepcopy(obs_1)
                actions, neglogp, entropy, probs, p = network(np.array([obs_1_dc[0]]))
                mb_neglogpacs[step] = neglogp
                mb_dones[step] = dones_1
                mb_actions[step] = actions
                mb_obs[step] = obs_1_dc[0]
                mb_probs[step] = probs
                agent_actions = np.array([actions[0]])
                mb_qs[step] = network.get_global_critic(np.array([obs_1_dc[0]]))
                mb_q[step] = mb_qs[step, agent_actions[0]]
                obs_1, rewards, dones_1 = env_1.step(agent_actions)
                # Handle rewards
                mb_rewards[step] = np.array([[rewards]])
                if dones_1:
                    obs_1 = env_1.reset()
                total_steps += 1
            # Get last worker values
            obs_1_dc = deepcopy(obs_1)
            last_actions, neglogp, entropy, last_probs, p = network(np.array([obs_1_dc[0]]))
            last_qs = network.get_global_critic(np.array([obs_1_dc[0]]))
            last_single_qs = last_qs[0, last_actions[0]]
            # discount/bootstrap off value fn
            mb_values = np.zeros_like(mb_rewards)
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            mb_qtargets = np.zeros_like(mb_rewards)
            lastgaelam = 0
            # perform GAE calculation
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    nextnonterminal = 1.0 - dones_1
                    nextvalues = 0
                    for idx in range(5):
                        nextvalues += last_qs[0, idx] * last_probs[0, idx]
                    target_q = last_single_qs
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    nextvalues = 0
                    for idx in range(5):
                        nextvalues += mb_qs[t+1, idx] * mb_probs[t+1, idx]
                    target_q = mb_q[t+1]
                # Actual
                for idx in range(5):
                    mb_values[t] += mb_qs[t, idx] * mb_probs[t, idx]
                delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                mb_qtargets[t] = mb_rewards[t] + nextnonterminal * gamma * target_q
            mb_returns = mb_advs + mb_values
            # Send trajectory to learner
            mb_actions = np.squeeze(mb_actions, axis=-1)
            mb_rewards = np.squeeze(mb_rewards, axis=-1)
            mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
            mb_returns = np.squeeze(mb_returns, axis=-1)
            mb_advs = np.squeeze(mb_advs, axis=-1)
            mb_values = np.squeeze(mb_values, axis=-1)
            mb_q = np.squeeze(mb_q, axis=-1)
            mb_qtargets = np.squeeze(mb_qtargets, axis=-1)
            # Check if we should switch stage
            if total_steps >= state_switch_steps:
                training_state = 2
            #  Trajectory
            trajectory = {
                    'training_state': training_state,
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
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
        else:
            mb_obs = np.zeros([nsteps, n_agents, 6], dtype=np.float32)
            mb_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
            mb_q = np.zeros([nsteps, n_agents, n_agents, 1], dtype=np.float32)
            mb_qs = np.zeros([nsteps, n_agents, n_agents, 5], dtype=np.float32)
            mb_neglogpacs = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
            mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
            mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
            mb_probs = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
            for step in range(nsteps):
                # Get actions of training agent
                agent_actions = []
                for ai in range(n_agents):
                    obs_dc = deepcopy(obs)
                    actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]), np.array([obs_dc[1 - ai]]), stage=2)
                    mb_neglogpacs[step, ai] = neglogp
                    mb_dones[step, ai] = dones
                    mb_actions[step, ai] = actions
                    mb_obs[step, ai] = obs_dc[ai]
                    mb_probs[step, ai] = probs
                    agent_actions.append(actions[0])
                agent_actions = np.array(agent_actions)
                for ai in range(n_agents):
                    for aj in range(n_agents):
                        mb_qs[step, ai, aj] = network.get_credit_critic(np.array([obs_dc[ai]]), np.array([obs_dc[aj]]), np.array([agent_actions[aj]]))
                        mb_q[step, ai, aj] = mb_qs[step, ai, aj, agent_actions[ai]]
                obs, rewards, dones = env.step(agent_actions)
                # Handle rewards
                mb_rewards[step] = np.array([[rewards] for _ in range(n_agents)])
                if dones:
                    obs = env.reset()
            # Get last worker values
            last_actions, last_qs, last_q, last_probs = [], [], [], []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]), np.array([obs_dc[1 - ai]]), stage=2)
                last_actions.append(actions[0])
                last_probs.append(probs)
            for ai in range(n_agents):
                last_qs_buffer, last_q_buffer = [], []
                for aj in range(n_agents):
                    obs_dc = deepcopy(obs)
                    lqs = network.get_credit_critic(np.array([obs_dc[ai]]), np.array([obs_dc[aj]]), np.array([last_actions[aj]]))
                    last_qs_buffer.append(lqs)
                    lq = lqs[0, last_actions[ai]]
                    last_q_buffer.append(lq)
                last_qs.append(last_qs_buffer)
                last_q.append(last_q_buffer)
            # discount/bootstrap off value fn
            mb_advs = np.zeros([mb_qs.shape[0], n_agents, n_agents, 1])
            mb_qtargets = np.zeros_like(mb_advs)
            mb_values = np.zeros_like(mb_advs)
            mb_returns = np.zeros_like(mb_advs)
            lastgaelam = 0
            # perform GAE calculation
            for ai in range(n_agents):
                for t in reversed(range(nsteps)):
                    for aj in range(n_agents):    
                    # Compute credit q target
                        if t == nsteps - 1:
                            nextnonterminal = 1.0 - dones
                            target_q = last_q[ai][aj]
                            nextvalues = 0
                            for idx in range(5):
                                nextvalues += last_qs[ai][aj][0, idx] * last_probs[aj][0, idx]
                        else:
                            nextnonterminal = 1.0 - mb_dones[t+1, ai]
                            target_q = mb_q[t+1, ai, aj]
                            nextvalues = 0
                            for idx in range(5):
                                nextvalues += mb_qs[t+1, ai, aj, idx] * mb_probs[t+1, aj, idx]
                        for idx in range(5):
                            mb_values[t, ai, aj] += mb_qs[t, ai, aj, idx] * mb_probs[t, aj, idx]
                        delta = mb_rewards[t, ai] + gamma * nextvalues * nextnonterminal - mb_values[t, ai, aj]
                        mb_advs[t, ai, aj] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                        mb_qtargets[t, ai, aj] = mb_rewards[t, ai] + nextnonterminal * gamma * target_q
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
                    'training_state': training_state,
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
                    'mb_dones': mb_dones,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_rewards': mb_rewards,
                    'mb_advs': mb_advs,
                    'mb_returns': mb_returns,
                    'mb_values': mb_values,
                    'mb_probs': mb_probs,
                    'mb_qtargets': mb_qtargets,
                    'mb_qs': mb_qs,
                    'mb_q': mb_q
                    }
            comm.send(trajectory, dest=learners[agent_type])


def learner():
    start_time = time.time()
    # Learner hyperparameters
    ent_coef = 0.05
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
    network(np.array([obs[0]]), np.array([obs[0]]), stage=2)
    network.get_global_critic(np.array([obs[0]]))
    network.get_credit_critic(np.array([obs[0]]), np.array([obs[0]]), np.array([0]))
    switched_stage = False
    should_initialize_credit = False
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_neglogpacs, b_rewards, b_values, b_probs, b_qtargets, b_q):
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
                mb_values = tf.gather(b_values, mbinds)
                mb_probs = tf.gather(b_probs, mbinds)
                mb_qtargets = tf.gather(b_qtargets, mbinds)
                mb_qvalues = tf.gather(b_q, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs, taken_action=[mb_actions])
                    qpred = network.get_global_critic(mb_obs)
                    batch_idx = tf.range(mb_obs.shape[0])
                    batch_idx = tf.expand_dims(batch_idx, axis=-1)
                    gather_actions = tf.concat([batch_idx, tf.expand_dims(mb_actions, axis=-1)], axis=-1)
                    qpred = tf.gather_nd(qpred, gather_actions)
                    qpredclipped = mb_qvalues + tf.clip_by_value(qpred - mb_qvalues, -CLIPRANGE, CLIPRANGE)
                    q_losses1 = tf.square(qpred - mb_qtargets)
                    q_losses2 = tf.square(qpredclipped - mb_qtargets)
                    q_loss = .5 * tf.reduce_mean(tf.maximum(q_losses1, q_losses2))
                    # Batch normalize the advantages
                    advs = mb_returns - mb_values
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    # Calculate neglogpac
                    neglogpac = taken_action_neglogp
                    entropy = tf.reduce_mean(p_entropy)
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    pg_losses = -advs * ratio
                    pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    loss = pg_loss - entropy * ent_coef + q_loss * vf_coef
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
    # Stage 2 PPO loss function
    @tf.function
    def ppo_loss_s2(b_obs, b_actions, b_returns, b_dones, b_neglogpacs, b_rewards, b_values, b_probs, b_qtargets, b_q):
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
                mb_values = tf.gather(b_values, mbinds)
                mb_probs = tf.gather(b_probs, mbinds)
                mb_qtargets = tf.gather(b_qtargets, mbinds)
                mb_qvalues = tf.gather(b_q, mbinds)
                with tf.GradientTape() as tape:
                    all_actions, all_neglogp, all_entropy, all_taken_action_neglogp = [], [], [], []
                    for ai in range(n_agents):
                        p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs[:, ai], mb_obs[:, 1 - ai], taken_action=[mb_actions[:, ai]], stage=2)
                        all_actions.append(p_actions)
                        all_neglogp.append(p_neglogp)
                        all_entropy.append(p_entropy)
                        all_taken_action_neglogp.append(taken_action_neglogp)
                    all_actions = tf.stack(all_actions, axis=1)
                    all_neglogp = tf.stack(all_neglogp, axis=1)
                    all_entropy = tf.stack(all_entropy, axis=1)
                    all_taken_action_neglogp = tf.stack(all_taken_action_neglogp, axis=1)
                    neglogpac = all_taken_action_neglogp
                    qpreds = []
                    for ai in range(n_agents):
                        qpred = []
                        for aj in range(n_agents):
                            qq = network.get_credit_critic(mb_obs[:, ai], mb_obs[:, aj], mb_actions[:, aj])
                            qpred.append(qq)
                        qpred = tf.stack(qpred, axis=1)
                        qpreds.append(qpred)
                    qpreds = tf.stack(qpreds, axis=1)
                    tiled_mb_actions = tf.stack([tf.one_hot(mb_actions, 5) for _ in range(n_agents)], axis=2)
                    qpreds = tf.reduce_sum(qpreds * tiled_mb_actions, axis=-1)
                    qpredclipped = mb_qvalues + tf.clip_by_value(qpreds - mb_qvalues, -CLIPRANGE, CLIPRANGE)
                    q_losses1 = tf.square(qpreds - mb_qtargets)
                    q_losses2 = tf.square(qpredclipped - mb_qtargets)
                    q_losses = tf.maximum(q_losses1, q_losses2)
                    q_losses = tf.reduce_sum(q_losses, axis=-1)
                    q_losses = tf.reduce_sum(q_losses, axis=-1)
                    q_losses /= n_agents**2
                    q_loss = .5 * tf.reduce_mean(q_losses)
                    # Batch normalize the advantages
                    pg_i_loss = []
                    for ai in range(n_agents):
                        pg_j_loss = []
                        for aj in range(n_agents):
                            advs = mb_returns[:, ai, aj] - mb_values[:, ai, aj]
                            advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                            ratio = tf.exp(mb_neglogpacs[:, ai] - neglogpac[:,  ai])
                            pg_losses1 = -advs * ratio
                            pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                            pg_losses = tf.maximum(pg_losses1, pg_losses2)
                            pg_j_loss.append(pg_losses)
                        pg_j_loss = tf.stack(pg_j_loss, axis=1)
                        pg_i_loss.append(pg_j_loss)
                    pg_i_loss = tf.stack(pg_i_loss, axis=1)
                    pg_loss = tf.reduce_sum(pg_i_loss, axis=-1)
                    pg_loss = tf.reduce_sum(pg_loss, axis=-1)
                    pg_loss = tf.reduce_mean(pg_loss)
                    # Entropy, KL
                    entropy = tf.reduce_mean(p_entropy)
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    loss = pg_loss - entropy * ent_coef + q_loss * vf_coef
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
        trajectory_s1 = {
                    'mb_obs':                            np.empty((nbatch, 6),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch,),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch,),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch,),                   dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch,),                   dtype=np.float32),
                    'mb_advs':                           np.empty((nbatch,),                   dtype=np.float32),
                    'mb_returns':                        np.empty((nbatch,),                   dtype=np.float32),
                    'mb_values':                                np.empty((nbatch,),                   dtype=np.float32),
                    'mb_probs':                          np.empty((nbatch, 5),                   dtype=np.float32),
                    'mb_qtargets':                       np.empty((nbatch,),                   dtype=np.float32),
                    'mb_q':                              np.empty((nbatch,),                   dtype=np.float32)
                    }
        trajectory_s2 = {
                    'mb_obs':                            np.empty((nbatch, n_agents, 6),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_advs':                           np.empty((nbatch, n_agents, n_agents),                   dtype=np.float32),
                    'mb_returns':                        np.empty((nbatch, n_agents, n_agents),                   dtype=np.float32),
                    'mb_values':                                np.empty((nbatch, n_agents, n_agents),                   dtype=np.float32),
                    'mb_probs':                          np.empty((nbatch, n_agents, 5),                   dtype=np.float32),
                    'mb_qtargets':                       np.empty((nbatch, n_agents, n_agents),                   dtype=np.float32),
                    'mb_qs':                             np.empty((nbatch, n_agents, n_agents, 5),                   dtype=np.float32),
                    'mb_q':                              np.empty((nbatch, n_agents, n_agents),                   dtype=np.float32)
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        should_switch_state = False
        while cur_traj_size < nbatch:
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_rewards'].shape[0]
            training_state = a_trajectory['training_state']
            if training_state == 2 and switched_stage == False:
                should_switch_state = True
                training_state = 1
            for k, v in a_trajectory.items():
                if k != 'training_state':
                    if training_state == 1:
                        trajectory_s1[k][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-cur_traj_size)]
                    else:
                        trajectory_s2[k][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-cur_traj_size)]
            cur_traj_size += min(a_traj_size, nbatch-cur_traj_size)
        if should_switch_state:
            should_initialize_credit = True
            switched_stage = True
            should_switch_state = False
        # Collect trajectories
        if training_state == 1:
            b_obs = trajectory_s1['mb_obs']
            b_actions = trajectory_s1['mb_actions']
            b_dones = trajectory_s1['mb_dones']
            b_neglogpacs = trajectory_s1['mb_neglogpacs']
            b_rewards = trajectory_s1['mb_rewards']
            b_returns = trajectory_s1['mb_returns']
            b_values = trajectory_s1['mb_values']
            b_probs = trajectory_s1['mb_probs']
            b_qtargets = trajectory_s1['mb_qtargets']
            b_q = trajectory_s1['mb_q']
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss(b_obs, b_actions,
                                                    b_returns, b_dones,
                                                    b_neglogpacs,
                                                    b_rewards,
                                                    b_values, b_probs,
                                                    b_qtargets, b_q)
        else:
            b_obs = trajectory_s2['mb_obs']
            b_actions = trajectory_s2['mb_actions']
            b_dones = trajectory_s2['mb_dones']
            b_neglogpacs = trajectory_s2['mb_neglogpacs']
            b_rewards = trajectory_s2['mb_rewards']
            b_returns = trajectory_s2['mb_returns']
            b_values = trajectory_s2['mb_values']
            b_probs = trajectory_s2['mb_probs']
            b_qtargets = trajectory_s2['mb_qtargets']
            b_q = trajectory_s2['mb_q']
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss_s2(b_obs, b_actions,
                                                    b_returns, b_dones,
                                                    b_neglogpacs,
                                                    b_rewards,
                                                    b_values, b_probs,
                                                    b_qtargets, b_q)
        # Initialize credit
        if should_initialize_credit:
            network.initialize_credit_weights()
            network.reset_optimizer()
            should_initialize_credit = False
        # Check rew boundary
        if training_state == 2 and np.mean(b_rewards[:, 0], axis=0) >= 0.6:
            print(f"REWARD TARGET REACHED. TIME TAKEN = {time.time() - start_time}")
            return
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Send updated agent and summaries to coordinator
        print(f"PPO (run-time = {time.time() - start_time}):")
        print(f"training state: {training_state}")
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
