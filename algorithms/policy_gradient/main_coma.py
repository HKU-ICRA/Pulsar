import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_coma import Network

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
batch_size = nsteps = 25
no_of_agents_per_env = 2


def actor(n_agents=no_of_agents_per_env):
    # GAE hyper-parameters
    lam = 0.95
    gamma = 0.99
    # Setup environment
    env = Game()
    obs = env.reset()
    dones = False
    # Build network architecture
    network = Network(1)
    agent_actions = np.array([0 for _ in range(n_agents)])
    network(np.array([obs[0]]))
    network.get_q_value(np.array([obs[0]]), np.expand_dims(agent_actions[1:], axis=0))
    network.get_tq_value(np.array([obs[0]]), np.expand_dims(agent_actions[1:], axis=0))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_obs = np.zeros([n_agents, nsteps, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([n_agents, nsteps, 1], dtype=np.float32)
        mb_qvalues = np.zeros([n_agents, nsteps, 1], dtype=np.float32)
        mb_all_qvalues = np.zeros([n_agents, nsteps, 5], dtype=np.float32)
        mb_neglogpacs = np.zeros([n_agents, nsteps, 1], dtype=np.float32)
        mb_dones = np.zeros([n_agents, nsteps,], dtype=np.float32)
        mb_actions = np.zeros([n_agents, nsteps,], dtype=np.float32)
        mb_all_actions = np.zeros([n_agents, nsteps, n_agents], dtype=np.float32)
        mb_qprob = np.zeros([n_agents, nsteps, 5], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions = []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                agent_actions.append(actions[0])
                mb_neglogpacs[ai, step] = neglogp
                mb_obs[ai, step] = obs_dc[ai]
                mb_actions[ai, step] = actions
                mb_qprob[ai, step] = probs
            agent_actions = np.array(agent_actions)
            for ai in range(n_agents):
                mb_all_actions[ai, step] = agent_actions
            obs_dc = deepcopy(obs)
            for ai in range(n_agents):
                other_idx = 1 - ai
                mb_all_qvalues[ai, step] = network.get_tq_value(np.array([obs_dc[ai]]), np.expand_dims(agent_actions[other_idx:other_idx+1], axis=0))
                mb_qvalues[ai, step] = mb_all_qvalues[ai, step, int(mb_actions[ai, step])]
            obs, rewards, dones = env.step(agent_actions)
            # Handle rewards
            for ai in range(n_agents):
                mb_dones[ai, step] = dones
                mb_rewards[ai, step] = rewards
            if dones:
                obs = env.reset()
        # Get last values
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        # perform GAE calculation
        for ai in range(n_agents):
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    #nextnonterminal = 1.0 - dones
                    nextnonterminal = 1.0 - mb_dones[ai, t]
                    donesum = np.sum(mb_dones[ai], axis=0)
                    agent_actions = []
                    for ai in range(n_agents):
                        obs_dc = deepcopy(obs)
                        actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                        agent_actions.append(actions[0])
                    other_idx = 1 - ai
                    next_q = network.get_tq_value(np.array([obs_dc[ai]]), np.expand_dims(agent_actions[other_idx:other_idx+1], axis=0).astype(np.int32))[:, int(agent_actions[ai])]
                    last_G = np.array(next_q) * (1 - donesum)
                else:
                    #nextnonterminal = 1.0 - mb_dones[t+1]
                    nextnonterminal = 1.0 - mb_dones[ai, t]
                    last_G = mb_returns[ai, t+1]
                    next_q = mb_qvalues[ai, t+1]
                mb_returns[ai, t] = mb_rewards[ai, t] + lam * gamma * nextnonterminal * last_G + (1 - lam) * gamma * next_q
        # Send trajectory to learner
        mb_qvalues = np.squeeze(mb_qvalues, axis=-1)
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_advs = np.squeeze(mb_advs, axis=-1)
        trajectory = {
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
                    'mb_dones': mb_dones,
                    'mb_qvalues': mb_qvalues,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_rewards': mb_rewards,
                    'mb_qprob': mb_qprob,
                    'mb_all_qvalues': mb_all_qvalues,
                    'mb_advs': mb_advs,
                    'mb_all_actions': mb_all_actions,
                    'mb_returns': mb_returns
                    }
        comm.send(trajectory, dest=learners[agent_type])


def learner():
    # Learner hyperparameters
    ent_coef = 0.0
    vf_coef = 1.0#0.5
    CLIPRANGE = 0.2
    max_grad_norm = 5.0
    noptepochs = 1
    bptt_ts = 2 #16
    n_actors = len(actors[0])
    n_agents = no_of_agents_per_env
    nbatch = batch_size * n_actors
    nbatch_steps = nbatch // 1
    current_t_steps = 0
    # Build network architecture
    network = Network(nbatch_steps)
    env = Game()
    obs = env.reset()
    agent_actions = np.array([0 for _ in range(no_of_agents_per_env)])
    network(np.array([obs[0]]))
    network.get_q_value(np.array([obs[0]]), np.expand_dims(agent_actions[1:], axis=0))
    network.get_tq_value(np.array([obs[0]]), np.expand_dims(agent_actions[1:], axis=0))
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # PPO RL optimization loss function
    def critic_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues, b_advs, b_all_actions, b_rewards):
        for t in reversed(range(b_obs.shape[1])):
            for ai in range(n_agents):
                with tf.GradientTape() as tape:
                    other_idx = 1 - ai
                    mb_all_actions = tf.dtypes.cast(b_all_actions[ai, t:t+1, other_idx:other_idx+1], tf.int32)
                    vpred = network.get_q_value(b_obs[ai, t:t+1], mb_all_actions)
                    batch_idxes = tf.expand_dims(tf.range(mb_all_actions.shape[0]), axis=1)
                    mb_actions = tf.expand_dims(b_actions[ai, t:t+1], axis=1)
                    mb_actions_idxes = tf.concat([batch_idxes, mb_actions], axis=-1)
                    vpred = tf.gather_nd(vpred, mb_actions_idxes)
                    vf_loss = tf.reduce_mean(tf.square(vpred - b_returns[ai, t:t+1]))
                    vf_loss = vf_loss * vf_coef
                    var = network.trainable_variables
                    var = [v for v in var if "qvalue" in v.name]
                    grads = tape.gradient(vf_loss, var)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                    grads_and_var = zip(grads, var)
                    network.coptimizer.apply_gradients(grads_and_var)
    #@tf.function
    def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues, b_advs, b_all_actions, b_rewards):
        # Stochastic selection
        inds = tf.range(nbatch)
        # Buffers for recording
        losses_total = []
        approxkls = []
        entropies = []
        advss = []
        # Start SGD
        for _ in range(noptepochs):
            inds = tf.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_steps):
                end = start + nbatch_steps
                # Gather mini-batch
                mbinds = inds[start:end]
                mb_obs = tf.gather(b_obs, mbinds, axis=1)
                mb_actions = tf.gather(b_actions, mbinds, axis=1)
                mb_dones = tf.gather(b_dones, mbinds, axis=1)
                #mb_values = tf.gather(b_qvalues, mbinds)
                #advs = tf.gather(b_advs, mbinds, axis=1)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds, axis=1)
                mb_returns = tf.gather(b_returns, mbinds, axis=1)
                mb_all_actions = tf.gather(b_all_actions, mbinds, axis=1)
                mb_all_actions = tf.dtypes.cast(mb_all_actions, tf.int32)
                mb_rewards = tf.gather(b_rewards, mbinds, axis=1)
                mb_qprob = tf.gather(b_qprob, mbinds, axis=1)
                # Get Q-val
                new_vpreds = []
                all_baseline_qs = []
                for ai in range(n_agents):
                    other_idx = 1 - ai
                    new_vpred = network.get_q_value(mb_obs[ai], mb_all_actions[ai, :, other_idx:other_idx+1])
                    all_baseline_qs.append(new_vpred)
                    batch_idxes = tf.expand_dims(tf.range(mb_obs[ai].shape[0]), axis=1)
                    mb_actions_b = tf.expand_dims(mb_actions[ai], axis=1)
                    mb_actions_idxes = tf.concat([batch_idxes, mb_actions_b], axis=-1)
                    gathered_new_vpred = tf.gather_nd(new_vpred, mb_actions_idxes)
                    new_vpreds.append(gathered_new_vpred)
                new_vpreds = np.array(new_vpreds)
                all_baseline_qs = np.array(all_baseline_qs)
                # Calculate advantage
                baseline = tf.reduce_sum(all_baseline_qs * mb_qprob, axis=-1)
                advs = new_vpreds - baseline
                #advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                advss.append(advs[0])
                with tf.GradientTape() as tape:
                    # Action
                    p_entropies = []
                    p_probss = []
                    taken_action_neglogps = []
                    for ai in range(n_agents):
                        _, _, p_entropy, p_probs, _, taken_action_neglogp = network(mb_obs[ai], [mb_actions[ai]])
                        p_entropies.append(p_entropy)
                        p_probss.append(p_probs)
                        taken_action_neglogps.append(taken_action_neglogp)
                    taken_action_neglogps = np.array(taken_action_neglogps)
                    neglogpac = tf.reduce_mean(taken_action_neglogps, axis=-1)
                    entropy = tf.reduce_mean(p_entropies, axis=-1)
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - tf.reduce_mean(mb_neglogpacs, axis=-1)))
                    #ratio = tf.exp(mb_neglogpacs - neglogpac)
                    #pg_losses = -advs * ratio
                    #pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    #pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    logps = []
                    for ai in range(n_agents):
                        batch_idxes = tf.expand_dims(tf.range(mb_obs[ai].shape[0]), axis=1)
                        mb_actions_b = tf.expand_dims(mb_actions[ai], axis=1)
                        mb_actions_idxes = tf.concat([batch_idxes, mb_actions_b], axis=-1)
                        logp = tf.math.log(tf.gather_nd(p_probss[ai], mb_actions_idxes))
                        logps.append(logp)
                    logps = tf.convert_to_tensor(logps)
                    pg_loss = -tf.reduce_mean(logps * advs)
                    # Total loss
                    loss = pg_loss# - entropy * ent_coef
                    # 1. Get the model parameters
                    var = network.trainable_variables
                    var = [v for v in var if "policy" in v.name]
                    grads = tape.gradient(loss, var)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                    grads_and_var = zip(grads, var)
                    network.poptimizer.apply_gradients(grads_and_var)
                # Record
                losses_total.append(loss)
                approxkls.append(approxkl)
                entropies.append(entropy)
        losses_total = tf.reduce_mean(losses_total)
        approxkls = tf.reduce_mean(approxkls)
        entropies = tf.reduce_mean(entropies)
        advss = tf.reduce_mean(advss)
        return losses_total, approxkls, entropies, advss
    # Start learner process loop
    while True:
        trajectory = {
                    'mb_obs':                            np.empty((n_agents, nbatch, n_agents * 4),     dtype=np.float32),
                    'mb_actions':                        np.empty((n_agents, nbatch,),     dtype=np.int32),
                    'mb_dones':                          np.empty((n_agents, nbatch),         dtype=np.float32),
                    'mb_qvalues':                        np.empty((n_agents, nbatch),         dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((n_agents, nbatch,),         dtype=np.float32),
                    'mb_rewards':                        np.empty((n_agents, nbatch),         dtype=np.float32),
                    'mb_qprob':                          np.empty((n_agents, nbatch, 5),         dtype=np.float32),
                    'mb_all_qvalues':                    np.empty((n_agents, nbatch, 5),         dtype=np.float32),
                    'mb_advs':                           np.empty((n_agents, nbatch),         dtype=np.float32),
                    'mb_all_actions':                    np.empty((n_agents, nbatch, no_of_agents_per_env),         dtype=np.float32),
                    'mb_returns':                        np.empty((n_agents, nbatch,),         dtype=np.float32),
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        while cur_traj_size < nbatch:
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_rewards'].shape[1]
            for k, v in trajectory.items():
                trajectory[k][:, cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory[k]
            cur_traj_size += min(a_traj_size, nbatch-cur_traj_size)
        # Collect trajectories
        b_obs = trajectory['mb_obs']
        b_actions = trajectory['mb_actions']
        b_dones = trajectory['mb_dones']
        b_qvalues = trajectory['mb_qvalues']
        b_neglogpacs = trajectory['mb_neglogpacs']
        b_advs = trajectory['mb_advs']
        mb_rewards = trajectory['mb_rewards']
        b_qprob = trajectory['mb_qprob']
        b_all_qvalues = trajectory['mb_all_qvalues']
        b_all_actions = trajectory['mb_all_actions']
        b_returns = trajectory['mb_returns']
        # Start SGD and optimize model via Adam
        critic_loss(b_obs, b_actions, b_returns, b_dones,
                                                b_qvalues, b_neglogpacs, b_qprob,
                                                b_all_qvalues, b_advs, b_all_actions,
                                                mb_rewards)
        losses, approxkls, entropies, advss = ppo_loss(b_obs, b_actions, b_returns, b_dones,
                                                b_qvalues, b_neglogpacs, b_qprob,
                                                b_all_qvalues, b_advs, b_all_actions,
                                                mb_rewards)
        current_t_steps += 1
        if current_t_steps % (150//nsteps) == 0:
        #network.polyak_qnet()
            network.mlp1_tq.set_weights(network.mlp1_q.get_weights())
            network.mlp2_tq.set_weights(network.mlp2_q.get_weights())
            network.mlp3_tq.set_weights(network.mlp3_q.get_weights())
            network.tq.set_weights(network.q.get_weights())
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Send updated agent and summaries to coordinator
        print("PPO NAV:")
        print("approxkl: ", np.array(approxkls))
        print("loss: ", np.array(losses))
        print("entropy: ", np.array(entropies))
        print("rewards: ", np.mean(np.mean(mb_rewards, axis=0), axis=0))
        print("advs: ", np.array(advss))
        print("")
        sys.stdout.flush()


if rank >= 0 and rank <= 5:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 6 and rank <= 6:
    learner()



"""
def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues, b_advs, b_all_actions, b_rewards):
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
                #mb_values = tf.gather(b_qvalues, mbinds)
                advs = tf.gather(b_advs, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_all_actions = tf.gather(b_all_actions, mbinds)
                mb_all_actions = tf.dtypes.cast(mb_all_actions, tf.int32)
                mb_rewards = tf.gather(b_rewards, mbinds)
                mb_qprob = tf.gather(b_qprob, mbinds)
                with tf.GradientTape() as tape:
                    # Action
                    p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    new_vpred = network.get_q_value(mb_obs, mb_all_actions)
                    batch_idxes = tf.expand_dims(tf.range(mb_obs.shape[0]), axis=1)
                    mb_actions_b = tf.expand_dims(mb_actions, axis=1)
                    mb_actions_idxes = tf.concat([batch_idxes, mb_actions_b], axis=-1)
                    new_vpred = tf.gather_nd(new_vpred, mb_actions_idxes)
                    baseline = 0
                    for ai in range(5):
                        dub_all_actions = np.array(mb_all_actions)
                        dub_all_actions[:, 0] = ai
                        baseline += network.get_q_value(mb_obs, dub_all_actions)[:, ai] * mb_qprob[:, ai]
                    advs = new_vpred - baseline
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    neglogpac = taken_action_neglogp
                    entropy = tf.reduce_mean(p_entropy)
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    # Calculate ratio (pi current policy / pi old policy)
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    # Defining Loss = - J is equivalent to max J
                    pg_losses = -advs * ratio
                    pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    # Final PG loss
                    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    #pg_loss = tf.reduce_mean(neglogpac * advs)
                    # Total loss
                    loss = pg_loss - entropy * ent_coef
                    # 1. Get the model parameters
                    var = network.trainable_variables
                    var = [v for v in var if "policy" in v.name]
                    grads = tape.gradient(loss, var)
                    # 3. Calculate the gradients
                    # Clip the gradients (normalize)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                    grads_and_var = zip(grads, var)
                    # zip aggregate each gradient with parameters associated
                    # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
                    network.optimizer.apply_gradients(grads_and_var)
                # Record
                losses_total.append(loss)
                approxkls.append(approxkl)
                entropies.append(entropy)
        losses_total = tf.reduce_mean(losses_total)
        approxkls = tf.reduce_mean(approxkls)
        entropies = tf.reduce_mean(entropies)
        return losses_total, approxkls, entropies
"""
