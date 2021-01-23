import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_pr2 import Network


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
    env = Game()
    obs = env.reset()
    dones = False
    # Build network architecture
    network = Network(1)
    network(np.array([obs[0]]))
    network.get_tp(np.array([obs[0]]))
    network.get_q_value(np.array([obs[0]]), np.array([0]), np.array([0]))
    network.get_tq_value(np.array([obs[0]]), np.array([0]), np.array([0]))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_obs = np.zeros([nsteps, n_agents, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_qvalues = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_values = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_tqvalues = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_besta_tqvalues = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_all_qvalues = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
        mb_all_tqvalues = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_qprob = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
        mb_qtprob = np.zeros([nsteps, n_agents, 5], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions, agent_tactions = [], []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                mb_values[step, ai] = network.get_value(np.array([obs_dc[ai]]))
                mb_neglogpacs[step, ai] = neglogp
                mb_qprob[step, ai] = probs
                mb_dones[step, ai] = dones
                mb_actions[step, ai] = actions
                mb_obs[step, ai] = obs_dc[ai]
                agent_actions.append(actions[0])
                t_actions, t_probs = network.get_tp(np.array([obs_dc[ai]]))
                mb_qtprob[step, ai] = t_probs
                agent_tactions.append(t_actions[0])
            for ai in range(n_agents):
                other_ai = 1 - ai
                mb_qvalues[step, ai] = network.get_q_value(np.array([obs_dc[ai]]), np.array([agent_actions[ai]]),
                                                                                   np.array([agent_actions[other_ai]]))
                mb_tqvalues[step, ai] = network.get_tq_value(np.array([obs_dc[ai]]), np.array([agent_tactions[ai]]),
                                                                                     np.array([agent_tactions[other_ai]]))
                for idx in range(5):
                    mb_all_tqvalues[step, ai, idx] = network.get_tq_value(np.array([obs_dc[ai]]), np.array([agent_tactions[ai]]),
                                                                                              np.array([idx]))
            agent_actions = np.array(agent_actions)
            obs, rewards, dones = env.step(agent_actions)
            # Handle rewards
            mb_rewards[step, :] = np.array([[rewards] for _ in range(n_agents)])
            if dones:
                obs = env.reset()
        # Get last values
        last_agent_actions, last_agent_tactions, last_agent_qtprobs = [], [], []
        for ai in range(n_agents):
            obs_dc = deepcopy(obs)
            actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
            last_agent_actions.append(actions[0])
            t_actions, t_probs = network.get_tp(np.array([obs_dc[ai]]))
            last_agent_qtprobs.append(t_probs[0])
            last_agent_tactions.append(t_actions[0])
        last_agent_actions = np.array(last_agent_actions)
        last_agent_tactions = np.array(last_agent_tactions)
        last_agent_qtprobs = np.array(last_agent_qtprobs)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_qtargets = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        # perform GAE calculation
        for ai in range(n_agents):
            other_ai = 1 - ai
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    nextnonterminal = 1.0 - dones
                    obs_dc = deepcopy(obs)
                    target_q = 0.0
                    for idx in range(5):
                        tq_val = network.get_tq_value(np.array([obs_dc[ai]]), np.array([last_agent_tactions[ai]]),
                                                                                np.array([idx]))
                        target_q += tq_val * last_agent_qtprobs[other_ai, idx]
                        
                    last_values = network.get_value(np.array([obs_dc[ai]]))
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1, ai]
                    target_q = 0.0
                    for idx in range(5):
                        target_q += mb_all_tqvalues[t+1, ai, idx] * mb_qtprob[t+1, other_ai, idx]
                    last_values = mb_values[t+1, ai]
                mb_qtargets[t, ai] = mb_rewards[t, ai] + nextnonterminal * gamma * target_q
                mb_returns[t, ai] = mb_rewards[t, ai] + nextnonterminal * gamma * last_values
            for t in range(nsteps):
                mb_advs[t, ai] = mb_qvalues[t, ai] - mb_values[t, ai]
        # Send trajectory to learner
        mb_qvalues = np.squeeze(mb_qvalues, axis=-1)
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_advs = np.squeeze(mb_advs, axis=-1)
        mb_qtargets = np.squeeze(mb_qtargets, axis=-1)
        mb_values = np.squeeze(mb_values, axis=-1)
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
                    'mb_returns': mb_returns,
                    'mb_qtargets': mb_qtargets,
                    'mb_values': mb_values
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
    env = Game()
    obs = env.reset()
    network(np.array([obs[0]]))
    network.get_tp(np.array([obs[0]]))
    network.get_q_value(np.array([obs[0]]), np.array([0]), np.array([0]))
    network.get_tq_value(np.array([obs[0]]), np.array([0]), np.array([0]))
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_other_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues, b_advs, b_rewards, b_qtargets, b_values):
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
                mb_qvalues = tf.gather(b_qvalues, mbinds)
                advs = tf.gather(b_advs, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_rewards = tf.gather(b_rewards, mbinds)
                mb_qtargets = tf.gather(b_qtargets, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                mb_qprob = tf.gather(b_qprob, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    neglogpac = taken_action_neglogp
                    entropy = tf.reduce_mean(p_entropy)
                    mb_q = network.get_q_value(mb_obs, mb_actions, mb_other_actions)
                    q_loss = tf.reduce_mean(tf.square(mb_q - mb_qtargets))
                    # Calc value l
                    vpred = network.get_value(mb_obs)
                    vpredclipped = mb_values + tf.clip_by_value(vpred - mb_values, -CLIPRANGE, CLIPRANGE)
                    vf_losses1 = tf.square(vpred - mb_returns)
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    pg_losses = -advs * ratio
                    pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    # Total loss
                    loss = pg_loss - entropy * ent_coef + q_loss * q_coef + vf_loss * vf_coef
                # 1. Get the model parameters
                var = network.trainable_variables
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
    # Start learner process loop
    while True:
        trajectory = {
                    'mb_obs':                            np.empty((nbatch, n_agents, n_agents * 4),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_qvalues':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_qprob':                          np.empty((nbatch, n_agents, 5),                dtype=np.float32),
                    'mb_all_qvalues':                    np.empty((nbatch, n_agents, 5),                dtype=np.float32),
                    'mb_advs':                           np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_returns':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_qtargets':                       np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_values':                         np.empty((nbatch, n_agents),                   dtype=np.float32)
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
            other_ai = 1 - ai
            b_obs = trajectory['mb_obs'][:, ai]
            b_actions = trajectory['mb_actions'][:, ai]
            b_other_actions = trajectory['mb_actions'][:, other_ai]
            b_dones = trajectory['mb_dones'][:, ai]
            b_qvalues = trajectory['mb_qvalues'][:, ai]
            b_neglogpacs = trajectory['mb_neglogpacs'][:, ai]
            b_advs = trajectory['mb_advs'][:, ai]
            mb_rewards = trajectory['mb_rewards'][:, ai]
            b_qprob = trajectory['mb_qprob'][:, ai]
            b_all_qvalues = trajectory['mb_all_qvalues'][:, ai]
            b_returns = trajectory['mb_returns'][:, ai]
            b_qtargets = trajectory['mb_qtargets'][:, ai]
            b_values = trajectory['mb_values'][:, ai]
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss(b_obs, b_actions, b_other_actions,
                                                    b_returns, b_dones,
                                                    b_qvalues, b_neglogpacs, b_qprob,
                                                    b_all_qvalues, b_advs,
                                                    mb_rewards, b_qtargets, b_values)
        network.polyak_qnet()
        network.polyak_pnet()
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Send updated agent and summaries to coordinator
        print("PPO NAV:")
        print("approxkl: ", np.array(approxkls))
        print("loss: ", np.array(losses))
        print("entropy: ", np.array(entropies))
        print("rewards: ", np.mean(mb_rewards, axis=0))
        print("")
        sys.stdout.flush()


if rank >= 0 and rank <= 5:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 6 and rank <= 6:
    learner()
