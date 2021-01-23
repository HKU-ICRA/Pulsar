import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_normal import Network

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
    network(np.array([obs[0]]))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_obs = np.zeros([nsteps, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([nsteps, 1], dtype=np.float32)
        mb_qvalues = np.zeros([nsteps, 1], dtype=np.float32)
        mb_all_qvalues = np.zeros([nsteps, 5], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps,], dtype=np.float32)
        mb_actions = np.zeros([nsteps,], dtype=np.float32)
        mb_qprob = np.zeros([nsteps, 5], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions = []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, q, probs, p = network(np.array([obs_dc[ai]]))
                agent_actions.append(actions[0])
                if ai == 0:
                    mb_neglogpacs[step] = neglogp
                    mb_obs[step] = obs_dc[0]
                    mb_qvalues[step] = q
                    mb_dones[step] = dones
                    mb_actions[step] = actions
                    mb_qprob[step] = probs
            obs, rewards, dones = env.step(np.array(agent_actions))
            # Handle rewards
            mb_rewards[step] = rewards
            if dones:
                obs = env.reset()
        #batch of steps to batch of rollouts
        obs_dc = deepcopy(obs)
        _, _, _, last_values, _, _ = network(np.array([obs_dc[0]]))
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        # perform GAE calculation
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_qvalues[t+1]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_qvalues[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_qvalues
        # Send trajectory to learner
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_qvalues = np.squeeze(mb_qvalues, axis=-1)
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        trajectory = {
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
                    'mb_returns': mb_returns,
                    'mb_dones': mb_dones,
                    'mb_qvalues': mb_qvalues,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_rewards': mb_rewards,
                    'mb_qprob': mb_qprob,
                    'mb_all_qvalues': mb_all_qvalues
                    }
        comm.send(trajectory, dest=learners[agent_type])


def learner():
    # Learner hyperparameters
    ent_coef = 0.01
    vf_coef = 0.5
    CLIPRANGE = 0.2
    max_grad_norm = 5.0
    noptepochs = 4
    bptt_ts = 2 #16
    n_actors = len(actors[0])
    nbatch = batch_size * n_actors
    nbatch_steps = nbatch // 4
    # Build network architecture
    network = Network(nbatch_steps)
    env = Game()
    obs = env.reset()
    network(np.array([obs[0]]))
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # Transformer loss function
    @tf.function
    def transformer_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues):
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
                mb_obs = tf.gather(b_obs, mbinds)
                mb_actions = tf.gather(b_actions, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_values = tf.gather(b_qvalues, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, vpred, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    advs = mb_returns - mb_values
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    advss.append(advs)
                    # Calculate neglogpac
                    neglogpac = taken_action_neglogp
                    # Calculate the entropy
                    # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                    entropy = tf.reduce_mean(p_entropy)
                    # CALCULATE THE LOSS
                    # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
                    # Clip the value to reduce variability during Critic training
                    # Get the predicted value
                    vpredclipped = mb_values + tf.clip_by_value(vpred - mb_values, -CLIPRANGE, CLIPRANGE)
                    # Unclipped value
                    vf_losses1 = tf.square(vpred - mb_returns)
                    # Clipped value
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    # Calculate ratio (pi current policy / pi old policy)
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    # Defining Loss = - J is equivalent to max J
                    pg_losses = -advs * ratio
                    pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    # Final PG loss
                    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    # Total loss
                    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
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
                losses_total.append(loss)
                approxkls.append(approxkl)
                entropies.append(entropy)
        losses_total = tf.reduce_mean(losses_total)
        approxkls = tf.reduce_mean(approxkls)
        entropies = tf.reduce_mean(entropies)
        advss = tf.reduce_mean(advss)
        return losses_total, approxkls, entropies, advss
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues):
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
                mb_obs = tf.gather(b_obs, mbinds)
                mb_actions = tf.gather(b_actions, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_values = tf.gather(b_qvalues, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, vpred, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    advs = mb_returns - mb_values
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    advss.append(advs)
                    # Calculate neglogpac
                    neglogpac = taken_action_neglogp
                    # Calculate the entropy
                    # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                    entropy = tf.reduce_mean(p_entropy)
                    # CALCULATE THE LOSS
                    # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
                    # Clip the value to reduce variability during Critic training
                    # Get the predicted value
                    vpredclipped = mb_values + tf.clip_by_value(vpred - mb_values, -CLIPRANGE, CLIPRANGE)
                    # Unclipped value
                    vf_losses1 = tf.square(vpred - mb_returns)
                    # Clipped value
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                    # Calculate ratio (pi current policy / pi old policy)
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    # Defining Loss = - J is equivalent to max J
                    pg_losses = -advs * ratio
                    pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                    # Final PG loss
                    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    # Total loss
                    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
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
                    'mb_obs':                            np.empty((nbatch, 4 * no_of_agents_per_env),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch,),     dtype=np.int32),
                    'mb_returns':                        np.empty((nbatch),         dtype=np.float32),
                    'mb_dones':                          np.empty((nbatch),         dtype=np.float32),
                    'mb_qvalues':                        np.empty((nbatch),         dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch,),         dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch),         dtype=np.float32),
                    'mb_qprob':                          np.empty((nbatch, 5),         dtype=np.float32),
                    'mb_all_qvalues':                    np.empty((nbatch, 5),         dtype=np.float32)
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        while cur_traj_size < nbatch:
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_returns'].shape[0]
            for k, v in trajectory.items():
                trajectory[k][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-cur_traj_size)]
            cur_traj_size += min(a_traj_size, nbatch-cur_traj_size)
        # Collect trajectories
        b_obs = trajectory['mb_obs']
        b_actions = trajectory['mb_actions']
        b_dones = trajectory['mb_dones']
        b_qvalues = trajectory['mb_qvalues']
        b_neglogpacs = trajectory['mb_neglogpacs']
        b_returns = trajectory['mb_returns']
        mb_rewards = trajectory['mb_rewards']
        b_qprob = trajectory['mb_qprob']
        b_all_qvalues = trajectory['mb_all_qvalues']
        # Start SGD and optimize model via Adam
        losses, approxkls, entropies, advss = ppo_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues)
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Send updated agent and summaries to coordinator
        print("PPO NAV:")
        print("approxkl: ", np.array(approxkls))
        print("loss: ", np.array(losses))
        print("entropy: ", np.array(entropies))
        print("return: ", np.mean(b_returns, axis=0))
        print("rewards: ", np.mean(mb_rewards, axis=0))
        print("advs: ", np.array(advss))
        print("")
        sys.stdout.flush()


if rank >= 0 and rank <= 5:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 6 and rank <= 6:
    learner()
