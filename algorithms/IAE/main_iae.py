import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_iae import Network


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
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_obs = np.zeros([nsteps, n_agents, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_values = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_ivalues = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_rews = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_probs = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
        mb_dependencies = np.zeros([nsteps, n_agents, 5], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions = []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                mb_values[step, ai] = network.get_value(np.array([obs_dc[ai]]))
                mb_ivalues[step, ai] = network.get_ivalue(np.array([obs_dc[ai]]), np.array([actions[0]]))
                mb_neglogpacs[step, ai] = neglogp
                mb_dones[step, ai] = dones
                mb_actions[step, ai] = actions
                mb_obs[step, ai] = obs_dc[ai]
                mb_probs[step, ai] = probs
                agent_actions.append(actions[0])
            agent_actions = np.array(agent_actions)
            obs, rewards, dones = env.step(agent_actions)
            # Handle rewards
            mb_rewards[step, :] = np.array([[rewards] for _ in range(n_agents)])
            if dones:
                obs = env.reset()
        # Get last values
        last_values, last_probs, last_actions = [], [], []
        for ai in range(n_agents):
            last_action, _, _, last_prob, _ = network(np.array([obs_dc[ai]]))
            last_value = network.get_value(np.array([obs_dc[ai]]))
            last_values.append(last_value)
            last_probs.append(last_prob)
            last_actions.append(last_action[0])
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_ireturns = np.zeros_like(mb_rewards)
        mb_RD = np.zeros_like(mb_rewards)
        mb_J = np.zeros_like(mb_rewards)
        mb_C = np.zeros_like(mb_rewards)
        mb_gadvs = np.zeros_like(mb_rewards)
        mb_iadvs = np.zeros_like(mb_rewards)
        mb_dependency_td = np.zeros([nsteps * nsteps, n_agents, 1])
        mb_dependency_target = np.zeros([nsteps * nsteps, n_agents, 1])
        lastgaelam = 0
        # perform GAE calculation
        for ai in range(n_agents):
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    # GAE
                    nextnonterminal = 1.0 - dones
                    nextvalues = last_values[ai]
                    # REW
                    obs_dc = deepcopy(obs)
                    obs1 = mb_obs[t:t+1, ai]
                    obs2 = np.array([obs_dc[ai]])
                    # J
                    jobs1 = np.array([obs_dc[ai]])
                    jenv = deepcopy(env)
                    jagent_actions = []
                    for ai in range(n_agents):
                        obs_dc = deepcopy(obs)
                        jactions, _, _, _, _ = network(np.array([obs_dc[ai]]))
                        jagent_actions.append(jactions[0])
                    jagent_actions = np.array(jagent_actions)
                    jobs2, _, _ = jenv.step(jagent_actions)
                    jobs2 = np.array([jobs2[ai]])
                    jdelta = np.array(rewards) + gamma * nextvalues * nextnonterminal - mb_values[t, ai]
                    jgadv = jdelta + gamma * lam * nextnonterminal * lastgaelam
                    last_network_rew = network.get_rew(jobs1, jobs2, np.array([[rewards]]).astype(np.float32), np.array([jgadv]), np.array([[1]]))
                    last_J = np.array(rewards) - last_network_rew
                    last_C = (network.get_dependency(jobs1, jobs2, np.array([[1]]))[0, last_actions[0]] / last_probs[ai][0, last_actions[0]]) - 1.0
                    last_ireturns = last_network_rew * last_C
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1, ai]
                    nextvalues = mb_values[t+1, ai]
                    obs1 = mb_obs[t:t+1, ai]
                    obs2 = mb_obs[t+1:t+2, ai]
                    last_J = mb_J[t+1:t+2, ai]
                mb_dependencies[t, ai] = network.get_dependency(obs1, obs2, np.array([[1]]))
                action_taken = int(mb_actions[t, ai])
                mb_C[t, ai] = (mb_dependencies[t, ai, action_taken] / mb_probs[t, ai, action_taken]) - 1.0
                delta = mb_rewards[t, ai] + gamma * nextvalues * nextnonterminal - mb_values[t, ai]
                mb_gadvs[t, ai] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                mb_RD[t, ai] = network.get_rew(obs1, obs2, mb_rewards[t:t+1, ai], mb_gadvs[t:t+1, ai], np.array([[1]]))
                mb_J[t, ai] =  (mb_rewards[t, ai] - mb_RD[t, ai]) + lam * last_J
                mb_iadvs[t, ai] = mb_J[t, ai] - mb_values[t, ai] + mb_ivalues[t, ai]
                mb_ireturns[t, ai] = (mb_RD[t, ai] * mb_C[t, ai]) + lam * last_ireturns
            mb_returns[:, ai] = mb_gadvs[:, ai] + mb_values[:, ai]
        for ai in range(n_agents):
            for t1 in range(nsteps):
                for t2 in range(t1 + 1, nsteps):
                    k = t2 - t1
                    mb_dependency_td = network.get_dependency(obs1, obs2, np.array([[k]])) + 0.99 * ()
            for t1 in range(nsteps):
                for t2 in range(t1 + 1, nsteps):
                    td_targ = 
                    k = t2 - t1
                    mb_dependency_target[t1 * nsteps + t2, ai] = (0.85**k) * mb_probs[t1, ai, mb_actions[t1, ai]] + (1 - 0.85**k) * td_targ
        # Send trajectory to learner
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_advs = np.squeeze(mb_iadvs, axis=-1)
        mb_values = np.squeeze(mb_values, axis=-1)
        trajectory = {
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
                    'mb_dones': mb_dones,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_rewards': mb_rewards,
                    'mb_advs': mb_advs,
                    'mb_returns': mb_returns,
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
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_neglogpacs, b_rewards, b_values):
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
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, vpred, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    advs = mb_returns - mb_values
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
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
        return losses_total, approxkls, entropies
    # Start learner process loop
    while True:
        trajectory = {
                    'mb_obs':                            np.empty((nbatch, n_agents, n_agents * 4),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_advs':                           np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_returns':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
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
            b_obs = trajectory['mb_obs'][:, ai]
            b_actions = trajectory['mb_actions'][:, ai]
            b_dones = trajectory['mb_dones'][:, ai]
            b_neglogpacs = trajectory['mb_neglogpacs'][:, ai]
            mb_rewards = trajectory['mb_rewards'][:, ai]
            b_returns = trajectory['mb_returns'][:, ai]
            b_values = trajectory['mb_values'][:, ai]
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss(b_obs, b_actions,
                                                    b_returns, b_dones,
                                                    b_neglogpacs,
                                                    mb_rewards, b_values)
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
