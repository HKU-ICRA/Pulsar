import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_qmix import Network

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
total_timesteps = int(1e7)
batch_size = nsteps = 128
no_of_agents_per_env = 2


def actor(n_agents=2):
    # EPS scheduler
    exploration_fraction = 0.1
    exploration_final_eps = 0.02
    class LinearSchedule:
        def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
            """Linear interpolation between initial_p and final_p over
            schedule_timesteps. After this many timesteps pass final_p is
            returned.
            Parameters
            ----------
            schedule_timesteps: int
                Number of timesteps for which to linearly anneal initial_p
                to final_p
            initial_p: float
                initial output value
            final_p: float
                final output value
            """
            self.schedule_timesteps = schedule_timesteps
            self.final_p = final_p
            self.initial_p = initial_p

        def value(self, t):
            """See Schedule.value"""
            fraction = min(float(t) / self.schedule_timesteps, 1.0)
            return self.initial_p + fraction * (self.final_p - self.initial_p)
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    # GAE hyper-parameters
    lam = 0.95
    # Setup environment
    total_ts = 0
    env = Game()
    eval_env = Game()
    obs = env.reset()
    eval_env.reset()
    dones = False
    # Build network architecture
    network = Network(1)
    network(np.array([obs[0]]), np.array([0]))
    network.get_tq_value(np.array([obs[0]]), np.array([0]))
    network.qmixer(np.zeros([n_agents, 1, 1]), np.array([obs[0]]))
    network.tqmixer(np.zeros([n_agents, 1, 1]), np.array([obs[0]]))
    eps_b4 = network.eps.value()
    last_action = np.array([0 for _ in range(n_agents)])
    last_eval_action = np.array([0 for _ in range(n_agents)])
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)
        network.eps.assign(eps_b4)

        mb_obs = np.zeros([nsteps, n_agents, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_last_actions = np.zeros([nsteps, n_agents], dtype=np.float32) 
        mb_eval_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32) 

        for step in range(nsteps):
            update_eps = exploration.value(total_ts)
            network.update_eps.assign(update_eps)
            # Get actions of training agent
            agent_actions, eval_actions = [], []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions = network(np.array([obs_dc[ai]]), np.array([last_action[ai]]))[0]
                agent_actions.append(actions)
                mb_last_actions[step, ai] = last_action[ai]
                last_action[ai] = actions
                eval_action = network(np.array([obs_dc[ai]]), np.array([last_eval_action[ai]]), stochastic=False)[0]
                eval_actions.append(eval_action)
                last_eval_action[ai] = eval_action
                mb_obs[step, ai] = obs_dc[ai]
                mb_actions[step, ai] = actions
            obs, rewards, dones = env.step(np.array(agent_actions))
            _, eval_reward, eval_dones = eval_env.step(np.array(eval_actions))
            # Handle rewards
            mb_dones[step, :] = np.array([dones for _ in range(n_agents)])
            mb_rewards[step, :] = np.array([[rewards] for _  in range(n_agents)])
            mb_eval_rewards[step, :] = np.array([[eval_reward] for _  in range(n_agents)])
            if dones:
                last_action = np.array([0 for _ in range(n_agents)])
                obs = env.reset()
            if eval_dones:
                last_eval_action = np.array([0 for _ in range(n_agents)])
                eval_env.reset()
            total_ts += 1
        # Send trajectory to learner
        eps_b4 = network.eps.value()
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_eval_rewards = np.squeeze(mb_eval_rewards, axis=-1)
        trajectory = {
                    'mb_obs': mb_obs,
                    'mb_actions': mb_actions,
                    'mb_last_actions': mb_last_actions,
                    'mb_dones': mb_dones,
                    'mb_rewards': mb_rewards,
                    'mb_eval_rewards': mb_eval_rewards,
                    'eps': network.eps.value()
                    }
        comm.send(trajectory, dest=learners[agent_type])
        

def learner():
    # Learner hyperparameters
    gamma = 0.99
    ent_coef = 0.01
    vf_coef = 0.5
    CLIPRANGE = 0.2
    max_grad_norm = 5.0
    noptepochs = 4
    bptt_ts = 2 #16
    n_actors = len(actors[0])
    n_agents = no_of_agents_per_env
    nbatch = batch_size * n_actors
    nbatch_steps = nbatch // 4
    # Build network architecture
    network = Network(nbatch_steps)
    env = Game()
    obs = env.reset()
    network(np.array([obs[0]]), np.array([0]))
    network.get_tq_value(np.array([obs[0]]), np.array([0]))
    network.qmixer(np.zeros([n_agents, 1, 1]), np.array([obs[0]]))
    network.tqmixer(np.zeros([n_agents, 1, 1]), np.array([obs[0]]))
    network.update_target()
    current_update_ts = 0
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # ReplayBuffer
    class ReplayBuffer:
        def __init__(self, size):
            """Create Replay buffer.
            Parameters
            ----------
            size: int
                Max number of transitions to store in the buffer. When the buffer
                overflows the old memories are dropped.
            """
            self._storage = []
            self._maxsize = size
            self._next_idx = 0

        def __len__(self):
            return len(self._storage)

        def add(self, obs_t, action, last_action, reward, obs_tp1, done):
            data = (obs_t, action, last_action, reward, obs_tp1, done)

            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

        def _encode_sample(self, idxes):
            obses_t, actions, last_actions, rewards, obses_tp1, dones = [], [], [], [], [], []
            for i in idxes:
                data = self._storage[i]
                obs_t, action, last_action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                last_actions.append(np.array(last_action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
            return np.array(obses_t), np.array(actions), np.array(last_actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

        def sample(self, batch_size):
            """Sample a batch of experiences.
            Parameters
            ----------
            batch_size: int
                How many transitions to sample.
            Returns
            -------
            obs_batch: np.array
                batch of observations
            act_batch: np.array
                batch of actions executed given obs_batch
            rew_batch: np.array
                rewards received as results of executing act_batch
            next_obs_batch: np.array
                next set of observations seen after executing act_batch
            done_mask: np.array
                done_mask[i] = 1 if executing act_batch[i] resulted in
                the end of an episode and 0 otherwise.
            """
            idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            return self._encode_sample(idxes)
    buffer = ReplayBuffer(50000)
    # PPO RL optimization loss function
    @tf.function
    def q_loss():
        b_obs, b_actions, b_last_actions, b_rewards, b_obs1, b_dones = buffer.sample(nbatch)
        b_obs = tf.transpose(b_obs, [1, 0, 2])
        b_obs1 = tf.transpose(b_obs1, [1, 0, 2])
        b_actions = tf.transpose(b_actions, [1, 0])
        b_last_actions = tf.transpose(b_last_actions, [1, 0])
        b_rewards = tf.transpose(b_rewards, [1, 0])
        b_dones = tf.transpose(b_dones, [1, 0])
        bs = b_obs.shape[1]
        # Calculate estimated Q
        with tf.GradientTape() as tape:
            qacts = []
            for ai in range(n_agents):
                #for t in range(bs):
                #    if t == 0:
                #        last_action = np.array([0])
                #    else:
                #        last_action = b_actions[ai, t-1:t]
                agent_outs = network.get_q_value(b_obs[ai], tf.dtypes.cast(b_last_actions[ai], tf.int32))
                qacts.append(agent_outs)
            qacts = tf.stack(qacts, axis=0)
            # Select qvals of action taken
            taken_qs = []
            for ai in range(n_agents):
                batch_idxs = tf.range(qacts[ai].shape[0])
                taken_qs_idxs = tf.concat([tf.expand_dims(batch_idxs, axis=1),
                                        tf.expand_dims(tf.dtypes.cast(b_actions[ai], tf.int32), axis=1)],
                                        axis=-1)
                taken_qs.append(tf.gather_nd(qacts[ai], taken_qs_idxs))
            taken_qs = tf.stack(taken_qs, axis=0)
            # Select target qvals
            target_qacts = [[] for _ in range(n_agents)]
            for ai in range(n_agents):
                #for t in range(bs):
                #    last_action = b_actions[ai, t:t+1]
                agent_outs = network.get_tq_value(b_obs1[ai], tf.dtypes.cast(b_actions[ai], tf.int32))
                target_qacts[ai].append(agent_outs)
            target_qs = tf.stack(target_qacts, axis=0)
            target_qs = tf.squeeze(target_qs, axis=1)
            # Choose max over target-qs
            target_max_qs = tf.reduce_max(target_qs, axis=-1)
            # Mix
            taken_qs = tf.expand_dims(taken_qs, axis=2)
            chosen_action_qvals = network.qmixer(taken_qs, b_obs[0])
            target_max_qs = tf.expand_dims(target_max_qs, axis=2)
            target_max_qvals = network.tqmixer(target_max_qs, b_obs1[0])
            # 1-step q-learning target
            b_targets = b_rewards[0] + gamma * (1 - b_dones[0]) * target_max_qvals
            # td error
            td_error = chosen_action_qvals - b_targets
            # loss
            loss = tf.reduce_mean(tf.square(td_error))
            # train
            var = network.trainable_variables
            grads = tape.gradient(loss, var)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_var = zip(grads, var)
            network.optimizer.apply_gradients(grads_and_var)
        return loss
    # Start learner process loop
    while True:
        trajectory = {
                    'mb_eval_rewards':                   np.empty((nbatch, n_agents),         dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch, n_agents),         dtype=np.float32)
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        for _ in range(n_actors):
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_obs'].shape[0]
            network.eps.assign(a_trajectory['eps'])
            trajectory['mb_eval_rewards'][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory['mb_eval_rewards'][0:max(0, nbatch-cur_traj_size)]
            trajectory['mb_rewards'][cur_traj_size:min(cur_traj_size+a_traj_size, nbatch)] = a_trajectory['mb_rewards'][0:max(0, nbatch-cur_traj_size)]
            cur_traj_size += min(a_traj_size, nbatch-cur_traj_size)
            for t in range(a_traj_size - 1):
                buffer.add(a_trajectory['mb_obs'][t],
                           a_trajectory['mb_actions'][t],
                           a_trajectory['mb_last_actions'][t],
                           a_trajectory['mb_rewards'][t],
                           a_trajectory['mb_obs'][t+1],
                           a_trajectory['mb_dones'][t])
        # Collect trajectories
        #b_obs = trajectory['mb_obs']
        #b_actions = trajectory['mb_actions'][:, :-1]
        #b_dones = trajectory['mb_dones'][:, :-1]
        b_rewards = trajectory['mb_rewards'][:, 0]
        b_eval_rewards = trajectory['mb_eval_rewards'][:, 0]
        # Start SGD and optimize model via Adam
        loss = q_loss()
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)
        # Update target
        current_update_ts += 1
        if current_update_ts % 50 == 0:
            network.update_target()
            #network.polyak_qnet()
        # Send updated agent and summaries to coordinator
        print(f"Q UPDATE NO.{current_update_ts}:")
        #print("approxkl: ", np.array(approxkls))
        print("loss: ", np.array(loss))
        #print("entropy: ", np.array(entropies))
        #print("return: ", np.mean(b_returns, axis=0))
        print("Exploratory rewards: ", np.mean(b_rewards, axis=0))
        print("Deterministic rewards: ", np.mean(b_eval_rewards, axis=0))
        print("")
        sys.stdout.flush()


if rank >= 0 and rank <= 5:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 6 and rank <= 6:
    learner()



"""
def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_qvalues, b_neglogpacs, b_qprob, b_all_qvalues):
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
                mb_returns = tf.gather(b_returns, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_qvalues = tf.gather(b_qvalues, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_qprob = tf.gather(b_qprob, mbinds)
                mb_all_qvalues = tf.gather(b_all_qvalues, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, p_q, p_probs, p_p = network(mb_obs)
                    # Calculate neglogpac
                    neglogpacs = []
                    for qi in range(4):
                        neglogpacs.append(tf.expand_dims(network.get_neglogp(p_p, np.array([qi])), axis=-1))
                    neglogpac = tf.concat(neglogpacs, axis=-1)
                    # Calculate ratio (pi current policy / pi old policy)
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    b_sample_sum = mb_all_qvalues[:, 0] + mb_all_qvalues[:, 1] + mb_all_qvalues[:, 2] + mb_all_qvalues[:, 3]
                    pg_loss = 0
                    for a in range(4):
                        advs = mb_all_qvalues[:, a] - b_sample_sum
                        advs = tf.maximum(advs, 0)
                        advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                        pg_losses = -advs * ratio[:, a]
                        pg_losses2 = -advs * tf.clip_by_value(ratio[:, a], 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                        pg_loss += tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    pg_loss /= 4
                    # Batch normalize the advantages
                    # Calculate the entropy
                    # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                    entropy = tf.reduce_mean(p_entropy)
                    # CALCULATE THE LOSS
                    # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
                    # Clip the value to reduce variability during Critic training
                    # Get the predicted value
                    vpredclipped = mb_qvalues + tf.clip_by_value(p_q - mb_qvalues, -CLIPRANGE, CLIPRANGE)
                    # Unclipped value
                    vf_losses1 = tf.square(p_q - mb_returns)
                    # Clipped value
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
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
"""
