import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from env import Game
from network_q import Network

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
total_timesteps = int(5e6)
batch_size = nsteps = 2
no_of_agents_per_env = 1


def actor(n_agents=1):
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
    network(np.array([obs[0]]))
    network.get_tq_value(np.array([obs[0]]))
    eps_b4 = network.eps.value()
    last_action = np.array([0 for _ in range(n_agents)])
    last_eval_action = np.array([0 for _ in range(n_agents)])
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)
        network.eps.assign(eps_b4)

        mb_obs = np.zeros([nsteps, n_agents, 4*n_agents], dtype=np.float32)
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
                actions = network(np.array([obs_dc[ai]]))[0]
                agent_actions.append(actions)
                mb_last_actions[step, ai] = last_action[ai]
                last_action[ai] = actions
                eval_action = network(np.array([obs_dc[ai]]), stochastic=False)[0]
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
    gamma = 1.0
    ent_coef = 0.01
    vf_coef = 0.5
    CLIPRANGE = 0.2
    max_grad_norm = 10.0
    bptt_ts = 2 #16
    n_actors = len(actors[0])
    n_agents = no_of_agents_per_env
    agent_batch = batch_size * n_actors
    nbatch = 150
    # Build network architecture
    network = Network(nbatch)
    env = Game()
    obs = env.reset()
    network(np.array([obs[0]]))
    network.get_tq_value(np.array([obs[0]]))
    network.update_target()
    current_update_ts = 0
    all_rewards_buffer = []
    all_eval_rewards_buffer = []
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # Huber loss
    def huber_loss(x, delta=1.0):
        """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )
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
        b_obs = tf.squeeze(b_obs, axis=1)
        b_obs1 = tf.squeeze(b_obs1, axis=1)
        b_actions = tf.squeeze(b_actions, axis=1)
        b_last_actions = tf.squeeze(b_last_actions, axis=1)
        b_rewards = tf.squeeze(b_rewards, axis=1)
        b_dones = tf.squeeze(b_dones, axis=1)
        bs = b_obs.shape[0]
        # Calculate estimated Q
        with tf.GradientTape() as tape:
            qacts = network.get_q_value(b_obs)
            # Select qvals of action taken
            taken_qs = tf.reduce_sum(qacts * tf.one_hot(tf.dtypes.cast(b_actions, tf.int32), 5), -1)
            # Select target qvals
            target_qs = network.get_tq_value(b_obs1)
            # Choose max over target-qs
            target_max_qs = tf.reduce_max(target_qs, axis=-1)
            # 1-step q-learning target
            b_targets = b_rewards + gamma * (1 - b_dones) * target_max_qs
            # td error
            td_error = taken_qs - b_targets
            td_error = huber_loss(td_error)
            # loss
            weights = tf.ones_like(b_rewards)
            loss = tf.reduce_mean(weights * td_error)
            #loss = tf.reduce_mean(tf.square(td_error))
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
                    'mb_eval_rewards':                   np.empty((agent_batch, n_agents),         dtype=np.float32),
                    'mb_rewards':                        np.empty((agent_batch, n_agents),         dtype=np.float32)
                    }
        # Collect enough rollout to fill batch_size
        cur_traj_size = 0
        for _ in range(n_actors):
            a_trajectory = comm.recv()
            a_traj_size = a_trajectory['mb_obs'].shape[0]
            network.eps.assign(a_trajectory['eps'])
            trajectory['mb_eval_rewards'][cur_traj_size:min(cur_traj_size+a_traj_size, agent_batch)] = a_trajectory['mb_eval_rewards'][0:max(0, agent_batch-cur_traj_size)]
            trajectory['mb_rewards'][cur_traj_size:min(cur_traj_size+a_traj_size, agent_batch)] = a_trajectory['mb_rewards'][0:max(0, agent_batch-cur_traj_size)]
            cur_traj_size += min(a_traj_size, agent_batch-cur_traj_size)
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
        all_rewards_buffer.append(np.mean(trajectory['mb_rewards'][:, 0]))
        all_eval_rewards_buffer.append(np.mean(trajectory['mb_eval_rewards'][:, 0]))
        # Start SGD and optimize model via Adam
        if buffer.__len__() >= 1000:
            loss = q_loss()
            # Update target
            current_update_ts += 1
            if current_update_ts % 5000 == 0:
                network.update_target()
                #network.polyak_qnet()
            # Send updated agent and summaries to coordinator
            print(f"Q UPDATE NO.{current_update_ts}:")
            #print("approxkl: ", np.array(approxkls))
            print("loss: ", np.array(loss))
            #print("entropy: ", np.array(entropies))
            #print("return: ", np.mean(b_returns, axis=0))
            print("Exploratory rewards: ", np.mean(all_rewards_buffer[-1000:-1], axis=0))
            print("Deterministic rewards: ", np.mean(all_eval_rewards_buffer[-1000:-1], axis=0))
            print("")
            sys.stdout.flush()
        # Send updated agent to actor
        for actor in actors[agent_type]:
            comm.send(network.get_weights(), dest=actor)


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
