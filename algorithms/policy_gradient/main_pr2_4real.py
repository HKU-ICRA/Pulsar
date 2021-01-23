import os
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy
import functools

from env import Game
from network_pr2_4real import Network


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
batch_size = nsteps = 20
no_of_agents_per_env = 2


def actor(n_agents=2):
    # GAE hyper-parameters
    lam = 0.95
    gamma = 0.99
    value_n_particles = 16
    # Setup environment
    env = Game()
    obs = env.reset()
    dones = False
    # Build network architecture
    network = Network(1)
    network(np.array([obs[0]]))
    network.get_ocp(np.array([obs[0]]), np.array([0]))
    network.get_q_value(np.array([obs[0]]), np.array([0]))
    network.get_jq_value(np.array([obs[0]]), np.array([0]), np.array([[0, 0, 0, 0, 0]]).astype(np.float32))
    network.get_tjq_value(np.array([obs[0]]), np.array([0]), np.array([[0, 0, 0, 0, 0]]).astype(np.float32))
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    while True:
        weights = comm.recv(source=learners[agent_type])
        network.set_weights(weights)

        mb_obs = np.zeros([nsteps, n_agents, 4 * n_agents], dtype=np.float32)
        mb_rewards = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_qvalues = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_jqvalues = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_values = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_all_qvalues = np.zeros([nsteps, n_agents, 5], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, n_agents, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_qdones = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_actions = np.zeros([nsteps, n_agents], dtype=np.float32)
        mb_other_actions = np.zeros([nsteps, n_agents, value_n_particles, 5], dtype=np.float32)
        mb_qprob = np.zeros([nsteps, n_agents, 5], dtype=np.float32)

        for step in range(nsteps):
            # Get actions of training agent
            agent_actions = []
            for ai in range(n_agents):
                obs_dc = deepcopy(obs)
                actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
                mb_values[step, ai] = network.get_value(np.array([obs_dc[ai]]))
                mb_qvalues[step, ai] = network.get_q_value(np.array([obs_dc[ai]]), np.array([actions[0]]))
                mb_neglogpacs[step, ai] = neglogp
                mb_qprob[step, ai] = probs
                mb_dones[step, ai] = dones
                mb_actions[step, ai] = actions
                mb_obs[step, ai] = obs_dc[ai]
                agent_actions.append(actions[0])
            for ai in range(n_agents):
                other_ai = 1 - ai
                jq_value = 0.0
                opponent_actions = network.get_ocp(np.array([obs_dc[ai]]), np.array([actions[0]]), value_n_particles)
                for idx in range(value_n_particles):
                    jq_value += network.get_jq_value(np.array([obs_dc[ai]]), np.array([agent_actions[ai]]), opponent_actions[:, idx])
                jq_value /= value_n_particles
                mb_jqvalues[step, ai] = jq_value
                mb_other_actions[step, ai] = tf.one_hot(agent_actions[other_ai], 5)
            agent_actions = np.array(agent_actions)
            obs, rewards, dones = env.step(agent_actions)
            # Handle rewards
            mb_rewards[step, :] = np.array([[rewards] for _ in range(n_agents)])
            mb_qdones[step, :] = np.array([dones for _ in range(n_agents)])
            if dones:
                obs = env.reset()
        # Get last values
        mb_next_obs = np.zeros_like(mb_obs)
        mb_next_obs[0:-1] = mb_obs[1:]
        mb_next_actions = np.zeros_like(mb_actions)
        mb_next_actions[0:-1] = mb_actions[1:]
        for ai in range(n_agents):
            obs_dc = deepcopy(obs)
            actions, neglogp, entropy, probs, p = network(np.array([obs_dc[ai]]))
            mb_next_actions[-1, ai] = actions
            mb_next_obs[-1, ai] = obs_dc[ai]
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        # perform GAE calculation
        for ai in range(n_agents):
            other_ai = 1 - ai
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    nextnonterminal = 1.0 - dones
                    obs_dc = deepcopy(obs)
                    last_values = network.get_value(np.array([obs_dc[ai]]))
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1, ai]
                    last_values = mb_values[t+1, ai]
                mb_returns[t, ai] = mb_rewards[t, ai] + nextnonterminal * gamma * last_values
            for t in range(nsteps):
                mb_advs[t, ai] = mb_jqvalues[t, ai] - mb_values[t, ai]
        # Send trajectory to learner
        mb_qvalues = np.squeeze(mb_qvalues, axis=-1)
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_advs = np.squeeze(mb_advs, axis=-1)
        mb_values = np.squeeze(mb_values, axis=-1)
        trajectory = {
                    'mb_obs': mb_obs,
                    'mb_next_obs': mb_next_obs,
                    'mb_next_actions': mb_next_actions,
                    'mb_other_actions': mb_other_actions,
                    'mb_actions': mb_actions,
                    'mb_dones': mb_dones,
                    'mb_qdones': mb_qdones,
                    'mb_qvalues': mb_qvalues,
                    'mb_neglogpacs': mb_neglogpacs,
                    'mb_rewards': mb_rewards,
                    'mb_qprob': mb_qprob,
                    'mb_all_qvalues': mb_all_qvalues,
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
    network = Network(1)
    network(np.array([obs[0]]))
    network.get_ocp(np.array([obs[0]]), np.array([0]))
    network.get_q_value(np.array([obs[0]]), np.array([0]))
    network.get_jq_value(np.array([obs[0]]), np.array([0]), np.array([[0, 0, 0, 0, 0]]).astype(np.float32))
    network.get_tjq_value(np.array([obs[0]]), np.array([0]), np.array([[0, 0, 0, 0, 0]]).astype(np.float32))
    # Receive agent from coordinator and set weights
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(network.get_weights(), dest=actor)
    # Kernel func
    def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
        """Gaussian kernel with dynamic bandwidth.
        The bandwidth is adjusted dynamically to match median_distance / log(Kx).
        See [2] for more information.
        Args:
            xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
                particles of dimension D. This is the first kernel argument.
            ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
                particles of dimension D. This is the second kernel argument.
            h_min(`float`): Minimum bandwidth.
        Returns:
            `dict`: Returned dictionary has two fields:
                'output': A `tf.Tensor` object of shape (N x Kx x Ky) representing
                    the kernel matrix for inputs `xs` and `ys`.
                'gradient': A 'tf.Tensor` object of shape (N x Kx x Ky x D)
                    representing the gradient of the kernel with respect to `xs`.
        Reference:
            [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
                Purpose Bayesian Inference Algorithm," Neural Information Processing
                Systems (NIPS), 2016.
        """
        Kx, D = xs.get_shape().as_list()[-2:]
        Ky, D2 = ys.get_shape().as_list()[-2:]
        assert D == D2
        leading_shape = tf.shape(xs)[:-2]
        # Compute the pairwise distances of left and right particles.
        diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
        # ... x Kx x Ky x D
        dist_sq = tf.reduce_sum(diff**2, axis=-1, keepdims=False)
        # ... x Kx x Ky
        # Get median.
        input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
        values, _ = tf.nn.top_k(
            input=tf.reshape(dist_sq, input_shape),
            k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
            sorted=True)  # ... x floor(Ks*Kd/2)
        medians_sq = values[..., -1]  # ... (shape) (last element is the median)
        h = medians_sq / np.log(Kx)  # ... (shape)
        h = tf.maximum(h, h_min)
        h = tf.stop_gradient(h)  # Just in case.
        h_expanded_twice = tf.expand_dims(tf.expand_dims(h, -1), -1)
        # ... x 1 x 1
        kappa = tf.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky
        # Construct the gradient
        h_expanded_thrice = tf.expand_dims(h_expanded_twice, -1)
        # ... x 1 x 1 x 1
        kappa_expanded = tf.expand_dims(kappa, -1)  # ... x Kx x Ky x 1
        kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
        # ... x Kx x Ky x D
        return {"output": kappa, "gradient": kappa_grad}
    # Opponent conditional policy loss function
    kernel_n_particles = 32
    kernel_update_ratio = 0.5
    value_n_particles = 16
    annealing = 1.0
    EPS = 1e-6
    ocp_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
    @tf.function
    def ocp_loss(b_obs, b_actions):
        actions = network.get_ocp(b_obs, b_actions, n_action_samples=kernel_n_particles)
        n_updated_actions = int(kernel_n_particles * kernel_update_ratio)
        n_fixed_actions = kernel_n_particles - n_updated_actions
        fixed_actions, updated_actions = tf.split(actions, [n_fixed_actions, n_updated_actions], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        svgd_target_values = []
        for idx in range(n_fixed_actions):
            svgd_target_value = network.get_jq_value(b_obs, b_actions, fixed_actions[:, idx])
            svgd_target_values.append(svgd_target_value)
        svgd_target_values = tf.stack(svgd_target_values, axis=1)
        baseline_ind_q = network.get_q_value(b_obs, b_actions)
        baseline_ind_q = tf.tile(tf.reshape(baseline_ind_q, [-1, 1]), [1, n_fixed_actions])
        svgd_target_values = (svgd_target_values - baseline_ind_q) / annealing
        squash_correction = tf.reduce_sum(tf.math.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction
        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        kernel_dict = adaptive_isotropic_gaussian_kernel(xs=fixed_actions, ys=updated_actions)
        kappa = tf.expand_dims(kernel_dict["output"], axis=3)
        action_gradients = tf.reduce_mean(kappa * grad_log_p + kernel_dict["gradient"], axis=1)
        ocp_vars = network.ocp.trainable_variables
        gradients = tf.gradients(updated_actions, ocp_vars, grad_ys=action_gradients)
        surrogate_loss = -tf.reduce_sum([tf.reduce_sum(w * tf.stop_gradient(g)) for w, g in zip(ocp_vars, gradients)])
        return surrogate_loss
    # Q loss function
    @tf.function
    def q_loss(b_obs, b_next_obs, b_actions, b_next_actions, b_other_actions, b_qdones, b_rewards):
        opponent_target_actions = tf.random.uniform((b_obs.shape[0], value_n_particles, 5), * (-1., 1.))
        opponent_target_actions = tf.nn.softmax(opponent_target_actions, axis=-1)
        q_value_targets = []
        q_values = []
        for idx in range(value_n_particles):
            q_value_target = network.get_tjq_value(b_next_obs, b_next_actions, opponent_target_actions[:, idx])
            q_value_targets.append(q_value_target)
            q_value = network.get_jq_value(b_obs, b_actions, tf.dtypes.cast(b_other_actions[:, idx], tf.float32))
            q_values.append(q_value)
        q_value_targets = tf.stack(q_value_targets, axis=1)
        q_values = tf.stack(q_values, axis=1)
        q_values = tf.reduce_mean(q_values, axis=-1)
        next_value = annealing * tf.reduce_logsumexp(q_value_targets / annealing, axis=1)
        next_value -= tf.math.log(tf.cast(value_n_particles, tf.float32))
        next_value += (5) * np.log(2)
        ys = tf.stop_gradient(b_rewards + (1 - b_qdones) * 0.99 * next_value)
        bellman_residual = 0.5 * tf.reduce_mean((ys - q_values)**2)
        jq_vars = [var for var in network.trainable_variables if "joint_qvalue" in var.name]
        ind_q_values = network.get_q_value(b_obs, b_actions)
        ind_bellman_residual = 0.5 * tf.reduce_mean((ys - ind_q_values) ** 2)
        q_vars = [var for var in network.trainable_variables if "qvalue" in var.name]
        return bellman_residual + ind_bellman_residual
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_obs, b_actions, b_returns, b_dones, b_neglogpacs, b_advs, b_rewards, b_values):
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
                advs = tf.gather(b_advs, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_rewards = tf.gather(b_rewards, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, p_probs, p_p, taken_action_neglogp = network(mb_obs, [mb_actions])
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    neglogpac = taken_action_neglogp
                    entropy = tf.reduce_mean(p_entropy)
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
                    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
                # 1. Get the model parameters
                var = network.trainable_variables
                var = [v for v in var if "policy" or "value" in v.name]
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
                    'mb_next_obs':                            np.empty((nbatch, n_agents, n_agents * 4),     dtype=np.float32),
                    'mb_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_next_actions':                        np.empty((nbatch, n_agents),                   dtype=np.int32),
                    'mb_other_actions':                        np.empty((nbatch, n_agents, value_n_particles, 5),                   dtype=np.int32),
                    'mb_dones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_qdones':                          np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_qvalues':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_neglogpacs':                     np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_rewards':                        np.empty((nbatch, n_agents),                   dtype=np.float32),
                    'mb_qprob':                          np.empty((nbatch, n_agents, 5),                dtype=np.float32),
                    'mb_all_qvalues':                    np.empty((nbatch, n_agents, 5),                dtype=np.float32),
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
            other_ai = 1 - ai
            b_obs = trajectory['mb_obs'][:, ai]
            b_next_obs = trajectory['mb_next_obs'][:, ai]
            b_actions = trajectory['mb_actions'][:, ai]
            b_next_actions = trajectory['mb_next_actions'][:, ai]
            b_other_actions = trajectory['mb_other_actions'][:, ai]
            b_dones = trajectory['mb_dones'][:, ai]
            b_qdones = trajectory['mb_qdones'][:, ai]
            b_qvalues = trajectory['mb_qvalues'][:, ai]
            b_neglogpacs = trajectory['mb_neglogpacs'][:, ai]
            b_advs = trajectory['mb_advs'][:, ai]
            b_rewards = trajectory['mb_rewards'][:, ai]
            b_qprob = trajectory['mb_qprob'][:, ai]
            b_all_qvalues = trajectory['mb_all_qvalues'][:, ai]
            b_returns = trajectory['mb_returns'][:, ai]
            b_values = trajectory['mb_values'][:, ai]
            # OCP
            ocp_loss_val = functools.partial(ocp_loss, b_obs=b_obs, b_actions=b_actions)
            ocp_vars = network.ocp.trainable_variables
            ocp_optimizer.minimize(ocp_loss_val, ocp_vars)
            # Q
            q_loss_val = functools.partial(q_loss, b_obs=b_obs, b_next_obs=b_next_obs, b_actions=b_actions,
                                                   b_next_actions=b_next_actions, b_other_actions=b_other_actions,
                                                   b_qdones=b_qdones, b_rewards=b_rewards)
            q_vars = [var for var in network.trainable_variables if "qvalue" in var.name]
            q_optimizer.minimize(q_loss_val, q_vars)
            # PPO
            losses, approxkls, entropies = ppo_loss(b_obs, b_actions,
                                                    b_returns, b_dones,
                                                    b_neglogpacs, b_advs,
                                                    b_rewards, b_values)
        network.polyak_qnet()
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
