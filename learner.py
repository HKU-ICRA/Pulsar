import sys
from mpi4py import MPI
import tensorflow as tf
import numpy as np

from architecture.pulsar import Pulsar


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
agent_index = int(sys.argv[1])

# Learner hyperparameters
ent_coef = 0.01
vf_coef = 0.5
CLIPRANGE = 0.2
max_grad_norm = 0.5
noptepochs = 4
batch_size = 30
nbatch_steps = 15   # Should be a multiple of batch_size - bptt_ts and is smaller than batch_size - bptt_ts
bptt_ts = 15
# Training parameters
actor_lower_bound = int(sys.argv[2])
actor_upper_bound = int(sys.argv[3])
# Build network architecture
pulsar = Pulsar(training=True)
pulsar.call_build()
# Receive agent from coordinator and set weights
agent = comm.recv()
pulsar.set_weights(agent.get_weights())
# Send agent to actor
for idx in range(actor_lower_bound, actor_upper_bound):
    MPI.COMM_WORLD.send(agent, dest=idx)

while True:
    trajectory = {'mb_scalar_features': {'match_time': np.empty((batch_size, 1), dtype=np.float32),
                                         'bptt_match_time': np.empty((batch_size, 1, 1), dtype=np.float32)},
                  'mb_entities': np.empty((batch_size, 1, 5, 4), dtype=np.float32),
                  'mb_entity_masks': np.empty((batch_size, 5), dtype=np.float32),
                  'mb_baselines': np.empty((batch_size, 18), dtype=np.float32),
                  'mb_actions_xy': np.empty((batch_size, 2), dtype=np.float32),
                  'mb_actions_yaw': np.empty((batch_size, 1), dtype=np.float32),
                  'mb_returns': np.empty((batch_size), dtype=np.float32),
                  'mb_dones': np.empty((batch_size), dtype=np.float32),
                  'mb_values': np.empty((batch_size), dtype=np.float32),
                  'mb_neglogpacs_xy': np.empty((batch_size), dtype=np.float32),
                  'mb_neglogpacs_yaw': np.empty((batch_size), dtype=np.float32),
                  'mb_states': np.empty((batch_size, 3, 2, 1, 384), dtype=np.float32)}
    
    # Collect enough rollout to fill batch_size
    agent.add_steps(batch_size)
    cur_traj_size = 0
    for idx in range(actor_lower_bound, actor_upper_bound):
        a_trajectory = MPI.COMM_WORLD.recv()
        a_traj_size = a_trajectory['mb_returns'].shape[0]
        for k, v in trajectory.items():
            if k == 'mb_scalar_features':
                trajectory[k]['match_time'][cur_traj_size:min(cur_traj_size+a_traj_size, batch_size)] = a_trajectory[k]['match_time'][0:max(0, batch_size-cur_traj_size)]
                trajectory[k]['bptt_match_time'][cur_traj_size:min(cur_traj_size+a_traj_size, batch_size)] = a_trajectory[k]['bptt_match_time'][0:max(0, batch_size-cur_traj_size)]
            else:
                trajectory[k][cur_traj_size:min(cur_traj_size+a_traj_size, batch_size)] = a_trajectory[k][0:max(0, batch_size-cur_traj_size)]
        
    # Setup BPTT and SGD
    b_scalar_features = trajectory['mb_scalar_features']
    b_entities = trajectory['mb_entities']
    new_b_scalar_features = {'match_time': [], 'bptt_match_time': []}
    new_b_entities = []
    for idx in range(bptt_ts, batch_size):
        new_b_scalar_features['match_time'].append(b_scalar_features['match_time'][idx])
        new_b_scalar_features['bptt_match_time'].append(b_scalar_features['bptt_match_time'][idx - bptt_ts:idx, 0])
        new_b_entities.append(b_entities[idx - bptt_ts:idx, 0])
    b_scalar_features['match_time'] = np.array(new_b_scalar_features['match_time'])
    b_scalar_features['bptt_match_time'] = np.array(new_b_scalar_features['bptt_match_time'])
    b_entities = np.array(new_b_entities)
    b_entity_masks = trajectory['mb_entity_masks'][bptt_ts:batch_size]
    b_baselines = trajectory['mb_baselines'][bptt_ts:batch_size]
    b_actions_xy = trajectory['mb_actions_xy'][bptt_ts:batch_size]
    b_actions_yaw = trajectory['mb_actions_yaw'][bptt_ts:batch_size]
    b_returns = trajectory['mb_returns'][bptt_ts:batch_size]
    b_dones = trajectory['mb_dones'][bptt_ts:batch_size]
    b_values = trajectory['mb_values'][bptt_ts:batch_size]
    b_neglogpacs_xy = trajectory['mb_neglogpacs_xy'][bptt_ts:batch_size]
    b_neglogpacs_yaw = trajectory['mb_neglogpacs_yaw'][bptt_ts:batch_size]
    b_states = trajectory['mb_states'][0:batch_size-bptt_ts]

    # Start SGD
    nbatch = b_returns.shape[0]
    inds = np.arange(nbatch)
    for _ in range(noptepochs):
        np.random.shuffle(inds)
        for start in range(0, nbatch, nbatch_steps):
            end = start + nbatch_steps
            mbinds = inds[start:end]

            mb_scalar_features = dict()
            mb_scalar_features['match_time'] = b_scalar_features['match_time'][mbinds]
            mb_scalar_features['bptt_match_time'] = b_scalar_features['bptt_match_time'][mbinds]
            mb_entities = b_entities[mbinds]
            mb_entity_masks = b_entity_masks[mbinds]
            mb_baselines = b_baselines[mbinds]
            mb_actions_xy = b_actions_xy[mbinds]
            mb_actions_yaw = b_actions_yaw[mbinds]
            mb_returns = b_returns[mbinds]
            mb_dones = b_dones[mbinds]
            mb_values = b_values[mbinds]
            mb_neglogpacs_xy = b_neglogpacs_xy[mbinds]
            mb_neglogpacs_yaw = b_neglogpacs_yaw[mbinds]
            mb_neglogpacs = mb_neglogpacs_xy + mb_neglogpacs_yaw
            mb_states = b_states[mbinds]

            # Setup states
            mb_states = np.concatenate(mb_states[:], axis=2)
            mb_states = mb_states.tolist()
            for i in range(len(mb_states)):
                for j in range(len(mb_states[i])):
                    mb_states[i][j] = tf.convert_to_tensor(mb_states[i][j])

            with tf.GradientTape() as tape:
                p_actions, p_neglogp, p_entropy, p_mean, vpred, p_states, p_prev_states = pulsar(mb_scalar_features, mb_entities, mb_entity_masks, mb_baselines, mb_states)

                advs = mb_returns - mb_values
                # Batch normalize the advantages
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                # Calculate neglogpac
                neglogpac_xy = pulsar.neglogp_xy(p_mean['xyvel'], mb_actions_xy)
                neglogpac_yaw = pulsar.neglogp_xy(p_mean['yaw'], mb_actions_yaw)
                neglogpac = neglogpac_xy + neglogpac_yaw
                # Calculate the entropy
                # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                entropy = tf.reduce_mean(p_entropy['xyvel'] + p_entropy['yaw'])
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
                # Calculate ratio (pi current policy / pi old policy)
                ratio = tf.exp(mb_neglogpacs - neglogpac)
                # Defining Loss = - J is equivalent to max J
                pg_losses = -advs * ratio
                pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                # Final PG loss
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - mb_neglogpacs))
                clipfrac = tf.reduce_mean(tf.dtypes.cast(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE), dtype=tf.float32))
                # Total loss
                loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
                # UPDATE THE PARAMETERS USING LOSS
                # 1. Get the model parameters
                var = pulsar.trainable_variables
                grads = tape.gradient(loss, var)
                # 3. Calculate the gradients
                #grads_and_var = pulsar.optimizer.get_gradients(loss, params)
                #grads, var = zip(*grads_and_var)
                if max_grad_norm is not None:
                    # Clip the gradients (normalize)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                grads_and_var = list(zip(grads, var))
                # zip aggregate each gradient with parameters associated
                # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
                pulsar.optimizer.apply_gradients(grads_and_var)

    # Send updated agent to actor
    agent.set_weights(pulsar.get_weights())
    for idx in range(actor_lower_bound, actor_upper_bound):
        MPI.COMM_WORLD.send(agent, dest=idx)
    # Send updated agent to coordinator
    comm.send(agent, dest=0)
