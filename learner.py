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
batch_size = 2048
bptt_ts = 16
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
    for idx in range(actor_lower_bound, actor_upper_bound):
        trajectory = MPI.COMM_WORLD.recv()
    
    nsteps = trajectory['mb_returns'].shape[0]
    
    b_scalar_features = trajectory['mb_scalar_features']
    b_entities = trajectory['mb_entities']
    mb_entity_masks = trajectory['mb_entity_masks']
    mb_baselines = trajectory['mb_baselines']
    mb_actions_xy = trajectory['mb_actions_xy']
    mb_actions_yaw = trajectory['mb_actions_yaw']
    mb_returns = trajectory['mb_returns']
    mb_dones = trajectory['mb_dones']
    mb_values = trajectory['mb_values']
    mb_neglogpacs_xy = trajectory['mb_neglogpacs_xy']
    mb_neglogpacs_yaw = trajectory['mb_neglogpacs_yaw']
    mb_states = trajectory['mb_states']
    agent.add_steps(nsteps)

    # Set up BPTT
    mb_scalar_features = {'match_time': [], 'bptt_match_time': []}
    mb_entities = []
    idxs = []
    for idx in range(bptt_ts, nsteps):
        if True in mb_dones[idx - bptt_ts:idx] and (idx + 1) % bptt_ts != 0:
            continue
        else:
            idxs.append(idx)
        mb_scalar_features['match_time'].append(b_scalar_features['match_time'][idx])
        mb_scalar_features['bptt_match_time'].append(b_scalar_features['bptt_match_time'][idx - bptt_ts:idx, 0])
        mb_entities.append(b_entities[idx - bptt_ts:idx, 0])
    mb_scalar_features['match_time'] = np.array(mb_scalar_features['match_time'])
    mb_scalar_features['bptt_match_time'] = np.array(mb_scalar_features['bptt_match_time'])
    mb_entities = np.array(mb_entities)
    nsteps = mb_entities.shape[0]

    mb_entity_masks = trajectory['mb_entity_masks'][idxs]
    mb_baselines = trajectory['mb_baselines'][idxs]
    mb_actions_xy = trajectory['mb_actions_xy'][idxs]
    mb_actions_yaw = trajectory['mb_actions_yaw'][idxs]
    mb_returns = trajectory['mb_returns'][idxs, :, idxs]
    mb_dones = trajectory['mb_dones'][idxs]
    mb_values = trajectory['mb_values'][idxs]
    mb_neglogpacs_xy = trajectory['mb_neglogpacs_xy'][idxs]
    mb_neglogpacs_yaw = trajectory['mb_neglogpacs_yaw'][idxs]
    mb_neglogpacs = mb_neglogpacs_xy + mb_neglogpacs_yaw

    if mb_states != None:
        for i in range(len(mb_states)):
            for j in range(len(mb_states[i])):
                mb_states[i][j] = tf.convert_to_tensor(np.repeat(mb_states[i][j][0:1][:], repeats=nsteps, axis=0))

    with tf.GradientTape() as tape:
        p_actions, p_neglogp, p_entropy, p_mean, vpred, p_states = pulsar(mb_scalar_features, mb_entities, mb_entity_masks, mb_baselines, mb_states)

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
