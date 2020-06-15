import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy

from rmleague.agent import Agent
from rmleague.league import League
from environment.envs.icra import make_env
from environment.envhandler import EnvHandler
from environment.viewer.monitor import Monitor
from architecture.pulsar import Pulsar
from architecture.entity_encoder.entity_formatter import Entity_formatter
from sim2real.time_warper import TimeWarper
from save import Save


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# 0: main, 1: main_exploit, 2:league_exploit
actors = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
learners = [15]


if rank == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
elif rank >= 1 and rank <= 14:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
elif rank == 15:
    pass
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf


# Training parameters
batch_size = nsteps = 104
no_of_agents_per_env = 4


def main():
    """
    Trains the Robomaster league.
    This main function acts as the coordinator for rmleague.
    """
    checkpoint_steps = 1e7
    def load_rmleague(league_file):
        if os.path.isfile(league_file):
            with open(league_file, 'rb') as f:
                return pickle.load(f)
        else:
            rmleague = League(main_agents=1, main_exploiters=0, league_exploiters=0, checkpoint_steps=checkpoint_steps)
            return rmleague
    league_file = os.path.join(os.getcwd(), 'data', 'league')
    rmleague = load_rmleague(league_file)
    rmleague.set_ckpt_steps(checkpoint_steps)
    # Start main-agent's learner processes    
    agent = rmleague.get_player_agent('main_agent', 0)
    comm.send(agent, dest=learners[0])
    # Start/Add main-exploiter's learner processes
    if len(learners) >= 2:
        if len(rmleague._learning_agents['main_exploiter']) == 0:
            rmleague.add_main_exploiter()
        agent = rmleague.get_player_agent('main_exploiter', 0)
        comm.send(agent, dest=learners[1])
    # Start/Add league-exploiter's learner processes
    if len(learners) >= 3:
        if len(rmleague._learning_agents['league_exploiter']) == 0:
            rmleague.add_league_exploiter()
        agent = rmleague.get_player_agent('league_exploiter', 0)
        comm.send(agent, dest=learners[2])
    # Send opponent to Actor
    opponent_coordinator = dict()
    player_types = {0: 'main_agent', 1: 'main_exploiter', 2: 'league_exploiter'}
    for i, actor in enumerate(actors):
        player = rmleague.get_player(player_types[i], 0)
        opponent, match_bool = player.get_match()
        opponent_coordinator[i] = opponent
        for a in actor:
            comm.send(opponent.get_agent(), dest=a)
    # Setup summary writer
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.getcwd(), "data", "logs", dt_string)
    summary_writer = tf.summary.create_file_writer(log_dir)
    # Run-time variables
    main_p_t = time.time()
    learner_updated = False
    # Start coordinator loop
    while True:
        # Receive trajectory outcome from Actor and perhaps checkpoint player
        for ia, actor in enumerate(actors):
            player = rmleague.get_player(player_types[ia], 0)
            if player.ready_to_checkpoint():
                rmleague.add_player(player.checkpoint())
            for ja, a in enumerate(actor):
                if comm.Iprobe(source=a, tag=5):
                #if actors_rcv_traj_p[ia][ja].Get_status():
                    #traj_outcome = actors_rcv_traj_p[ia][ja].wait()
                    traj_outcome = comm.recv(source=a, tag=5)
                    opponent = opponent_coordinator[ia]
                    rmleague.update(player, opponent, traj_outcome['outcome'])
                    comm.send(opponent.get_agent(), dest=a, tag=4)
        # Receive new agent and ppo summaries from Learner
        for il, learner in enumerate(learners):
            # Update agent of player
            if comm.Iprobe(source=learner, tag=3):
            #if learner_rcv_p[il].Get_status():
                learner_updated = True
                results = comm.recv(source=learner, tag=3)
                player = rmleague.get_player(player_types[il], 0)
                player.set_agent(results['agent'])
                player.incre_updates()
                # Report and log each Player's PPO details
                with summary_writer.as_default():
                    tf.summary.scalar(player.get_name()+':approxkl', results['approxkl'], step=player.get_updates())
                    tf.summary.scalar(player.get_name()+':loss', results['loss'], step=player.get_updates())
                    tf.summary.scalar(player.get_name()+':entropy', results['entropy'], step=player.get_updates())
                    tf.summary.scalar(player.get_name()+':return', results['return'], step=player.get_updates())
                    tf.summary.scalar(player.get_name()+':rewards', results['rewards'], step=player.get_updates())
                    tf.summary.scalar(player.get_name()+':total_steps', player.get_agent().get_steps(), step=player.get_updates())
                    print(f"PPO info (learner = {str(learner)}, steps = {player.get_agent().get_steps()}, time = {rmleague.training_time}):")
                    print(player.get_name()+":approxkl =", results['approxkl'])
                    print(player.get_name()+":loss =", results['loss'])
                    print(player.get_name()+":entropy =", results['entropy'])
                    print(player.get_name()+":return =", results['return'])
                    print(player.get_name()+":rewards =", results['rewards'])
                    print(player.get_name()+":number of fresh trajectories =", str(results['n_fresh_trajs']))
                    print(player.get_name()+":average response time =", str(results['average_response_time']))
                    print()
                    sys.stdout.flush()
                    summary_writer.flush()
        # Increment training timer and save updated rmLeague
        if learner_updated:
            rmleague.training_time += time.time() - main_p_t
            main_p_t = time.time()
            # Save progress
            rmleague.nupdates += 1
            Save(rmleague, os.path.join(os.getcwd(), 'data'), league_file)
            learner_updated = False


def actor(n_agents):
    print(f"STARTING ACTOR with rank {rank}")
    sys.stdout.flush()
    # GAE hyper-parameters
    lam = 0.95
    gamma = 0.99
    # Build network architecture
    pulsars = [Pulsar(1, 1, training=False) for _ in range(n_agents)]
    for pulsar in pulsars:
        pulsar.call_build()
    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]
    # Receive opponent from main
    opponent = comm.recv(source=0)
    for oidx in range(2, n_agents):
        pulsars[oidx].set_all_weights(opponent.get_weights())
    # Receive agent from learner
    agent = comm.recv(source=learners[agent_type])
    # Setup environment
    env = EnvHandler(make_env(env_no=rank))
    env = TimeWarper(env, pulsars, comm, n_agents, env.mjco_ts, env.n_substeps, nsteps)
    env.reset_env()
    env.set_agent(agent)
    while True:
        # Collect rollout from TimeWrapper
        env.reset()
        trajectory = env.collect()
        agent_steps = trajectory['agent_steps']
        mb_scalar_features = trajectory['mb_scalar_features']
        mb_entity_masks =    trajectory['mb_entity_masks']
        mb_neglogpacs =      trajectory['mb_neglogpacs']
        mb_baselines =       trajectory['mb_baselines']
        mb_entities =        trajectory['mb_entities']
        mb_rewards =         trajectory['mb_rewards']
        mb_actions =         trajectory['mb_actions']
        mb_logits =          trajectory['mb_logits']
        mb_states =          trajectory['mb_states']
        mb_values =          trajectory['mb_values']
        mb_dones =           trajectory['mb_dones']
        last_values =        trajectory['last_values']
        last_dones =         trajectory['last_dones']
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        # perform GAE calculation
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - last_dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        # Send trajectory to learner
        mb_values = np.squeeze(mb_values, axis=-1)
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_dones = np.squeeze(mb_dones, axis=-1)
        trajectory = {
            'agent_steps': agent_steps,
            'mb_scalar_features': mb_scalar_features,
            'mb_entity_masks': mb_entity_masks,
            'mb_entities': mb_entities,
            'mb_baselines': mb_baselines,
            'mb_actions': mb_actions,
            'mb_logits': mb_logits,
            'mb_returns': mb_returns,
            'mb_dones': mb_dones,
            'mb_values': mb_values,
            'mb_neglogpacs': mb_neglogpacs,
            'mb_states': mb_states,
            'mb_rewards': mb_rewards
        }
        # Send trajectory
        comm.send(trajectory, dest=learners[agent_type], tag=11)
        # Request for updated agent if any
        comm.send(True, dest=learners[agent_type], tag=15)
        agent = comm.recv(source=learners[agent_type], tag=15)
        env.set_agent(agent)
            

def learner():
    print(f"STARTING LEARNER with rank {rank}")
    sys.stdout.flush()
    # Truly-ppo hyperparameters
    KLRANGE = 0.03
    slope_rollback = -5
    slope_likelihood = 1
    # Learner hyperparameters
    ent_coef = 0.01
    vf_coef = 0.5
    CLIPRANGE = 0.2
    max_grad_norm = 5.0
    noptepochs = 8
    bptt_ts = 8
    n_actors = 10
    assert n_actors <= len(actors[0]), "Number of actors must be smaller than the actual amount"
    nbatch = batch_size * n_actors
    nbatch_bptt = int(nbatch/bptt_ts)
    actual_nbatch = nbatch_bptt * 2
    batch_scale = 1
    nbatch_steps = actual_nbatch // batch_scale
    # Build network architecture
    pulsar = Pulsar(nbatch_steps, bptt_ts, training=True)
    pulsar.call_build(learner=True)
    entity_formatter = Entity_formatter()
    # Receive agent from coordinator and set weights
    agent = comm.recv(source=0)
    pulsar.set_all_weights(agent.get_weights(), learner=True)
    agent.set_weights(pulsar.get_all_weights())
    # Get agent type
    agent_type = np.where(np.array(learners) == rank)[0][0]
    # Send agent to actor
    n_fresh_trajs = 0
    average_response_time = 40
    agent_last_traj_time = [0 for _ in range(len(actors[agent_type]))]
    agent_response_time = [40 for _ in range(len(actors[agent_type]))]
    for actor in actors[agent_type]:
        comm.send(agent, dest=actor)
    # Set learning rate
    LEARNING_RATE = 3e-4
    pulsar.set_optimizer(learning_rate=LEARNING_RATE)
    # PPO RL optimization loss function
    @tf.function
    def ppo_loss(b_scalar_features, b_entity_masks, b_entities, b_baselines, b_actions, b_logits, b_returns, b_dones, b_values, b_neglogpacs, b_states):
        # Reshape tensors to bptt len
        b_entities = {k: tf.reshape(b_entities[k], [nbatch_bptt, bptt_ts] + b_entities[k].shape[1:]) for k, v in b_entities.items()}
        b_scalar_features = {k: tf.reshape(b_scalar_features[k], [nbatch_bptt, bptt_ts] + b_scalar_features[k].shape[1:]) for k, v in b_scalar_features.items()}
        b_actions = {k: tf.reshape(b_actions[k], [nbatch_bptt, bptt_ts] + b_actions[k].shape[1:]) for k, v in b_actions.items()}
        b_logits = {k: tf.reshape(b_logits[k], [nbatch_bptt, bptt_ts] + b_logits[k].shape[1:]) for k, v in b_logits.items()}
        b_entity_masks = tf.reshape(b_entity_masks, [nbatch_bptt, bptt_ts] + b_entity_masks.shape[1:])
        b_baselines = tf.reshape(b_baselines, [nbatch_bptt, bptt_ts] + b_baselines.shape[1:])
        b_returns = tf.reshape(b_returns, [nbatch_bptt, bptt_ts] + b_returns.shape[1:])
        b_dones = tf.reshape(b_dones, [nbatch_bptt, bptt_ts] + b_dones.shape[1:])
        b_values = tf.reshape(b_values, [nbatch_bptt, bptt_ts] + b_values.shape[1:])
        b_neglogpacs = tf.reshape(b_neglogpacs, [nbatch_bptt, bptt_ts] + b_neglogpacs.shape[1:])
        b_states = tf.reshape(b_states, [nbatch_bptt, bptt_ts] + b_states.shape[1:])
        # Fold multi-agent obs into single batch
        b_entities = {k: tf.concat([b_entities[k][:, :, 0:1], b_entities[k][:, :, 1:2]], axis=0) for k, v in b_entities.items()}
        b_scalar_features = {k: tf.concat([b_scalar_features[k][:, :, 0], b_scalar_features[k][:, :, 1]], axis=0) for k, v in b_scalar_features.items()}
        b_actions = {k: tf.concat([b_actions[k][:, :, 0], b_actions[k][:, :, 1]], axis=0) for k, v in b_actions.items()}
        b_logits = {k: tf.concat([b_logits[k][:, :, 0], b_logits[k][:, :, 1]], axis=0) for k, v in b_logits.items()}
        b_entity_masks = tf.concat([b_entity_masks[:, :, 0], b_entity_masks[:, :, 1]], axis=0)
        b_baselines = tf.concat([b_baselines[:, :, 0], b_baselines[:, :, 1]], axis=0)
        b_returns = tf.concat([b_returns[:, :, 0], b_returns[:, :, 1]], axis=0)
        b_dones = tf.concat([b_dones[:, :, 0], b_dones[:, :, 1]], axis=0)
        b_values = tf.concat([b_values[:, :, 0], b_values[:, :, 1]], axis=0)
        b_neglogpacs = tf.concat([b_neglogpacs[:, :, 0], b_neglogpacs[:, :, 1]], axis=0)
        b_states = tf.concat([b_states[:, :, 0], b_states[:, :, 1]], axis=0)
        # Stochastic selection
        inds = tf.range(actual_nbatch)
        # Buffers for recording
        losses_total = []
        approxkls = []
        entropies = []
        # Start SGD
        for _ in range(noptepochs):
            inds = tf.random.shuffle(inds)
            for start in range(0, actual_nbatch, nbatch_steps):
                end = start + nbatch_steps
                # Gather mini-batch
                mbinds = inds[start:end]
                mb_entities = {k: tf.gather(b_entities[k], mbinds) for k, v in b_entities.items()}
                mb_scalar_features = {k: tf.gather(b_scalar_features[k], mbinds) for k, v in b_scalar_features.items()}
                mb_actions = {k: tf.gather(b_actions[k], mbinds) for k, v in b_actions.items()}
                mb_logits = {k: tf.gather(b_logits[k], mbinds) for k, v in b_logits.items()}
                mb_entity_masks = tf.gather(b_entity_masks, mbinds)
                mb_baselines = tf.gather(b_baselines, mbinds)
                mb_returns = tf.gather(b_returns, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                mb_states = tf.gather(b_states, mbinds)
                # Flatten for input
                mb_entities = {k: tf.reshape(mb_entities[k], [nbatch_steps * bptt_ts] + mb_entities[k].shape[2:]) for k, v in mb_entities.items()}
                mb_scalar_features = {k: tf.reshape(mb_scalar_features[k], [nbatch_steps * bptt_ts] + mb_scalar_features[k].shape[2:]) for k, v in mb_scalar_features.items()}
                mb_actions = {k: tf.reshape(mb_actions[k], [nbatch_steps * bptt_ts] + mb_actions[k].shape[2:]) for k, v in mb_actions.items()}
                mb_logits = {k: tf.reshape(mb_logits[k], [nbatch_steps * bptt_ts] + mb_logits[k].shape[2:]) for k, v in mb_logits.items()}
                mb_entity_masks = tf.reshape(mb_entity_masks, [nbatch_steps * bptt_ts] + mb_entity_masks.shape[2:])
                mb_baselines = tf.reshape(mb_baselines, [nbatch_steps * bptt_ts] + mb_baselines.shape[2:])
                mb_returns = tf.reshape(mb_returns, [nbatch_steps * bptt_ts] + mb_returns.shape[2:])
                mb_dones = tf.reshape(mb_dones, [nbatch_steps * bptt_ts] + mb_dones.shape[2:])
                mb_values = tf.reshape(mb_values, [nbatch_steps * bptt_ts] + mb_values.shape[2:])
                mb_neglogpacs = tf.reshape(mb_neglogpacs, [nbatch_steps * bptt_ts] + mb_neglogpacs.shape[2:])
                mb_states = tf.reshape(mb_states, [nbatch_steps * bptt_ts] + mb_states.shape[2:])
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, vpred, p_states, p_prev_states, taken_action_neglogp, kl = pulsar(mb_scalar_features,
                                                                                                                       mb_entity_masks,
                                                                                                                       mb_entities,
                                                                                                                       mb_baselines,
                                                                                                                       mb_states,
                                                                                                                       mb_dones,
                                                                                                                       mb_actions,
                                                                                                                       mb_logits)
                    advs = mb_returns - mb_values
                    # Batch normalize the advantages
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    # Calculate neglogpac
                    neglogpac = taken_action_neglogp
                    # Calculate the entropy
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
                    approxkl = tf.reduce_mean(kl)
                    # Calculate ratio (pi current policy / pi old policy)
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    # Defining Loss = - J is equivalent to max J
                    pg_targets = tf.where(
                        tf.logical_and( kl >= KLRANGE, ratio * advs > 1 * advs),
                        slope_likelihood * ratio * advs + slope_rollback * kl,
                        ratio * advs
                    )
                    pg_loss = -tf.reduce_mean(pg_targets)
                    # Total loss
                    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
                # 1. Get the model parameters
                var = pulsar.trainable_variables
                grads = tape.gradient(loss, var)
                # 3. Calculate the gradients
                # Clip the gradients (normalize)
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                grads_and_var = zip(grads, var)
                # zip aggregate each gradient with parameters associated
                # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
                pulsar.optimizer.apply_gradients(grads_and_var)
                losses_total.append(loss)
                approxkls.append(approxkl)
                entropies.append(entropy)
        losses_total = tf.reduce_mean(losses_total)
        approxkls = tf.reduce_mean(approxkls)
        entropies = tf.reduce_mean(entropies)
        return losses_total, approxkls, entropies
    # Handle trajectories
    UPDATE_PULSAR = False
    DEADLINE = 1e18
    # Normal trajectory
    cur_traj_size = 0
    trajectory = {
                'mb_entities': {
                    'my_qpos':                          np.empty((nbatch, 2, 3), dtype=np.float32),
                    'my_qvel':                          np.empty((nbatch, 2, 3), dtype=np.float32),
                    'local_qvel':                       np.empty((nbatch, 2, 2), dtype=np.float32),
                    'teammate_qpos':                    np.empty((nbatch, 2, 3), dtype=np.float32),
                    'opponent1_qpos':                   np.empty((nbatch, 2, 3), dtype=np.float32),
                    'opponent2_qpos':                   np.empty((nbatch, 2, 3), dtype=np.float32),
                    'my_hp' :                           np.empty((nbatch, 2, 1), dtype=np.float32),
                    'teammate_hp':                      np.empty((nbatch, 2, 1), dtype=np.float32),
                    'opponent1_hp':                     np.empty((nbatch, 2, 1), dtype=np.float32),
                    'opponent2_hp':                     np.empty((nbatch, 2, 1), dtype=np.float32),
                    'my_projs':                         np.empty((nbatch, 2, 1), dtype=np.float32),
                    'teammate_projs':                   np.empty((nbatch, 2, 1), dtype=np.float32),
                    'opponent1_projs':                  np.empty((nbatch, 2, 1), dtype=np.float32),
                    'opponent2_projs':                  np.empty((nbatch, 2, 1), dtype=np.float32),
                    'my_armors':                        np.empty((nbatch, 2, 4), dtype=np.float32),
                    'teammate_armors':                  np.empty((nbatch, 2, 4), dtype=np.float32),
                    'my_hp_deduct':                     np.empty((nbatch, 2, 2), dtype=np.float32),
                    'my_hp_deduct_res':                 np.empty((nbatch, 2, 2), dtype=np.float32),
                    'zone_1':                           np.empty((nbatch, 2, 4), dtype=np.float32),
                    'zone_2':                           np.empty((nbatch, 2, 4), dtype=np.float32),
                    'zone_3':                           np.empty((nbatch, 2, 4), dtype=np.float32),
                    'zone_4':                           np.empty((nbatch, 2, 4), dtype=np.float32),
                    'zone_5':                           np.empty((nbatch, 2, 4), dtype=np.float32),
                    'zone_6':                           np.empty((nbatch, 2, 4), dtype=np.float32)
                },
                'mb_scalar_features': {
                    'match_time':                       np.empty((nbatch, 2, 1), dtype=np.float32),
                    'n_opponents':                      np.empty((nbatch, 2, 1), dtype=np.float32)
                },
                'mb_actions': {
                    'x':                                np.empty((nbatch, 2, 1),      dtype=np.int32),
                    'y':                                np.empty((nbatch, 2, 1),      dtype=np.int32),
                    'yaw':                              np.empty((nbatch, 2, 1),      dtype=np.int32),
                    'opponent':                         np.empty((nbatch, 2, 1), dtype=np.int32),
                    'armor':                            np.empty((nbatch, 2, 1), dtype=np.int32)
                },
                'mb_logits': {
                    'x':                                np.empty((nbatch, 2, 21),      dtype=np.float32),
                    'y':                                np.empty((nbatch, 2, 21),      dtype=np.float32),
                    'yaw':                              np.empty((nbatch, 2, 21),      dtype=np.float32),
                    'opponent':                         np.empty((nbatch, 2, 3),      dtype=np.float32),
                    'armor':                            np.empty((nbatch, 2, 4),      dtype=np.float32)
                },
                'mb_entity_masks':                      np.empty((nbatch, 2, 24), dtype=np.float32),
                'mb_baselines':                         np.empty((nbatch, 2, 88), dtype=np.float32),
                'mb_returns':                           np.empty((nbatch, 2), dtype=np.float32),
                'mb_dones':                             np.empty((nbatch, 2), dtype=np.float32),
                'mb_values':                            np.empty((nbatch, 2), dtype=np.float32),
                'mb_neglogpacs':                        np.empty((nbatch, 2), dtype=np.float32),
                'mb_states':                            np.empty((nbatch, 2, 1, 2048), dtype=np.float32),
                'mb_rewards':                           np.empty((nbatch, 2), dtype=np.float32)
    }
    # Buffered trajectory
    MAX_TRAJ_BUFF = 100
    trajectory_buffer = []
    # Function to append trajectory
    def append_main_trajectory(a_trajectory, traj_size):
        a_traj_size = a_trajectory['mb_returns'].shape[0]
        for k, v in trajectory.items():
            if k == 'mb_scalar_features' or k == 'mb_actions' or k == 'mb_entities' or k == 'mb_logits':
                for k2 in trajectory[k].keys():
                    trajectory[k][k2][traj_size:min(traj_size+a_traj_size, nbatch)] = a_trajectory[k][k2][0:max(0, nbatch-traj_size)]
            else:
                trajectory[k][traj_size:min(traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-traj_size)]
        traj_size += min(a_traj_size, nbatch-traj_size)
        return traj_size
    # Function to append trajectory to buffer
    def append_buffer_trajectory(a_trajectory):
        if len(trajectory_buffer) >= MAX_TRAJ_BUFF:
            trajectory_buffer.pop(0)
        trajectory_buffer.append(a_trajectory)
    # Start learner process loop
    while True:
        # Send agent to actor if it requests for it
        for idx, actor in enumerate(actors[agent_type]):
            if comm.Iprobe(source=actor, tag=15):
                ready = comm.recv(source=actor, tag=15)
                comm.send(agent, dest=actor, tag=15)
                agent_last_traj_time[idx] = time.time()
        # Collect enough rollout to fill batch_size
        for idx, actor in enumerate(actors[agent_type]):
            if comm.Iprobe(source=actor, tag=11):
                # Append to trajectory
                a_trajectory = comm.recv(source=actor, tag=11)
                if a_trajectory['agent_steps'] == agent.get_steps():
                    cur_traj_size = append_main_trajectory(a_trajectory, cur_traj_size)
                    if len(trajectory_buffer) < 10:
                        trajectory_buffer.append(a_trajectory)
                    n_fresh_trajs += 1
                else:
                    append_buffer_trajectory(a_trajectory)
                # Record their response time
                this_agent_time = time.time()
                agent_response_time[idx] = this_agent_time - agent_last_traj_time[idx]
                average_response_time = np.mean(agent_response_time)
        # Check if we have enough data or if time's up
        if cur_traj_size >= nbatch:
            UPDATE_PULSAR = True
        elif time.time() - DEADLINE >= max(40.0, average_response_time + 5):
            # Fill rest of batch from buffer
            buffer_idx = len(trajectory_buffer) - 1                
            while cur_traj_size < nbatch:
                a_trajectory = trajectory_buffer[buffer_idx]
                cur_traj_size = append_main_trajectory(a_trajectory, cur_traj_size)
                buffer_idx -= 1
            UPDATE_PULSAR = True
        # Update Pulsar when conditions are met
        if UPDATE_PULSAR:
            UPDATE_PULSAR = False
            # Collect trajectories
            b_scalar_features = trajectory['mb_scalar_features']
            b_entity_masks = trajectory['mb_entity_masks']
            b_entities = trajectory['mb_entities']
            b_baselines = trajectory['mb_baselines']
            b_actions = trajectory['mb_actions']
            b_logits = trajectory['mb_logits']
            b_dones = trajectory['mb_dones']
            b_values = trajectory['mb_values']
            b_neglogpacs = trajectory['mb_neglogpacs']
            b_returns = trajectory['mb_returns']
            b_states = trajectory['mb_states']
            b_rewards = trajectory['mb_rewards']
            # Start SGD and optimize model via Adam
            losses, approxkls, entropies = ppo_loss(b_scalar_features, b_entity_masks,
                                                    b_entities, b_baselines, b_actions,
                                                    b_logits, b_returns, b_dones,
                                                    b_values, b_neglogpacs,
                                                    b_states)
            agent.set_weights(pulsar.get_all_weights())
            agent.add_steps(nbatch * 2)
            DEADLINE = time.time()
            # Send updated agent and summaries to coordinator
            results = {
                'agent': agent,
                'approxkl': np.array(approxkls),
                'loss': np.array(losses),
                'entropy': np.array(entropies),
                'return': np.mean(np.mean(b_returns, axis=0), axis=0),
                'rewards': np.mean(np.mean(b_rewards, axis=0), axis=0),
                'n_fresh_trajs': n_fresh_trajs,
                'average_response_time': average_response_time
            }
            comm.send(results, dest=0, tag=3)
            cur_traj_size = 0
            n_fresh_trajs = 0


if rank == 0:
    main()
elif rank >= 1 and rank <= 14:
    actor(n_agents=no_of_agents_per_env)
elif rank >= 15 and rank <= 15:
    learner()
