import sys
import time
import numpy as np
from copy import deepcopy

from architecture.entity_encoder.entity_formatter import Entity_formatter
from sim2real.info_masker import InfoMasker
from sim2real.noise import Noise
from sim2real.domain_rand import DomainRandomization


class TimeWarper():

    def __init__(self, env, networks, comm, n_agents, mjco_ts, n_substeps, nsteps):
        self.env = env
        self.networks = networks
        self.comm = comm
        self.n_agents = n_agents
        self.mjco_ts = mjco_ts
        self.n_substeps = n_substeps
        self.nsteps = nsteps + 1
        self.entity_formatter = Entity_formatter()
        self.noise = Noise()
        self.info_masker = InfoMasker(n_agents=n_agents, mjco_ts=mjco_ts, n_substeps=n_substeps)
        self.domain_rand = DomainRandomization(n_agents)
        # Parameters to simulate real-life scenario
        """
            nn_t_per_eval: time in seconds per network's forward pass
            nn_t_per_eval_std: standard deviation for "nn_t_per_eval"
        """
        self.nn_t_per_eval = 0.030
        self.nn_t_per_eval_std = 0.001
        self.done_flag = False
    
    def reset(self):
        # Buffer for trajectory
        self.cstep = [0, 0]
        self.done_flag = False
        self.extra_rew = [0, 0]
        self.mb_rewards = np.zeros([self.nsteps, 2, 1], dtype=np.float32)
        self.mb_values = np.zeros([self.nsteps, 2, 1], dtype=np.float32)
        self.mb_neglogpacs = np.zeros([self.nsteps, 2, 1], dtype=np.float32)
        self.mb_dones = np.zeros([self.nsteps, 2, 1], dtype=np.float32)
        self.mb_entity_masks = np.zeros([self.nsteps, 2, 24], dtype=np.float32)
        self.mb_baselines = np.zeros([self.nsteps, 2, 88], dtype=np.float32)
        self.mb_states = np.zeros([self.nsteps, 2, 1, 2048], dtype=np.float32)
        self.mb_entities = self.entity_formatter.get_empty_obs_with_shapes(self.nsteps, 2)
        self.mb_scalar_features = {'match_time': np.zeros([self.nsteps, 2, 1], dtype=np.float32),
                                   'n_opponents': np.zeros([self.nsteps, 2, 1], dtype=np.float32)
                                  }
        self.mb_actions = {'x': np.zeros([self.nsteps, 2, 1], dtype=np.float32),
                           'y': np.zeros([self.nsteps, 2, 1], dtype=np.float32),
                           'yaw': np.zeros([self.nsteps, 2, 1], dtype=np.float32),
                           'opponent': np.zeros([self.nsteps, 2, 1], dtype=np.float32),
                           'armor': np.zeros([self.nsteps, 2, 1], dtype=np.float32)
                          }
        self.mb_logits = {'x': np.zeros([self.nsteps, 2, 21], dtype=np.float32),
                          'y': np.zeros([self.nsteps, 2, 21], dtype=np.float32),
                          'yaw': np.zeros([self.nsteps, 2, 21], dtype=np.float32),
                          'opponent': np.zeros([self.nsteps, 2, 3], dtype=np.float32),
                          'armor': np.zeros([self.nsteps, 2, 4], dtype=np.float32)
                         }

    def reset_env(self):
        # Domain randomization
        #self.noise = self.domain_rand.noise_randomization(self.noise)
        #self.info_masker = self.domain_rand.info_masker_randomization(self.info_masker)
        self.domain_rand.kv_randomization(self.env)
        self.domain_rand.barrel_sight_randomization(self.env)
        # Reset masker
        self.info_masker.reset_masker()
        # Create sampling timesteps
        agents_sample_ts = [[[0, [idx]]] for idx in range(self.n_agents)]
        max_sample_ts = 0
        while max_sample_ts < self.secs_to_steps(3 * 60):
            for idx in range(self.n_agents):
                agents_sample_ts[idx].append(
                                            [self.secs_to_steps(np.random.normal(self.nn_t_per_eval, self.nn_t_per_eval_std))
                                            + agents_sample_ts[idx][len(agents_sample_ts[idx]) - 1][0], [idx]]
                                            )
                max_sample_ts = max(max_sample_ts, agents_sample_ts[idx][len(agents_sample_ts[idx]) - 1][0])
        all_sample_ts = []
        for idx in range(self.n_agents):
            all_sample_ts += agents_sample_ts[idx]
        # Order sampling sequentially
        all_sample_ts.sort(key=lambda x: x[0])
        #all_sample_ts = all_sample_ts[1:]
        # Concat samples at the same timestep
        concated_all_sample_ts = []
        all_sample_t = 0
        while all_sample_t < len(all_sample_ts):
            t_ahead = all_sample_t + 1
            while t_ahead < len(all_sample_ts) and all_sample_ts[all_sample_t][0] == all_sample_ts[t_ahead][0]:
                all_sample_ts[all_sample_t][1] += all_sample_ts[t_ahead][1]
                t_ahead += 1
            concated_all_sample_ts.append(all_sample_ts[all_sample_t])
            all_sample_t = t_ahead
        self.all_sample_ts = concated_all_sample_ts
        self.all_sample_ts = self.all_sample_ts[1:]
        # Env
        self.sample_ts = 0
        self.obs = self.env.reset()
        self.dones = False
        self.agent_first_step = [True for _ in range(self.n_agents)]
        # Clean states for each agent
        self.states = [self.networks[ai].get_initial_states() for ai in range(self.n_agents)]
        # Actions
        self.actions = [{
            'action_movement': np.array([0, 0, 0]),
            'opponent': 0,
            'armor': np.array([0])
        } for _ in range(self.n_agents)]

    def increment_buffer_cond(self, agent_no):
        if agent_no != 0 and agent_no != 1:
            return False
        if self.cstep[agent_no] >= self.nsteps:
            return False
        return True

    def env_step(self, agent_no, steps, render=False):
        # Collect reward if main agents, also increment buffer
        for ai in agent_no:
            if (not self.agent_first_step[ai] or self.done_flag) and self.increment_buffer_cond(ai):
                self.mb_rewards[self.cstep[ai] - 1, ai] = self.rewards[ai] + self.extra_rew[ai]
                self.extra_rew[ai] = 0
                self.done_flag = False
            self.agent_first_step[ai] = False
        # Get scalar features
        scalar_features = {
            'match_time': np.array([[self.env.t]], dtype=np.float32),
            'n_opponents': np.array([[2]], dtype=np.float32)
        }
        # Get embedded entities
        for ai in agent_no:
            masks_of_obs, entities = self.info_masker.get_masked_entities(ai)
            # Add noise to entities
            entities = self.noise.add_noise(masks_of_obs, entities)
            entity_masks = np.array([masks_of_obs["my_qpos"],           masks_of_obs["my_qvel"],
                                     masks_of_obs["local_qvel"],        masks_of_obs["teammate_qpos"],
                                     masks_of_obs["opponent1_qpos"],    masks_of_obs["opponent2_qpos"],
                                     masks_of_obs["my_hp"],             masks_of_obs["teammate_hp"],
                                     masks_of_obs["opponent1_hp"],      masks_of_obs["opponent2_hp"],
                                     masks_of_obs["my_projs"],          masks_of_obs["teammate_projs"],
                                     masks_of_obs["opponent1_projs"],   masks_of_obs["opponent2_projs"],
                                     masks_of_obs["my_armors"],         masks_of_obs["teammate_armors"],
                                     masks_of_obs["my_hp_deduct"],      masks_of_obs["my_hp_deduct_res"],
                                     masks_of_obs["zone_1"],            masks_of_obs["zone_2"],
                                     masks_of_obs["zone_3"],            masks_of_obs["zone_4"],
                                     masks_of_obs["zone_5"],            masks_of_obs["zone_6"]], dtype=np.float32)
            # Get baseline for value function for main agents only
            if ai == 0 or ai == 1:
                baseline = self.entity_formatter.get_baseline(self.n_agents, ai, deepcopy(self.obs), self.env.ts)
            else:
                baseline = np.zeros([1, 88], dtype=np.float32)
            current_action, neglogp, entropy, value, self.states[ai], prev_state, logits = self.networks[ai](scalar_features,
                                                                                                             entity_masks,
                                                                                                             entities,
                                                                                                             baseline,
                                                                                                             self.states[ai],
                                                                                                             np.array([self.dones], dtype=np.float32))
            # Form actions
            self.actions[ai]['action_movement'] = np.array([(np.array(current_action['x'])[0] - 10) / 10.0,
                                                            (np.array(current_action['y'])[0] - 10) / 10.0,
                                                            (np.array(current_action['yaw'])[0] - 10) / 10.0]
                                                          )
            self.actions[ai]['opponent'] = current_action['opponent'][0]
            self.actions[ai]['armor'] = np.array([current_action['armor'][0]])
            # Collect trajectory if main agents
            if self.increment_buffer_cond(ai):
                self.mb_values[self.cstep[ai], ai] = value
                self.mb_states[self.cstep[ai], ai] = prev_state
                self.mb_neglogpacs[self.cstep[ai], ai] = neglogp
                self.mb_entity_masks[self.cstep[ai], ai] = entity_masks
                for k in self.mb_scalar_features.keys():
                    self.mb_scalar_features[k][self.cstep[ai], ai] = scalar_features[k]
                for k in self.mb_entities.keys():
                    self.mb_entities[k][self.cstep[ai], ai] = entities[k]
                for k in self.mb_actions.keys():
                    self.mb_actions[k][self.cstep[ai], ai] = current_action[k]
                for k in self.mb_logits.keys():
                    self.mb_logits[k][self.cstep[ai], ai] = logits[k]
                self.mb_baselines[self.cstep[ai], ai] = baseline 
                self.mb_dones[self.cstep[ai], ai] = self.dones
        agent_actions = {
            'action_movement': np.array([self.actions[ai]['action_movement'] for ai in range(self.n_agents)]),
            'opponent': np.array([self.actions[ai]['opponent'] for ai in range(self.n_agents)]),
            'armor': np.array([self.actions[ai]['armor'] for ai in range(self.n_agents)])
        }
        # Take actions in env and look at the final reward
        for step in range(steps):
            self.obs, self.rewards, self.dones, self.info = self.env.step(agent_actions)
            # Store entities
            bare_entities = []
            for ai in range(self.n_agents):
                _, bare_entity = self.entity_formatter.concat_encoded_entity_obs(self.n_agents, ai, deepcopy(self.obs))
                bare_entities.append(bare_entity)
            self.info_masker.step(self.env.ts, bare_entities)
            for ai in range(2):
                self.extra_rew[ai] += self.info['lasting_rew'][ai]
            if render:
                self.env.render(mode="human")
            if self.dones:
                # Add the reward to buffer no matter what
                self.done_flag = True
                # Get opponent from rmleague
                self.new_opponent()
                break
        for ai in agent_no:
            if self.increment_buffer_cond(ai):
                self.cstep[ai] += 1
        
    def step(self, render=False):
        sample_ts = self.all_sample_ts[self.sample_ts]
        self.env_step(sample_ts[1], int(sample_ts[0] - self.env.ts), render=render)
        self.sample_ts += 1
        if self.dones:
            self.reset_env()
    
    def collect(self):
        # Collect a batch with size nsteps
        while self.cstep[0] < self.nsteps or self.cstep[1] < self.nsteps:
            self.step()
        traj_mb_entities = {k: v[0:-1] for k, v in self.mb_entities.items()}
        traj_mb_scalar_features = {k: v[0:-1] for k, v in self.mb_scalar_features.items()}
        traj_mb_actions = {k: v[0:-1] for k, v in self.mb_actions.items()}
        traj_mb_logits = {k: v[0:-1] for k, v in self.mb_logits.items()}
        trajectory = {
            'agent_steps': self.agent.get_steps(),
            'mb_rewards': self.mb_rewards[0:-1],
            'mb_values': self.mb_values[0:-1],
            'mb_neglogpacs': self.mb_neglogpacs[0:-1],
            'mb_dones': self.mb_dones[0:-1],
            'mb_entity_masks': self.mb_entity_masks[0:-1],
            'mb_baselines': self.mb_baselines[0:-1],
            'mb_states': self.mb_states[0:-1],
            'mb_entities': traj_mb_entities,
            'mb_scalar_features': traj_mb_scalar_features,
            'mb_actions': traj_mb_actions,
            'mb_logits': traj_mb_logits,
            'last_values': self.mb_values[-1],
            'last_dones': self.mb_dones[-1]
        }  
        return trajectory

    def new_opponent(self):
        # Send match outcome to coordinator
        match_outcome = self.info['true_rew']
        traj_outcome = {'outcome': match_outcome}
        self.comm.send(traj_outcome, dest=0, tag=5)
        opponent = self.comm.recv(source=0, tag=4)
        for oidx in range(2, self.n_agents):
            self.networks[oidx].set_all_weights(opponent.get_weights())

    def set_agent(self, agent):
        self.agent = agent
        self.networks[0].set_all_weights(self.agent.get_weights())
        self.networks[1].set_all_weights(self.agent.get_weights())

    def secs_to_steps(self, secs):
        return int(secs / (self.mjco_ts * self.n_substeps))
    