import sys
import random
import math
from copy import deepcopy
from itertools import compress
import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple
from mujoco_worldgen.util.rotation import mat2quat
from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs, joint_qpos_idxs, body_names_from_joint_prefix
from environment.wrappers.util_w import update_obs_space
from environment.utils.vision import insight, in_cone2d
from environment.module.util import clip_angle_range


class ProjectileManager:
    '''
        Manages the projectile hit-chance
        Args:
            env: simulator environment
            sim: mujoco sim
            n_agents: number of agents
            projectile_speed: max speed of projectile (m/s)
            response_time: mujoco timestep (in seconds) * n_substeps
            dodge_tolerance: distance of the  armor position from current bullet position in which counts as a hit
            add_bullets_visual: true to enable bullets visualization
            nbullets: number of bullets for visualization
    '''
    def __init__(self, env, sim, n_agents, projectile_speed, response_time, dodge_tolerance, add_bullets_visual=False, nbullets=0):
        self.agent_infos = env.metadata['agent_infos']
        self.n_agents = n_agents
        self.projectile_speed = projectile_speed
        self.response_time = response_time
        self.og_dodge_tolerance = dodge_tolerance
        self.dodge_tolerance = dodge_tolerance
        self.add_bullets_visual = add_bullets_visual
        self.nbullets = nbullets
        self.projectile_buffer = []
        if add_bullets_visual:
            self.bullets_idx = [0 for _ in range(n_agents)]
            self.bullets_site_id = np.array([[sim.model.site_name2id(f"agent{j}:bullet{i}") for i in range(nbullets)] for j in range(self.n_agents)])
        # Get all agent armor indexes
        self.agent_f_armors = [sim.model.geom_name2id(f"agent{i}:armor_f") for i in range(self.n_agents)]
        self.agent_b_armors = [sim.model.geom_name2id(f"agent{i}:armor_b") for i in range(self.n_agents)]
        self.agent_l_armors = [sim.model.geom_name2id(f"agent{i}:armor_l") for i in range(self.n_agents)]
        self.agent_r_armors = [sim.model.geom_name2id(f"agent{i}:armor_r") for i in range(self.n_agents)]
    
    def add_2_buffer(self, armor_pos, barrel_pos, bullet_agent_id, team_color):
        '''
            Args:
                armor_pos: actual position (xyz) of the armor
                barrel_pos: actual position (xyz) of the barrel
                bullet_agent_id: id of the bullet's agent
                team_color: team of the bullet's agent
            Returns:
                speed of projectile
        '''
        dist = np.linalg.norm(np.array(armor_pos) - np.array(barrel_pos))
        projectile_speed = self.projectile_speed - np.random.uniform(low=0.0, high=3.0)
        dist_per_step = (projectile_speed * 1000.0 * self.response_time)
        steps_needed = math.ceil(dist / dist_per_step)
        proj = {
            'steps_needed': steps_needed,
            'armor_pos': np.array(armor_pos),
            'barrel_pos': np.array(barrel_pos),
            'constant_steps_needed': steps_needed,
            'team_color': team_color,
            'bullet_agent_id': bullet_agent_id
        }
        if self.add_bullets_visual:
            proj['bullets_idx'] = self.bullets_idx[bullet_agent_id]
            self.bullets_idx[bullet_agent_id] = (self.bullets_idx[bullet_agent_id] + 1) % self.nbullets
        self.projectile_buffer.append(proj)
        return projectile_speed

    def check_all_armors(self, sim, bullet_agent_id, bullet_team, bullet_pos):
        """
            Args:
                sim: mujoco sim
                bullet_team: team of bullet's agent
                bullet_pos: current position of bullet
        """
        for a_idx in range(self.n_agents):
            if a_idx != bullet_agent_id:
                armor = 'f'
                dist = np.linalg.norm(np.array(sim.data.geom_xpos[self.agent_f_armors[a_idx]]) - bullet_pos)
                new_dist = np.linalg.norm(np.array(sim.data.geom_xpos[self.agent_b_armors[a_idx]]) - bullet_pos)
                if new_dist < dist:
                    dist = new_dist
                    armor = 'b'
                new_dist = np.linalg.norm(np.array(sim.data.geom_xpos[self.agent_l_armors[a_idx]]) - bullet_pos)
                if new_dist < dist:
                    dist = new_dist
                    armor = 'l'
                new_dist = np.linalg.norm(np.array(sim.data.geom_xpos[self.agent_r_armors[a_idx]]) - bullet_pos)
                if new_dist < dist:
                    dist = new_dist
                    armor = 'r'
                if dist <= self.dodge_tolerance:
                    return True, (a_idx, armor)
        return False, (None, 'n')

    def query(self, sim):
        """
            Args:
                sim: mujoco sim
        """
        # Randomize dodge tolerance
        self.dodge_tolerance = self.og_dodge_tolerance + np.random.uniform(low=-5.0, high=5.0)
        # Calculate bullet hits
        new_projectiles_buffer = []
        hits = [[0, 'n'] for i in range(self.n_agents)]
        for i, proj in enumerate(self.projectile_buffer):
            # Calculate current bullet position
            dist = proj['armor_pos'] - proj['barrel_pos']
            current_pos = proj['barrel_pos'] + dist * ( (proj['constant_steps_needed'] - proj['steps_needed']) / proj['constant_steps_needed'] )
            # Add bullets for visualization if True
            if self.add_bullets_visual:
                sim.data.site_xpos[self.bullets_site_id[proj['bullet_agent_id']][proj['bullets_idx']]] = current_pos
            # Check for armor hits
            hit, idx_armor = self.check_all_armors(sim, proj['bullet_agent_id'], proj['team_color'], current_pos)
            if hit:
                hits[idx_armor[0]][0] += 1
                hits[idx_armor[0]][1] = idx_armor[1]
            if self.add_bullets_visual and (hit or (proj['steps_needed'] <= 0)):
                sim.data.site_xpos[self.bullets_site_id[proj['bullet_agent_id']][proj['bullets_idx']]] = np.array([-100, -100, -100])
            else:
               self.projectile_buffer[i]['steps_needed'] -= 1
               new_projectiles_buffer.append(self.projectile_buffer[i])
        self.projectile_buffer = new_projectiles_buffer
        return hits
    
    def reset(self, env):
        """
            Args:
                env: simulator environment
        """
        self.projectile_buffer = []
        self.agent_infos = env.metadata['agent_infos']


class ProjectileWrapper(gym.Wrapper):
    '''
        Allows agent to shoot a projectile towards an armor whenever the opponent and the armor is visible
        Args:
            env: simulator environment
            add_bullets_visual: true to enable bullets visualization
            nbullets: number of bullets for visualization
    '''
    def __init__(self, env, add_bullets_visual=False, nbullets=0):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.add_bullets_visual = add_bullets_visual
        self.nbullets = nbullets
        mjco_ts = env.mjco_ts
        self.projmang = ProjectileManager(env=env, sim=self.unwrapped.sim, n_agents=self.n_agents,
                                          projectile_speed=25, response_time=mjco_ts * self.env.n_substeps,
                                          dodge_tolerance=30.0, add_bullets_visual=add_bullets_visual,
                                          nbullets=nbullets)
        # Add observation for number of projectiles
        self.observation_space = update_obs_space(self.env, {'nprojectiles': [self.n_agents, 1] })
        self.observation_space = update_obs_space(self.env, {'proj_dmg': [self.n_agents, 2]})
    
    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim
        # Get new agents' info
        self.agent_infos = self.metadata['agent_infos']
        # Reset projectile manager
        self.projmang.reset(self)
        # Cache agents rqpos
        self.agents_rqpos = [qpos_idxs_from_joint_prefix(sim, f'agent{i}:rz') for i in range(self.n_agents)]
        # Cache geom ids
        self.barrel_idxs = np.array([sim.model.site_name2id(f"agent{i}:barrel_sight") for i in range(self.n_agents)])
        # Cache armor ids
        self.agent_armors = []
        self.agent_armors.append([sim.model.geom_name2id(f"agent{i}:armor_f") for i in range(self.n_agents)])
        self.agent_armors.append([sim.model.geom_name2id(f"agent{i}:armor_b") for i in range(self.n_agents)])
        self.agent_armors.append([sim.model.geom_name2id(f"agent{i}:armor_l") for i in range(self.n_agents)])
        self.agent_armors.append([sim.model.geom_name2id(f"agent{i}:armor_r") for i in range(self.n_agents)])
        self.agent_armors = np.array(self.agent_armors)
        # Timer to account for the interval between armor hits
        self.agent_armor_timer = [{'f': -1000000, 'b': -1000000, 'l': -1000000, 'r': -1000000} for _ in range(self.n_agents)]
        # Buffer to store amount of projectiles each agent has
        self.magazines = [[0] for _ in range(self.n_agents)]
        # Barrel heat manager
        self.bheat = [[0] for _ in range(self.n_agents)]
        self.bheat_ts = [[0] for _ in range(self.n_agents)]
        # Fire rate recorder
        self.firerate = [[-1000000] for _ in range(self.n_agents)]
        # Damage recorder
        self.proj_dmg_taken = [[0, 'a'] for _ in range(self.n_agents)]
        # Each team starts with 50 projectiles
        redt_reloaded, bluet_reloaded = False, False
        for aidx, agent_info in enumerate(self.agent_infos):
            if agent_info['team'] == 'red' and not redt_reloaded:
                self.magazines[aidx][0] += 50
                redt_reloaded = True
            if agent_info['team'] == 'blue' and not bluet_reloaded:
                self.magazines[aidx][0] += 50
                bluet_reloaded = True
        return self.observation(obs)

    def shoot_projectile(self, obs, action):
        # DEBUG
        #action['opponent'] = np.array([[0] for _ in range(self.n_agents)])
        #action['armor'] = np.array([[0] for _ in range(self.n_agents)])
        # DEBUG
        sim = self.unwrapped.sim
        agents_hp = self.env.get_hp()
        hits = self.projmang.query(sim)
        # Check for armor hits.
        self.proj_dmg_taken = [[0, 'n'] for _ in range(self.n_agents)]
        for i, h in enumerate(hits):
            self.proj_dmg_taken[i][1] = h[1]
            # Check if the projectile is within armor detection interval
            if h[1] != 'n' and self.env.get_ts() - self.agent_armor_timer[i][h[1]] < self.env.secs_to_steps(0.05):
                continue
            elif h[1] != 'n':
                self.agent_armor_timer[i][h[1]] = self.env.get_ts()
            # Account for the armor hit by projectile
            if h[1] == 'f':
                self.env.minus_hp(i, h[0] * 20.0)
                self.proj_dmg_taken[i][0] = h[0] * 20.0
            elif h[1] == 'l' or h[1] == 'r':
                self.env.minus_hp(i, h[0] * 40.0)
                self.proj_dmg_taken[i][0] = h[0] * 40.0
            elif h[1] == 'b':
                self.env.minus_hp(i, h[0] * 60.0)
                self.proj_dmg_taken[i][0] = h[0] * 60.0
        # Handle shooting
        for ib, b in enumerate(self.barrel_idxs):
            # Check if shooting is available
            if (obs['Agent:buff'][ib][2][0] or
                agents_hp[ib][0] <= 0 or
                action['opponent'][ib] == 0 or
                self.env.get_ts() - self.firerate[ib][0] < self.env.secs_to_steps(0.1) or
                self.magazines[ib][0] == 0 or
                self.bheat[ib][0] > 215):
                # Can't shoot so skip
                pass
            else:
                # Check if opponent exists
                opp_cnt = 0
                opponent_id = None
                for oi in range(self.n_agents):
                    if self.agent_infos[ib]['team'] != self.agent_infos[oi]['team']:
                        opp_cnt += 1
                        if opp_cnt == action['opponent'][ib]:
                            opponent_id = oi
                            break
                if opponent_id == None:
                    # No selected opponent so skip
                    pass
                elif agents_hp[opponent_id][0] > 0:
                    barrel_pos = deepcopy(sim.data.site_xpos[b])
                    agent_angle = (clip_angle_range([sim.data.qpos[self.agents_rqpos[ib]]])[0] + 4.71239) % 6.28319
                    armor_in_cone = in_cone2d(origin_pts=np.array([barrel_pos[0:2]]),
                                            origin_angles=np.array(agent_angle),
                                            cone_angle=1.5708,
                                            target_pts=np.array([sim.data.geom_xpos[self.agent_armors[action['armor'][ib][0]][opponent_id]][0:2]]))[0][0]
                    armors_is = insight(sim, geom1_id=None, site1_id=b, geom2_id=self.agent_armors[action['armor'][ib][0]][opponent_id], dist_thresh=8100.0, check_body=False) if armor_in_cone else False
                    # Store projectile into buffer if armor is insight
                    if armors_is:
                        armor_pos = deepcopy(sim.data.geom_xpos[self.agent_armors[action['armor'][ib][0]][opponent_id]])
                        projectile_speed = self.projmang.add_2_buffer(armor_pos, barrel_pos, ib, self.agent_infos[ib]['team'])
                        self.magazines[ib][0] -= 1
                        self.bheat[ib][0] += projectile_speed
                        self.bheat_ts[ib][0] = self.env.get_ts()
                        self.firerate[ib][0] = self.env.get_ts()
            # Barrel heat cooldown
            if (self.env.get_ts() - self.bheat_ts[ib][0]) >= self.env.secs_to_steps(1.0):
                if agents_hp[ib][0] < 400:
                    self.bheat[ib][0] -= 240
                else:
                    self.bheat[ib][0] -= 120
                self.bheat[ib][0] = max(0, self.bheat[ib][0])
                if self.bheat[ib][0] != 0:
                    self.bheat_ts[ib][0] = self.env.get_ts()
    
    def ammo_buff(self, obs):
        for i in range(self.n_agents):
            more_ammo = obs['Agent:buff'][i][1][0]
            if more_ammo:
                self.env.set_buff_status(agent_idx=i, buff_idx=1, status=0)
                for j in range(self.n_agents):
                    if self.agent_infos[j]['team'] == self.agent_infos[i]['team']:
                        self.magazines[j][0] += 100

    def observation(self, obs):
        obs['nprojectiles'] = np.array(self.magazines)
        obs['proj_dmg'] = np.array(self.proj_dmg_taken)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.shoot_projectile(obs, action)
        self.ammo_buff(obs)
        return self.observation(obs), rew, done, info
