import numpy as np
from scipy.linalg import circulant


class Game:

    def __init__(self, n_agents=2, horizon=25):
        self.n_agents = n_agents
        self.horizon = horizon
    
    def reset(self):
        self.max_slot = 10
        self.targets = 2
        self.players_pos = [[np.random.randint(0, self.max_slot), np.random.randint(0, self.max_slot)] for _ in range(self.n_agents)]
        self.random_targs = [[np.random.randint(0, self.max_slot), np.random.randint(0, self.max_slot)] for _ in range(self.targets)]
        self.t = 0
        obs = self.get_indv_obs()
        return obs

    def get_obs(self):
        return np.array([self.players_pos, self.random_targs])

    def get_indv_obs(self):
        obs = self.get_obs()
        order = circulant(np.arange(self.n_agents))
        indv_obss = []
        for i in range(self.n_agents):
            indv_pos_obs = obs[0][i]
            flattened_obs1 = [obs[1][idx] for idx in range(self.targets)]
            indv_obs = np.array([indv_pos_obs] + flattened_obs1).flatten()
            indv_obss.append(indv_obs)
        return np.array(indv_obss).astype(np.float32)

    def step(self, actions):
        """
            actions:
                    0 = left
                    1 = right
                    2 = up
                    3 = down
                    4 = stay
        """
        # Reward
        covered = 0
        done = False
        for i in range(self.targets):
            for j in range(self.n_agents):
                if self.players_pos[j] == self.random_targs[i]:
                    covered += 1
                    break
        #if self.players_pos[0] == self.random_targs[0]:
        #rew = covered
        #    covered += 1
        if covered >= self.n_agents:
            rew = 1
        else:
            rew = 0
        # Action
        for i in range(self.n_agents):
            if actions[i] == 0:
                self.players_pos[i][0] -= 1
            elif actions[i] == 1:
                self.players_pos[i][0] += 1
            elif actions[i] == 2:
                self.players_pos[i][1] += 1
            elif actions[i] == 3:
                self.players_pos[i][1] -= 1
        self.t += 1
        if self.t == self.horizon:
            done = True
        # Check if out of bound
        #for i in range(self.n_agents):
        #    if (self.players_pos[i][0] < 0 or self.players_pos[i][0] >= self.max_slot
        #        or self.players_pos[i][1] < 0 or self.players_pos[i][1] >= self.max_slot):
        #        done = True
        # Obs
        obs = self.get_indv_obs()
        return obs, rew, done
