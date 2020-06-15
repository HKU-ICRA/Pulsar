import os
import numpy as np
import pickle

from rmleague.agent import Agent
from rmleague.payoff import Payoff
from rmleague.main_player import MainPlayer
from rmleague.main_exploiter import MainExploiter
from rmleague.league_exploiter import LeagueExploiter
from architecture.pulsar import Pulsar


class League(object):
    def __init__(self,
                 main_agents=1,
                 main_exploiters=1,
                 league_exploiters=1,
                 checkpoint_steps=5e8):
        # Load payoff
        self._payoff  = Payoff()
        self._learning_agents = {'main_agent': [], 'main_exploiter': [], 'league_exploiter': []}
        self.checkpoint_steps = checkpoint_steps
        self.nupdates = 0
        self.training_time = 0

        empty_weights = dict()

        for idx in range(main_agents):
            ma_name = "main_agent:"+str(idx)
            ma_agent = Agent(empty_weights)
            main_agent = MainPlayer(ma_agent, self._payoff, name=ma_name, checkpoint_steps=checkpoint_steps)
            self._learning_agents['main_agent'].append(main_agent)
            self._payoff.add_player(main_agent.checkpoint())

        for idx in range(main_exploiters):
            me_name = "main_exploit:"+str(idx)
            me_agent = Agent(empty_weights)
            main_exploiter = MainExploiter(me_agent, self._payoff, name=me_name, checkpoint_steps=checkpoint_steps)
            self._learning_agents['main_exploiter'].append(main_exploiter)

        for idx in range(league_exploiters):
            le_name = "league_exploit:"+str(idx)
            le_agent = Agent(empty_weights)
            league_exploiters = LeagueExploiter(le_agent, self._payoff, name=le_name, checkpoint_steps=checkpoint_steps)
            self._learning_agents['league_exploiter'].append(league_exploiters)

        for player_type in self._learning_agents.keys():
            for player in self._learning_agents[player_type]:
                self._payoff.add_player(player)
    
    def add_main_exploiter(self):
        me_name = f"main_exploit:{len(self._learning_agents['main_exploiter'])}"
        main_agent_weights = self._learning_agents['main_agent'][0].get_weights()
        me_agent = Agent(main_agent_weights)
        main_exploiter = MainExploiter(me_agent, self._payoff, name=me_name, checkpoint_steps=self.checkpoint_steps)
        self._learning_agents['main_exploiter'].append(main_exploiter)
        self._payoff.add_player(main_exploiter)
    
    def add_league_exploiter(self):
        le_name = f"league_exploit:{len(self._learning_agents['league_exploiter'])}"
        main_agent_weights = self._learning_agents[0].get_weights()
        le_agent = Agent(main_agent_weights)
        league_exploiters = LeagueExploiter(le_agent, self._payoff, name=le_name, checkpoint_steps=self.checkpoint_steps)
        self._learning_agents['league_exploiter'].append(league_exploiters)
        self._payoff.add_player(league_exploiters)

    def update(self, home, away, result):
        return self._payoff.update(home, away, result)

    def get_player(self, player_type, idx):
        return self._learning_agents[player_type][idx]
  
    def get_player_agent(self, player_type, idx):
        return self.get_player(player_type, idx).get_agent()
  
    def set_player_agent(self, player_type, idx, agent):
        self._learning_agents[player_type][idx].set_agent(agent)

    def add_player(self, player):
        self._payoff.add_player(player)
    
    def get_winrate(p1, p2):
        return self._payoff._win_rate(p1, p2)
    
    def get_payoff_players(self):
        return self._payoff.players
    
    def set_ckpt_steps(self, checkpoint_steps):
        for k, players in self._learning_agents.items():
            for player in players:
                player.set_ckpt_steps(checkpoint_steps)
    