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
                 league_exploiters=1):
        # Load payoff
        self._payoff  = Payoff()
        self._learning_agents = []

        pulsar = Pulsar(False)
        pulsar.call_build()    

        for idx in range(main_agents):
            ma_name = "main_agent:"+str(idx)
            ma_agent = Agent(pulsar.get_weights())
            main_agent = MainPlayer(ma_agent, self._payoff, name=ma_name)
            self._learning_agents.append(main_agent)
            self._payoff.add_player(main_agent.checkpoint())

        for idx in range(main_exploiters):
            me_name = "main_exploit:"+str(idx)
            me_agent = Agent(pulsar.get_weights())
            main_exploiter = MainExploiter(me_agent, self._payoff, name=me_name)
            self._learning_agents.append(main_exploiter)

        for idx in range(league_exploiters):
            le_name = "league_exploit:"+str(idx)
            le_agent = Agent(pulsar.get_weights())
            league_exploiters = LeagueExploiter(le_agent, self._payoff, name=le_name)
            self._learning_agents.append(league_exploiters)

        for player in self._learning_agents:
            self._payoff.add_player(player)

    def update(self, home, away, result):
        return self._payoff.update(home, away, result)

    def get_player(self, idx):
        return self._learning_agents[idx]
  
    def get_player_agent(self, idx):
        return self._learning_agents[idx].get_agent()
  
    def set_player_agent(self, idx, agent):
        self._learning_agents[idx].set_agent(agent)

    def add_player(self, player):
        self._payoff.add_player(player)
    
    def set_eval_intv_steps(self, steps):
        self.eval_steps = steps
        self.eval_intv_steps = steps
    
    def incre_eval_intv_steps(self):
        self.eval_intv_steps += self.eval_steps

    def get_eval_intv_steps(self):
        return self.eval_intv_steps
    
    def get_winrate(p1, p2):
        return self._payoff._win_rate(p1, p2)
    
    def get_payoff_players(self):
        return self._payoff.players
