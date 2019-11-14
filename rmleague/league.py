import os
import numpy as np
import pickle

from rmleague.agent import Agent
from rmleague.payoff import Payoff
from rmleague.main_player import MainPlayer
from rmleague.main_exploiter import MainExploiter
from rmleague.league_exploiter import LeagueExploiter
from architecture.pulsar import Pulsar


def get_agent_files(name):
  model_file = os.path.join(os.getcwd(), 'data', 'model:'+name)
  agent_file = os.path.join(os.getcwd(), 'data', 'agent:'+name)
  player_file = os.path.join(os.getcwd(), 'data', 'player:'+name)
  return model_file, agent_file, player_file


class League(object):

  def __init__(self,
               main_agents=1,
               main_exploiters=1,
               league_exploiters=2):
    # Load payoff else create new payoff
    payoff_file = os.path.join(os.getcwd(), 'data', 'payoff')
    if os.path.isfile(payoff_file):
      with open(payoff_file, 'rb') as f:
        self._payoff = pickle.load(f)
    else:
      self._payoff = Payoff()

    self._learning_agents = []

    pulsar = Pulsar(False)
    pulsar.call_build()    

    for idx in range(main_agents):
      ma_name = "main_agent:"+str(idx)
      ma_model_file, ma_agent_file, ma_player_file = get_agent_files(ma_name)
      pulsar.load(ma_model_file)
      ma_agent = Agent(pulsar.get_weights(), ma_agent_file)
      ma_agent.load()
      main_agent = MainPlayer(ma_agent, self._payoff, ma_player_file, name=ma_name)
      main_agent.load()
      self._learning_agents.append(main_agent)
      self._payoff.add_player(main_agent.checkpoint())

    for idx in range(main_exploiters):
      me_name = "main_exploit:"+str(idx)
      me_model_file, me_agent_file, me_player_file = get_agent_files(me_name)
      pulsar.load(me_model_file)
      me_agent = Agent(pulsar.get_weights(), me_agent_file)
      me_agent.load()
      main_exploiter = MainExploiter(me_agent, self._payoff, me_player_file, name=me_name)
      main_exploiter.load()
      self._learning_agents.append(main_exploiter)

    for idx in range(league_exploiters):
      le_name = "league_exploit:"+str(idx)
      le_model_file, le_agent_file, le_player_file = get_agent_files(le_name)
      pulsar.load(le_model_file)
      le_agent = Agent(pulsar.get_weights(), le_agent_file)
      le_agent.load()
      league_exploiters = LeagueExploiter(le_agent, self._payoff, le_player_file, name=le_name)
      league_exploiters.load()
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
