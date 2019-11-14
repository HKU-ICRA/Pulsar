from rmleague.player import Player
from rmleague.agent import Agent


class Historical(Player):

  def __init__(self, agent, payoff):
    self._agent = Agent(agent.get_weights(), agent.agent_file)
    self._payoff = payoff
    self._parent = agent

  @property
  def parent(self):
    return self._parent

  def get_agent(self):
    return self._agent

  def get_match(self):
    raise ValueError("Historical players should not request matches")

  def ready_to_checkpoint(self):
    return False
