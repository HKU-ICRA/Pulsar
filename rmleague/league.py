import numpy as np


def pfsp(win_rates, weighting="linear"):
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x)**2,
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


class League(object):

  def __init__(self,
               initial_agents,
               main_agents=1,
               main_exploiters=1,
               league_exploiters=2):
    self._payoff = Payoff()
    self._learning_agents = []
    for race in initial_agents:
      for _ in range(main_agents):
        main_agent = MainPlayer(race, initial_agents[race], self._payoff)
        self._learning_agents.append(main_agent)
        self._payoff.add_player(main_agent.checkpoint())

      for _ in range(main_exploiters):
        self._learning_agents.append(
            MainExploiter(race, initial_agents[race], self._payoff))
      for _ in range(league_exploiters):
        self._learning_agents.append(
            LeagueExploiter(race, initial_agents[race], self._payoff))
    for player in self._learning_agents:
      self._payoff.add_player(player)

  def update(self, home, away, result):
    return self._payoff.update(home, away, result)

  def get_player(self, idx):
    return self._learning_agents[idx]

  def add_player(self, player):
    self._payoff.add_player(player)
