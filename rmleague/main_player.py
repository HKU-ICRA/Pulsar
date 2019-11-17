import os
import numpy as np

from rmleague.player import Player
from rmleague.historical import Historical
from rmleague.main_exploiter import MainExploiter
from rmleague.league_exploiter import LeagueExploiter
from rmleague.agent import Agent
from rmleague.player_utils import remove_monotonic_suffix, pfsp


class MainPlayer(Player):

  def __init__(self, agent, payoff, player_file, name):
    self.agent = Agent(agent.get_weights(), agent.agent_file)
    self._payoff = payoff
    self.player_file = player_file
    self.name = name
    self._checkpoint_step = 0

  def _pfsp_branch(self):
    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical)
    ]
    win_rates = self._payoff[self, historical]
    return np.random.choice(
        historical, p=pfsp(win_rates, weighting="squared")), True

  def _selfplay_branch(self, opponent):
    # Play self-play match
    if self._payoff[self, opponent] > 0.3:
      return opponent, False

    # If opponent is too strong, look for a checkpoint
    # as curriculum
    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent == opponent
    ]
    win_rates = self._payoff[self, historical]
    return np.random.choice(
        historical, p=pfsp(win_rates, weighting="variance")), True

  def _verification_branch(self, opponent):
    # Check exploitation
    exploiters = set([
        player for player in self._payoff.players
        if isinstance(player, MainExploiter)
    ])
    exp_historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent in exploiters
    ]
    win_rates = self._payoff[self, exp_historical]
    if len(win_rates) and win_rates.min() < 0.3:
      return np.random.choice(
          exp_historical, p=pfsp(win_rates, weighting="squared")), True

    # Check forgetting
    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical) and player.parent == opponent
    ]
    win_rates = self._payoff[self, historical]
    win_rates, historical = remove_monotonic_suffix(win_rates, historical)
    if len(win_rates) and win_rates.min() < 0.7:
      return np.random.choice(
          historical, p=pfsp(win_rates, weighting="squared")), True

    return None

  def get_match(self):
    coin_toss = np.random.random()

    # Make sure you can beat the League
    if coin_toss < 0.5:
      return self._pfsp_branch()

    main_agents = [
        player for player in self._payoff.players
        if isinstance(player, MainPlayer)
    ]
    opponent = np.random.choice(main_agents)

    # Verify if there are some rare players we omitted
    if coin_toss < 0.5 + 0.15:
      request = self._verification_branch(opponent)
      if request is not None:
        return request

    return self._selfplay_branch(opponent)

  def ready_to_checkpoint(self):
    steps_passed = self.agent.get_steps() - self._checkpoint_step
    #if steps_passed < 2e9:
    if steps_passed < 5e8:
      return False

    historical = [
        player for player in self._payoff.players
        if isinstance(player, Historical)
    ]
    win_rates = self._payoff[self, historical]
    return win_rates.min() > 0.7 or steps_passed > 2e9

  def checkpoint(self):
    self._checkpoint_step = self.agent.get_steps()
    return self._create_checkpoint()
  
  def get_agent(self):
    return self.agent
  
  def set_agent(self, agent):
    self.agent = agent

  def get_weights(self):
    return self.agent.get_weights()

  @property
  def agent_file(self):
    return self.agent.agent_file

  def save(self):
    with open(self.player_file, 'wb') as f:
      pickle.dump(self._checkpoint_step, f)
  
  def load(self):
    if os.path.isfile(self.player_file):
      with open(self.player_file, 'rb') as f:
        self._checkpoint_step = pickle.load(f)
    