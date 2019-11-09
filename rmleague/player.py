import numpy as np

from rmleague.historical import Historical


class Player(object):
  
  def get_match(self):
    pass

  def ready_to_checkpoint(self):
    return False

  def _create_checkpoint(self):
    return Historical(self, self.payoff)

  @property
  def payoff(self):
    return self._payoff

  def checkpoint(self):
    raise NotImplementedError
