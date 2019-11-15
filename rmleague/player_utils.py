import numpy as np


def remove_monotonic_suffix(win_rates, players):
  if not list(win_rates):
    return win_rates, players

  for i in range(len(win_rates) - 1, 0, -1):
    if win_rates[i - 1] < win_rates[i]:
      return win_rates[:i + 1], players[:i + 1]

  return np.array([]), []


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
