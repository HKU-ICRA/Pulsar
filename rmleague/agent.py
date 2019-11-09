import numpy as np
import tensorflow as tf


class Agent(object):
  """Demonstrates agent interface.

  In practice, this needs to be instantiated with the right neural network
  architecture.
  """

  def __init__(self, initial_weights):
    self.steps = 0
    self.weights = initial_weights

  def initial_state(self):
    """Returns the hidden state of the agent for the start of an episode."""
    # Network details elided.
    return initial_state

  def set_weights(self, weights):
    self.weights = weights

  def get_steps(self):
    """How many agent steps the agent has been trained for."""
    return self.steps

  def step(self, observation, last_state):
    """Performs inference on the observation, given hidden state last_state."""
    # We are omitting the details of network inference here.
    # ...
    return action, policy_logits, new_state

  def unroll(self, trajectory):
    """Unrolls the network over the trajectory.

    The actions taken by the agent and the initial state of the unroll are
    dictated by trajectory.
    """
    # We omit the details of network inference here.
    return policy_logits, baselines
