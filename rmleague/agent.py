import os
import numpy as np
import tensorflow as tf
import pickle


class Agent(object):

  def __init__(self, weights):
    self.steps = 0
    self.weights = weights

  def initial_state(self):
    """Returns the hidden state of the agent for the start of an episode."""
    # Network details elided.
    return initial_state

  def set_weights(self, weights):
    self.weights = weights

  def get_weights(self):
    return self.weights

  def get_steps(self):
    """How many agent steps the agent has been trained for."""
    return self.steps

  def add_steps(self, steps):
    self.steps += steps
