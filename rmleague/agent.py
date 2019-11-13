import os
import numpy as np
import tensorflow as tf
import pickle


class Agent(object):

  def __init__(self, weights, agent_file):
    self.steps = 0
    self.weights = weights
    self.agent_file = agent_file

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

  def step(self):
    self.steps += 1

  def save(self):
    with open(self.agent_file, 'wb') as f:
      pickle.dump(self.steps, f)
  
  def load(self):
    if os.path.isfile(self.agent_file):
      with open(self.agent_file, 'rb') as f:
        self.steps = pickle.load(f)
