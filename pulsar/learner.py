import tensorflow as tf


class Learner:
  """Learner worker that updates agent parameters based on trajectories."""

  def __init__(self, player, BATCH_SIZE):
    self.player = player
    self.BATCH_SIZE = BATCH_SIZE
    self.trajectories = []
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, beta1=0, beta2=0.99, epsilon=1e-5)

  def get_parameters():
    return self.player.agent.get_weights()

  def send_trajectory(self, trajectory):
    self.trajectories.append(trajectory)

  def update_parameters(self):
    trajectories = self.trajectories[:self.BATCH_SIZE]
    self.trajectories = self.trajectories[self.BATCH_SIZE:]
    loss = loss_function(self.player.agent, trajectories)
    self.player.agent.steps += num_steps(trajectories)
    self.player.agent.set_weights(self.optimizer.minimize(loss))

  @background
  def run(self):
    while True:
      if len(self.trajectories) > self.BATCH_SIZE:
        self.update_parameters()
