import numpy as np
import tensorflow as tf


class RunningMeanStd(tf.keras.layers.Layer):
    """
        Running mean, std to be used with Pulsar.
        Variables are saved to allow for restoration.
        Args:
            shape: shape of variable to be kept track of
            epsilon: count rate
    """
    def __init__(self, shape, epsilon=1e-4):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self._mean = tf.Variable(name="mean", initial_value=np.zeros(shape, np.float64))
        self._var = tf.Variable(name="var", initial_value=np.ones(shape, np.float64))
        self._count = tf.Variable(name="count", initial_value=np.full((), epsilon, np.float64))
        self._set_mean_var_count()
    
    def _set_mean_var_count(self):
        # Detach gradients
        self.mean, self.var, self.count = self._mean.read_value(), self._var.read_value(), self._count.read_value()

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        """
            Args:
                mean: current mean
                var: current var
                count: current count
                batch_mean: mean of whole batch
                batch_var: var of whole batch
                batch_count: no. of samples in batch
            Returns:
                Updated mean, var, count
        """
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count
    
    def call(self, inputs):
        """
            Args:
                inputs: the variable to be tracked
        """
        batch_mean = np.mean(inputs, axis=0)
        batch_var = np.var(inputs, axis=0)
        batch_count = inputs.shape[0]
        new_mean, new_var, new_count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        self._mean.assign(new_mean)
        self._var.assign(new_var)
        self._count.assign(new_count)
        self._set_mean_var_count()
