# Modification of Tensorflow's official implementation of attention_layer found at
# github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py
#
# Original implementation splits input into different heads, then recombines them.
# This version does not recombine the heads via concaternation. Instead, the outputs of
# each head are passed through a CONV1D layer to double the number of hidden units.
# Finally, we sum the output of each heads to form one single head.
#
# Use 2 heads if original dimension of hidden units needs to be retained.
#
# ==============================================================================
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout, train):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False, name="v")

    # Layers for convoluting the output of each head
    self.conv1ds = []
    for i in range(num_heads):
        self.conv1ds.append(tf.keras.layers.Conv1D(2 * hidden_size // self.num_heads, 1))

    self.output_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if self.train:
      weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Double hidden_size via 1D convolution with kernel size 1
    attention_heads = []
    for i, conv1d in enumerate(self.conv1ds):
        attention_heads.append(conv1d(attention_output[:, i, :, :]))
    
    attention_output = attention_heads[0]
    for i in range(1, len(attention_heads)):
        attention_output += attention_heads[i]

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)
