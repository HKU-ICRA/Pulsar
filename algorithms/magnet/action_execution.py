import numpy as np
import tensorflow as tf


class ActionExecution(tf.keras.layers.Layer):

    def __init__(self):
        super(ActionExecution, self).__init__()
        # State 1
        self.conv1_state1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv1_pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2_state1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv2_pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        # Graph
        self.conv1_graph = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv1_graph_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2_graph = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
        # Final linear projection
        self.final_dense = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.final_dropout = tf.keras.layers.Dropout(rate=0.4)
        self.logits = tf.keras.layers.Dense(units=10)

    def call(self, state_1, graph, training=True):
        # State 1
        output_state1 = self.conv1_state1(state_1)
        output_state1 = self.conv1_pool1(output_state1)
        output_state1 = self.conv2_state1(output_state1)
        output_state1 = self.conv2_pool1(output_state1)
        output_state1 = tf.reshaoe(output_state1, [-1, ...])
        # Graph
        output_graph = self.conv1_graph(graph)
        output_graph = self.conv1_graph_pool(output_graph)
        output_graph = self.conv2_graph(output_graph)
        output_graph = tf.reshape(output_graph, [-1, ...])
        # Whole model
        whole_model = tf.concat([output_state1, output_graph], axis=1)
        # Final linear projection
        outputs = self.final_dense(whole_model)
        outputs = self.final_dropout(outputs, training)
        logits = self.logits(outputs)
        actions = f.argmax(input=logits, axis=1)
        probs = tf.nn.softmax(logits)
        return actions, probs
