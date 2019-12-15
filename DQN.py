import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class DQN(tf.keras.Model):
	def __init__(self, num_actions):
		"""
		The DQN class that inherits from tf.keras.Model
		The forward pass calculates the policy (buy, sell, stay)

		:param num_actions: number of actions in an environment
		"""
		super(DQN, self).__init__()
		self.model = tf.keras.Sequential()
		self.model.add(Dense(50, activation='elu', use_bias=True))
		self.model.add(Dropout(rate=0.2))
		self.model.add(Dense(50, activation='elu', use_bias=True))
		self.model.add(Dropout(rate=0.2))
		self.model.add(Dense(num_actions, activation='softmax', use_bias=True))
		
		self.gamma = .99
		self.E = 300
		self.games = 1000
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005) # Optimizer
		# TODO: Define network parameters and optimizer
	  
	@tf.function
	def call(self, states):
		"""
		Performs the forward pass on a batch of states to generate the action probabilities.
		This returns a policy tensor of shape [episode_length, num_actions], where each row is a
		probability distribution over actions for each state.

		:param states: An [episode_length, state_size] dimensioned array
		representing the history of states of an episode
		:return: A [episode_length,num_actions] matrix representing the probability distribution over actions
		of each state in the episode
		"""
		return self.model(states)

	def loss(self, next_action, q_val, next_q_val, reward):
		"""
		Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

		:param states: A batch of states of shape [episode_length, state_size]
		:param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
		:param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
		:return: loss, a TensorFlow scalar
		"""
		td_error = q_val.numpy()
		td_error[0][next_action] = reward + self.gamma*(np.max(next_q_val))
		return tf.reduce_sum(tf.square(td_error - q_val))

