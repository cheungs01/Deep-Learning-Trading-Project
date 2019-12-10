import os
import sys
import gym
import gym_anytrading
import pylab
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline
from DQN import DQN
import argparse

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)
parser = argparse.ArgumentParser(description='DCGAN')
parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
					help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')
parser.add_argument('--mode', type=str, default='REINFORCE',
					help='Can be "REINFORCE" or "REINFORCE_BASELINE" or "DQN"')
args = parser.parse_args()

def visualize_data_rewards(total_rewards, runtype):
	"""
	Takes in array of rewards from each episode, visualizes reward over episodes.

	:param rewards: List of rewards from all episodes
	"""
	x_values = pylab.arange(0, len(total_rewards), 1)
	y_values = total_rewards
	pylab.plot(x_values, y_values)
	pylab.xlabel('episodes')
	pylab.ylabel('cumulative rewards')
	pylab.title(runtype + ' Reward by Episode')
	pylab.grid(True)
	pylab.savefig(runtype + "_rewards.png")
	
def visualize_data_profits(total_profits, runtype):
	"""
	Takes in array of rewards from each episode, visualizes reward over episodes.

	:param rewards: List of rewards from all episodes
	"""
	x_values = pylab.arange(0, len(total_rewards), 1)
	y_values = total_rewards
	pylab.plot(x_values, y_values)
	pylab.xlabel('episodes')
	pylab.ylabel('cumulative profits')
	pylab.title(runtype + ' Profit by Episode')
	pylab.grid(True)
	pylab.savefig(runtype + "_profits.png")


def discount(rewards, discount_factor=.99):
	"""
	Takes in a list of rewards for each timestep in an episode, 
	and returns a list of the sum of discounted rewards for
	each timestep. Refer to the slides to see how this is done.

	:param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
	:param discount_factor: Gamma discounting factor to use, defaults to .99
	:return: discounted_rewards: list containing the sum of discounted rewards for each timestep in the original
	rewards list
	"""
	# TODO: Compute discounted rewards
	discounted_rewards = rewards.copy()
	for i in range(len(rewards)-2, -1, -1):
		discounted_rewards[i]=discount_factor*discounted_rewards[i+1]+rewards[i] # Discounts rewards in future and adds it to present reward
	return discounted_rewards

def generate_trajectory(env, model):
	"""
	Generates lists of states, actions, and rewards for one complete episode.

	:param env: The openai gym environment
	:param model: The model used to generate the actions
	:return: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps
	in the episode
	"""
	states = []
	actions = []
	rewards = []
	state = env.reset()
	done = False
	info = 1
	while not done:
		# TODO:
		# 1) use model to generate probability distribution over next actions
		# 2) sample from this distribution to pick the next action
		states.append(np.reshape(state,[-1]))
		next_actions = model.call(tf.convert_to_tensor([np.reshape(state,[-1])],dtype=tf.float32)) # Gets the probabilities for the next actions
		action = np.random.choice(a=env.action_space.n,p=np.reshape(next_actions, [-1])) # Picks an action at random
		actions.append(action)
		state, rwd, done, info = env.step(action) # Uses this action
		rewards.append(rwd)
	print("Information:",info)
	print("Max profit:",env.max_possible_profit())
	return states, actions, rewards, info

def train_dqn(env, model, iteration): 
	epsilon = model.E / (iteration + model.E)
	state = env.reset()
	rewards = []
	done = False
	info = 1
	while not done:
		with tf.GradientTape() as tape: 
			q_vals = model.call(tf.convert_to_tensor([np.reshape(state,[-1])],dtype=tf.float32))
			next_action = tf.math.argmax(q_vals, axis=1)
			if np.random.rand(1) < epsilon:
				next_action = env.action_space.sample()
				
			next_state, reward, done, info = env.step(next_action)
			next_q_vals = model.call(tf.convert_to_tensor([np.reshape(next_state,[-1])],dtype=tf.float32))
			rewards.append(reward)
			loss = model.loss(next_action, q_vals, next_q_vals, reward)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		state = next_state
	reward_sum = 0
	for rwd in rewards: 
		reward_sum += rwd
	print("Information:",info)
	print("Max profit:",env.max_possible_profit())
	return reward_sum, info.total_profit

def random_call(env): 
	state = env.reset()
	rewards = []
	done = False
	info = 1
	while not done: 
		next_action = env.action_space.sample()
		next_state, reward, done, info = env.step(next_action)
		rewards.append(reward)
		state = next_state
	reward_sum = 0
	for rwd in rewards: 
		reward_sum += rwd
	print("Information:",info)
	print("Max profit:",env.max_possible_profit())
	return reward_sum, info.total_profit
	

def train(env, model):
	"""
	This function should train your model for one episode.
	Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
	and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
	Make sure to return the total reward for the episode.

	:param env: The openai gym environment
	:param model: The model
	:return: The total reward for the episode
	"""

	# TODO:
	# 1) Use generate trajectory to run an episode and get states, actions, and rewards.
	# 2) Compute discounted rewards.
	# 3) Compute the loss from the model and run backpropagation on the model.
	states, actions, rewards, info = generate_trajectory(env, model) # Get states, actions, and rewards from one iteration
	with tf.GradientTape() as tape:
		loss = model.loss(tf.cast(tf.convert_to_tensor(states),dtype=tf.float32), tf.convert_to_tensor(actions), tf.cast(tf.convert_to_tensor(discount(rewards)),dtype=tf.float32))
	# Gets the gradients for this batch
	gradients = tape.gradient(loss, model.trainable_variables)

	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Does gradient descent
	reward_sum = 0
	for i in rewards:
		reward_sum+=i # Adds rewards
	return reward_sum, info.total_profit

def main():

	env = gym.make("forex-v0") # environment
	state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
	num_actions = env.action_space.n

	# Initialize model
	if args.mode == "REINFORCE":
		model = Reinforce(state_size, num_actions) 
	elif args.mode == "REINFORCE_BASELINE":
		model = ReinforceWithBaseline(state_size, num_actions)
	elif args.mode == "DQN": 
		model = DQN(2)

	# TODO: 
	# 1) Train your model for 650 episodes, passing in the environment and the agent. 
	# 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
	# 3) After training, print the average of the last 50 rewards you've collected.
	rewards = []
	profits = []
	if args.mode == "REINFORCE" or args.mode == "REINFORCE_BASELINE": 
		try:
			with tf.device('/device:' + args.device):
				for i in range(650):
					print(i)
					reward, profit = train(env, model)
					rewards.append(reward)
					profits.append(profit)
		except RuntimeError as e:
			print(e)
	else if args.mode == "DQN":
		try:
			with tf.device('/device:' + args.device):
				print(args.device)
				for i in range(1000): 
					print(i)
					reward, profit = train_dqn(env, model, i)
					rewards.append(reward)
					profits.append(profit)
		except RuntimeError as e:
			print(e)
	else: 
		for i in range(1000): 
			print(i)
			reward, profit = random_call(env)
			rewards.append(reward)
			profits.append(profit)
	print("Average of last 50 rewards:",tf.reduce_mean(rewards[-50:])) # Prints average of final 50 rewards
	# TODO: Visualize your rewards.
	visualize_data_rewards(rewards, args.mode)
	visualize_data_profits(profits, args.mode)

if __name__ == '__main__':
	main()

