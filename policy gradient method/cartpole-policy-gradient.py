#!/usr/bin/env python
"""Script to solve cartpole control problem using policy gradient with neural network 
	function approximation
	Polcy used is softmax
	Value function is approximated using a multi layered pereceptron
	Critic (value network) is updated using monte-carlo 
	Actor (softmax policy) is updated using TD(0) for Advantage estimation"""
import gym
import tensorflow as tf
import numpy as np
from collections import defaultdict
print ("Packs loaded")

#Replay memory consists of multiple lists of state, action, next state, reward, return from state
#TODO: Search for best way to store replay memory
replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []
class Actor:
	def __init__(self, env):
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.action_space_n = self.action_space.n
		self.weights = np.random.randn(self.action_space_n,len(self.observation_space.high)) #Initialize random weight

	def rollout_policy(self, timeSteps):
		"""Rollout policy for one episode, update the replay memory and return total reward"""
		#First clear the current replay memory
		self.reset_memory()
		total_reward = 0
		curr_state = self.env.reset()
		
		for time in xrange(timeSteps):
			self.env.render()	
			action = self.sample_from_policy(curr_state)
			next_state, reward, done, info = self.env.step(action)
			#Update the replay memory
			self.update_memory(time, curr_state.tolist(), action, reward, next_state.tolist())
			#Update the total reward
			total_reward += reward
			if done:
				break
			curr_state = next_state
		return total_reward

	def update_policy():
		return 0

	def sample_from_policy(self, state):
		#Use softmax policy to sample
		return self.action_space.sample()

	def update_memory(self, time, curr_state, action, reward, next_state):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		#Using first visit Monte Carlo so total return from a state is calculated from first time it is visited 
		if curr_state not in replay_states:
			replay_states.append(curr_state)
			replay_actions.append(action)
			replay_rewards.append(reward)
			replay_next_states.append(next_state)
			replay_return_from_states.append(reward)
			for i in xrange(len(replay_return_from_states)-1):
				replay_return_from_states[i] += reward
		else:
			#Iterate through the replay memory  and update the final return for all states 
			for i in xrange(len(replay_return_from_states)):
				replay_return_from_states[i] += reward
		print "Timestep: ", time
		print replay_states
		print replay_actions
		print replay_return_from_states
	
	def reset_memory(self):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]
		



# class Critic:
# 	def __init__():
# 		self.n_input = 4 #TODO replace with observation space size
# 		self.n_hidden_1 = 100
# 		self.n_hidden_2 = 50
# 		self.n_hidden_3 = 25
# 		self.weights = {
# 			'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
# 			'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
# 			'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]))
# 			'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
# 		}
# 		self.biases = {
#     		'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
#     		'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
#     		'b3': tf.Variable(tf.random_normal([self.n_hidden_3]))
#     		'out': tf.Variable(tf.random_normal([1]))
# 		}

# 	def construct_graph():	
# 		#Graph input
# 		x = tf.placeholder("float", [None, self.n_input])
# 		y = tf.placeholder("float", [None, 1])
# 		pred = self.multilayer_perceptron(x, self.weights, self.biases)
# 		cost = 
# 		optimizer = 
# 	def multilayer_perceptron(x, weights, biases):
# 		#First hidden layer
# 		layer_1 = tf.add(tf.matmul(x, self.weights['h1'], self.biases['b1']))
# 		layer_1 = tf.nn.............
# 		#Second hidden layer
# 		layer_2 = 

# 		#Third hidden layer
# 		layer_3 = 

# 		#Output layer
# 		out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
# 		return out_layer
# 	def 


# def ActorCriticLearner():

	
def main():
	env = gym.make('CartPole-v0')
	env.seed(1234)
	np.random.seed(1234)
	# env.monitor.start('./cartpole-pg-experiment-1')
	# env.render()
	actor = Actor(env)
	# critic = Critic()
	actor.rollout_policy(200)




if __name__=="__main__":
	main()