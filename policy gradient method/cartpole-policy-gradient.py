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
print ("Packs loaded")


class Actor:
	def __init__():
		self.weights = #Initialize random weights


	def rollout_policy(env, numEpisodes, episodeTime):
		"""Rollout policy for specified number of episodes and return average reward obtained"""
		total_reward = 0
		avg_reward = 0
		for episode in xrange(numEpisodes):
			curr_state = env.reset()
			for time in xrange(episodeTime):
				env.render()
				action = self.sample_from_policy(curr_state)
				curr_state, reward, done, info = env.step(action)
				total_reward += reward
				if done:
					break
		avg_reward = total_reward/numEpisodes
		return avg_reward

	def update_policy():

	def sample_from_policy(state):

class Critic:
	def __init__():
		self.n_input = 4 #TODO replace with observation space size
		self.n_hidden_1 = 100
		self.n_hidden_2 = 50
		self.n_hidden_3 = 25
		self.weights = {
			'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
			'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
			'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]))
			'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
		}
		self.biases = {
    		'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
    		'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
    		'b3': tf.Variable(tf.random_normal([self.n_hidden_3]))
    		'out': tf.Variable(tf.random_normal([1]))
		}

	def construct_graph():	
		#Graph input
		x = tf.placeholder("float", [None, self.n_input])
		y = tf.placeholder("float")
		pred = self.multilayer_perceptron(x, self.weights, self.biases)
		cost = 
		optimizer = 
	def multilayer_perceptron(x, weights, biases):
		#First hidden layer
		layer_1 = tf.add(tf.matmul(x, self.weights['h1'], self.biases['b1']))
		layer_1 = tf.nn.............
		#Second hidden layer
		layer_2 = 

		#Third hidden layer
		layer_3 = 

		#Output layer
		out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
		return out_layer
	def 


def ActorCriticLearner():

	

def main():
	env = gym.make('CartPole-v0')
	env.seed(1234)
	np.random.seed(1234)
	env.monitor.start('./cartpole-pg-experiment-1')
	env.render()



if __name__=="__main__":
	main()