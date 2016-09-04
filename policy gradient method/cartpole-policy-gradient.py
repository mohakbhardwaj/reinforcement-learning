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
import ad
import ad.admath

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
		#Learning parameters
		self.learning_rate = 0.01
		#Declare tf graph
		self.graph = tf.Graph()
		#Build the graph when instantiated
		with self.graph.as_default():
			self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
			self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

			#Inputs
			self.x = tf.placeholder("float", [None, len(self.observation_space.high)])#State input
			self.y = tf.placeholder("float") #Advantage input
			self.action_input = tf.placeholder("float", [None, self.action_space_n]) #Input action to return the probability associated with that action

			self.policy = self.softmax_policy(self.x, self.weights, self.biases) #Softmax policy
		
			self.log_action_probability = tf.reduce_sum(self.action_input*tf.log(self.policy))
			self.loss = -self.log_action_probability*self.y #Loss is score function times advantage
			self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
			#Initializing all variables
			self.init = tf.initialize_all_variables()
			print ("Policy Graph Constructed")
		self.sess = tf.Session(graph = self.graph)
		self.sess.run(self.init)
			

	def rollout_policy(self, timeSteps, episodeNumber):
		"""Rollout policy for one episode, update the replay memory and return total reward"""
		#First clear the current replay memory
		self.reset_memory()
		total_reward = 0
		curr_state = self.env.reset()
		
		for time in xrange(timeSteps):
			self.env.render()	
			action = self.choose_action(curr_state)
			# action = self.env.action_space.sample()
			# print "Action selected: ", action
			next_state, reward, done, info = self.env.step(action)
			#Update the replay memory
			self.update_memory(time, curr_state.tolist(), action, reward, next_state.tolist())
			#Update the total reward
			total_reward += reward
			if done or time > self.env.spec.timestep_limit :
				print "Episode {} ended at step {} with total reward {}".format(episodeNumber, time, total_reward)
				break
			curr_state = next_state
		return total_reward


	def update_policy(self):
		#Update the weights by running gradient descent on graph with loss function defined

		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		#TODO: Update to run for entire batch at once
		for i in xrange(len(replay_states)):
			action = self.to_action_input(replay_actions[i])
			# print "Action coming out: ", action
			# print "Return from state: ", replay_return_from_states[i]
			state = np.asarray(replay_states[i])
			state = state.reshape(1,4)
			# softmax_out = self.sess.run(self.policy, feed_dict={self.x:state})
			# # print "Softmax output: ", softmax_out
			# log_action_prob_out = self.sess.run(self.log_action_probability, feed_dict={self.x:state, self.action_input: action})
			# # print "Log prob: ", log_action_prob_out
			# loss_out = self.sess.run(self.loss, feed_dict={self.x:state, self.action_input: action, self.y: replay_return_from_states[i]})
			# # print "Loss: ", loss_out
			_, error_value = self.sess.run([self.optim, self.loss], feed_dict={self.x: state, self.action_input: action, self.y: replay_return_from_states[i] })
			# print "Error: ", error_value
		# print "Model after episode: Weights {}, Biases {} ".format(self.sess.run(self.weights), self.sess.run(self.biases))
	
	def softmax_policy(self, state, weights, biases):
		policy = tf.nn.softmax(tf.matmul(state, weights) + biases)
		return policy

	def choose_action(self, state):
		#Use softmax policy to sample
		state = np.asarray(state)
		state = state.reshape(1,4)
		softmax_out = self.sess.run(self.policy, feed_dict={self.x:state})
		# print "Softmax output: ",  softmax_out[0]
		# action = softmax_out[0].tolist().index(max(softmax_out[0])) #Choose best possible action
		action = np.random.choice([0,1], 1, replace = True, p = softmax_out[0])[0] #Sample action from prob density
		return action

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

	
	def reset_memory(self):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]
	
	def to_action_input(self, action):
		action_input = [0]*self.action_space_n
		# print "Action going in: ", action
		action_input[action] = 1
		action_input = np.asarray(action_input)
		action_input = action_input.reshape(1, self.action_space_n)
		return action_input




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
		self.graph = tf.Graph()
		construct_graph()

	def construct_graph():	
		#Graph input

		# x = tf.placeholder("float", [None, self.n_input])
		# y = tf.placeholder("float", [None, 1])
		# pred = self.multilayer_perceptron(x, self.weights, self.biases)
		# cost = 
		# optimizer = 
		
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


# def ActorCriticLearner():

	
def main():
	env = gym.make('CartPole-v0')
	env.seed(1234)
	np.random.seed(1234)
	# env.monitor.start('./cartpole-pg-experiment-1')
	# env.render()
	actor = Actor(env)
	# critic = Critic()
	numEpisodes = 10000
	global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
	for i in xrange(numEpisodes):
		actor.rollout_policy(200, i+1)	
		actor.update_policy()
	
	
		








if __name__=="__main__":
	main()