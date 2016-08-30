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
		self.weights = np.random.randn(self.action_space_n,len(self.observation_space.high)) #Initialize random weight
		self.biases = np.random.randn(self.action_space_n)
		# #Learning parameters
		self.learning_rate = 0.1
	def rollout_policy(self, timeSteps, episodeNumber):
		"""Rollout policy for one episode, update the replay memory and return total reward"""
		#First clear the current replay memory
		# print "Weights: ", self.weights
		# print "Biases: ", self.biases
		self.reset_memory()
		total_reward = 0
		curr_state = self.env.reset()
		
		for time in xrange(timeSteps):
			self.env.render()	
			action = self.choose_action(curr_state)
			
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

	def update_weights(self, grad):
		# self.weights = np.add(self.weights, grad)
		weight_grad = grad[np.ix_([0,1,2,3, 5,6,7,8])]
		weight_grad = weight_grad.reshape(2,4)
		bias_grad = [0.0]*2
		bias_grad[0] = grad[4]
		bias_grad[0] = grad[9]
		bias_grad = np.asarray(bias_grad)
		self.weights = np.add(self.weights, self.learning_rate*weight_grad)
		self.biases = np.add(self.biases, self.learning_rate*bias_grad)
	

	def advantage_vector(self):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		return replay_return_from_states

	def gradient_estimate(self, params):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		log_policy_vector = []
		weights = params[np.ix_([0,1,2,3, 5,6,7,8])] #TODO: Remove hardcoding
		weights = weights.reshape(2,4)
		# biases = params[np.ix_([0,1], [4])]
		biases = [0.0]*2
		biases[0] = params[4]
		biases[1] = params[9]
		
		for i in xrange(len(replay_states)):	
			chosen_action = replay_actions[i]
			action_probs = self.softmax(replay_states[i], weights, biases)
			chosen_action_log_prob = ad.admath.log(action_probs[chosen_action])
			log_policy_vector.append(chosen_action_log_prob)

		grad_estimate_value = np.dot(np.asarray(log_policy_vector), self.advantage_vector())
		return grad_estimate_value


	def update_policy(self):

		
		grad_function, hess_function = ad.gh(self.gradient_estimate)
		
		curr_params = np.zeros(2*len(self.observation_space.high)+2)
		curr_params[0] = self.weights[0][0]
		curr_params[1] = self.weights[0][1]
		curr_params[2] = self.weights[0][2]
		curr_params[3] = self.weights[0][3]
		curr_params[4] = self.biases[0]
		curr_params[5] = self.weights[1][0]
		curr_params[6] = self.weights[1][1]
		curr_params[7] = self.weights[1][2]
		curr_params[8] = self.weights[1][3]
		curr_params[9] = self.biases[1]

		grad = grad_function(curr_params)
		# print "Grad: ", grad
		self.update_weights(grad)


	
	def softmax(self, state, weights, biases):
		prob_actions = [0]*self.action_space_n

		prob_actions = ad.admath.exp(np.add(np.dot(weights, state), biases))
		
		prob_actions = np.true_divide(prob_actions, sum(prob_actions))
		
		return prob_actions



	def choose_action(self, state):
		#Use softmax policy to sample
		prob_actions = self.softmax(state, self.weights, self.biases)
		action = np.random.choice([0,1],1,replace=True, p=prob_actions)
		return action[0]

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
		# print "Timestep: ", time
		# print replay_states
		# print replay_actions
		# print replay_return_from_states
	
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
	numEpisodes = 20000
	for i in xrange(numEpisodes):
		actor.rollout_policy(200, i+1)
		actor.update_policy()







if __name__=="__main__":
	main()