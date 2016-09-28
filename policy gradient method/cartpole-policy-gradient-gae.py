#!/usr/bin/env python
"""Script to solve cartpole control problem using policy gradient with neural network 
	function approximation
	Polcy used is softmax
	Value function is approximated using a multi layered pereceptron with tanh units
	Critic (value network) is updated using monte-carlo prediction
	Actor (softmax policy) is updated using Generalized Advantage for Advantage estimation"""
#Author: Mohak Bhardwaj
import gym
import tensorflow as tf
import numpy as np
print ("Packs loaded")

#Replay memory consists of multiple lists of state, action, next state, reward, return from state
#TODO: Search for best way to store replay memory
replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []

class Actor:
	def __init__(self, env, ):
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.action_space_n = self.action_space.n
		#Learning parameters
		self.learning_rate = 0.008
		#Declare tf graph
		self.graph = tf.Graph()
		#Build the graph when instantiated
		with self.graph.as_default():
			tf.set_random_seed(1234)
			self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
			self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

			#Inputs
			self.x = tf.placeholder("float", [None, len(self.observation_space.high)])#State input
			self.y = tf.placeholder("float") #Advantage input
			self.action_input = tf.placeholder("float", [None, self.action_space_n]) #Input action to return the probability associated with that action

			self.policy = self.softmax_policy(self.x, self.weights, self.biases) #Softmax policy
		
			self.log_action_probability = tf.reduce_sum(self.action_input*tf.log(self.policy))
			self.loss = -self.log_action_probability*self.y #Loss is score function times advantage
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
			#Initializing all variables
			self.init = tf.initialize_all_variables()
			print ("Policy Graph Constructed")
		self.sess = tf.Session(graph = self.graph)
		self.sess.run(self.init)
			

	def rollout_policy(self, timeSteps, episodeNumber):
		"""Rollout policy for one episode, update the replay memory and return total reward"""
		#First clear the current replay memory
		
		total_reward = 0
		curr_state = self.env.reset()
		episode_states = []
		episode_actions = []
		episode_rewards = []
		episode_next_states = []
		episode_return_from_states = []
		
		for time in xrange(timeSteps):
			action = self.choose_action(curr_state)
			next_state, reward, done, info = self.env.step(action)
			#Update the total reward
			total_reward += reward
			if done or time >= self.env.spec.timestep_limit :
				# print "Episode {} ended at step {} with total reward {}".format(episodeNumber, time, total_reward)
				break
			#Updating the memory
			curr_state_l = curr_state.tolist()
			next_state_l = next_state.tolist()
			if curr_state_l not in episode_states:
				episode_states.append(curr_state_l)
				episode_actions.append(action)
				episode_rewards.append(reward)
				episode_next_states.append(next_state_l)
				episode_return_from_states.append(reward)
				for i in xrange(len(episode_return_from_states)-1):
					episode_return_from_states[i] += reward
			else:
				#Iterate through the replay memory  and update the final return for all states 
				for i in xrange(len(episode_return_from_states)):
					episode_return_from_states[i] += reward
			curr_state = next_state
		self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states)
		return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward


	def update_policy(self, advantage_vectors):
		#Update the weights by running gradient descent on graph with loss function defined

		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		# print replay_states
		#TODO: Update to run for entire batch at once
		for i in xrange(len(replay_states)):
			
			states = replay_states[i]
			actions = replay_actions[i]
			advantage_vector = advantage_vectors[i]
			for j in xrange(len(states)):
				action = self.to_action_input(actions[j])

				state = np.asarray(states[j])
				state = state.reshape(1,4)

				_, error_value = self.sess.run([self.optim, self.loss], feed_dict={self.x: state, self.action_input: action, self.y: advantage_vector[j] })
	
	
	def softmax_policy(self, state, weights, biases):
		policy = tf.nn.softmax(tf.matmul(state, weights) + biases)
		return policy

	def choose_action(self, state):
		#Use softmax policy to sample
		state = np.asarray(state)
		state = state.reshape(1,4)
		softmax_out = self.sess.run(self.policy, feed_dict={self.x:state})
		action = np.random.choice([0,1], 1, replace = True, p = softmax_out[0])[0] #Sample action from prob density
		return action

	def update_memory(self, episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		#Using first visit Monte Carlo so total return from a state is calculated from first time it is visited 

		replay_states.append(episode_states)
		replay_actions.append(episode_actions)
		replay_rewards.append(episode_rewards)
		replay_next_states.append(episode_next_states)
		replay_return_from_states.append(episode_return_from_states)

	
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
	def __init__(self, env):
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.action_space_n = self.action_space.n
		self.n_input = len(self.observation_space.high)
		self.n_hidden_1 = 20
		#Learning Parameters
		# self.learning_rate = 0.008 
		self.learning_rate = 0.008
		# self.num_epochs = 12
		# self.batch_size = 150
		#20 150
		self.num_epochs = 3
		self.batch_size = 32
		#Discount factor
		self.discount = 0.96
		#Advantage function parameter to trade off
		self.lm = 0.92
		self.graph = tf.Graph()
		with self.graph.as_default():
			tf.set_random_seed(1234)
			self.weights = {
			'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
			'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
			}
			self.biases = {
    		'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
    		'out': tf.Variable(tf.random_normal([1]))
			}
			self.state_input = self.x = tf.placeholder("float", [None, len(self.observation_space.high)])#State input
			self.return_input = tf.placeholder("float") #Target return
			self.value_pred = self.multilayer_perceptron(self.state_input, self.weights, self.biases)			
			self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.return_input,2))			
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
			init = tf.initialize_all_variables()
		print("Value Graph Constructed")
		self.sess = tf.Session(graph = self.graph)
		self.sess.run(init)

		
	def multilayer_perceptron(self, x, weights, biases):
		#First hidden layer
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.tanh(layer_1)
		#Second hidden layer
		out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
		return out_layer

	def update_value_estimate(self):
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		#Monte Carlo prediction 
		state_batches, return_batches, num_batches = self.get_all_batches(replay_states, replay_actions)
		
		for epoch in xrange(self.num_epochs):	
			#Loop over all batches
			#We need to sample with replacement from the batch
			for i in xrange(num_batches):
				batch_state_input = state_batches[i]
				batch_return_input = return_batches[i]
				#Fit training data using minibatch
				self.sess.run(self.optim, feed_dict={self.state_input:batch_state_input, self.return_input:batch_return_input})
	

	def get_advantage_vector(self, states, rewards, next_states):
		#Return Generalized Advantage for particular state and action
		#Get value of current state
		# rewards = np.asarray(rewards)
		# next_states = np.asarray(next_states)
		advantage_vector = []
		for i in xrange(len(states)):		
			state = np.asarray(states[i])
			state = state.reshape(1,len(self.observation_space.high))
			#For Generalized Advantage we need all next states
			#and all next rewards
			next_states_all = next_states[i:]	
			rewards_all = rewards[i:]
			gae = self.get_gae(state, next_states_all, rewards_all)
			advantage_vector.append(gae)
		return advantage_vector

	def get_gae(self, state, next_states, next_rewards):
		#Calculates generalized advantage for a state given
		#the state and sequence of nex states and rewards
		gae = 0
		state_value = self.sess.run(self.value_pred, feed_dict={self.state_input:state})
		for i in xrange(len(next_states)):
			#Casting state to appropriate shape and calculating value from a 
			#forward pass
			next_s = np.asarray(next_states[i]).reshape(1,len(self.observation_space.high))
			next_s_val = self.sess.run(self.value_pred, feed_dict={self.state_input:next_s})
			#Using the formula A(GAE) = sigma(i=0 to infinity)(gamma*lambda)^i*d_V_i
			d_V_i = next_rewards[i] + self.discount*next_s_val - state_value
			adv_i = pow(self.discount*self.lm, i)*(d_V_i)
			gae += adv_i
			state_value = next_s_val
		return gae



	def get_all_batches(self, states_data, returns_data):
		#Return all mini-batches of transitions from replay data 
		all_states = []
		all_returns = []
		state_batches = []
		return_batches = []

		for i in xrange(len(states_data)):
			episode_states = states_data[i]
			episode_returns = returns_data[i]
			for j in xrange(len(episode_states)):
				all_states.append(episode_states[j])
				all_returns.append(episode_returns[j])
		
		all_states = np.asarray(all_states)
		all_returns = np.asarray(all_returns)
		
		#Determine the feasible batch size
		batch_size = self.batch_size

		if np.ma.size(all_states,0) < batch_size:
			batch_size = np.ma.size(all_states,0)
		num_batches = np.ma.size(all_states,0)/batch_size

		for batch in xrange(num_batches):
			if all_states.shape[0] >= batch_size:
				randidx = np.random.randint(all_states.shape[0], size=batch_size)
				batch_states = all_states[randidx, :]
				batch_returns = all_returns[randidx]
				all_states = np.delete(all_states, randidx, axis=0)
				all_returns = np.delete(all_returns, randidx, axis=0)
				state_batches.append(batch_states)
				return_batches.append(batch_returns)

		return state_batches, return_batches, num_batches


class ActorCriticLearner:
	def __init__(self, env, max_episodes, episodes_before_update):
		self.env = env
		self.actor = Actor(self.env)
		self.critic = Critic(self.env)
		#Learner parameters
		self.max_episodes = max_episodes
		self.episodes_before_update = episodes_before_update


	def learn(self):
		
		advantage_vectors = []
		sum_reward = 0
		update = True			
		for i in xrange(self.max_episodes):
			episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(200, i+1)	
			advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
			advantage_vectors.append(advantage_vector)
			sum_reward += episode_total_reward
			if (i+1)%self.episodes_before_update == 0:
				avg_reward = sum_reward/self.episodes_before_update
				print "Current {} episode average reward: {}".format(self.episodes_before_update, avg_reward)
				#In this part of the code I try to reduce the effects of randomness leading to oscillations in my 
				#network by sticking to a solution if it is close to final solution.
				#If the average reward for past batch of episodes exceeds that for solving the environment, continue with it
				if avg_reward >= 195: #This is the criteria for having solved the environment by Open-AI Gym
					update = False
				else:
					update = True

				if update:
					print "Updating"
					self.actor.update_policy(advantage_vectors)
					self.critic.update_value_estimate()
				else:
					print "Good Solution, not updating"
				#Delete the data collected so far
				del advantage_vectors[:]
				self.actor.reset_memory()
				sum_reward = 0
		
	
def main():
	env = gym.make('CartPole-v0')
	env.seed(1234)
	np.random.seed(1234)
	env.monitor.start('./cartpole-pg-experiment', force=True)
	#Learning Parameters
	max_episodes = 500
	episodes_before_update = 2


	ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update)
	ac_learner.learn()
	env.monitor.close()
	

if __name__=="__main__":
	main()