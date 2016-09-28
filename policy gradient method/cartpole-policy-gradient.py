#!/usr/bin/env python
"""Script to solve cartpole control problem using policy gradient with neural network 
	function approximation
	Polcy used is softmax
	Value function is approximated using a multi layered pereceptron with tanh units
	Critic (value network) is updated using monte-carlo prediction
	Actor (softmax policy) is updated using TD(0) for Advantage estimation"""
import gym
import tensorflow as tf
import numpy as np


print ("Packs loaded")

#Replay memory consists of multiple lists of state, action, next state, reward, return from state
#[TODO: Search for best way to store replay memory]
replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []

class Actor:
	def __init__(self, env, discount = 0.90, learning_rate = 0.01):
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.action_space_n = self.action_space.n
		#Learning parameters
		self.learning_rate = learning_rate
		self.discount = discount
		#Declare tf graph
		self.graph = tf.Graph()
		#Build the graph when instantiated
		with self.graph.as_default():
			tf.set_random_seed(1234)
			self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
			self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

			#Neural Network inputs
			#The types of inputs possible include: state, advantage, action(to return probability of executing that action)
			self.x = tf.placeholder("float", [None, len(self.observation_space.high)])#State input
			self.y = tf.placeholder("float") #Advantage input
			self.action_input = tf.placeholder("float", [None, self.action_space_n]) #Input action to return the probability associated with that action

			#Current policy is a simple softmax policy since actions are discrete in this environment
			self.policy = self.softmax_policy(self.x, self.weights, self.biases) #Softmax policy
			#The following are derived directly from the formula for gradient of policy 
			self.log_action_probability = tf.reduce_sum(self.action_input*tf.log(self.policy))
			self.loss = -self.log_action_probability*self.y #Loss is score function times advantage
			#Use Adam Optimizer to optimize
			#[TODO: Add Trust Region Policy Optimization(TRPO)]
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
			#Initializing all variables
			self.init = tf.initialize_all_variables()
			print ("Policy Graph Constructed")
		
		#Declare a TF session and initialize it
		self.sess = tf.Session(graph = self.graph)
		self.sess.run(self.init)
			

	def rollout_policy(self, timeSteps, episodeNumber):
		"""Rollout policy for one episode, update the replay memory and return total reward"""
		total_reward = 0
		curr_state = self.env.reset()
		#Initialize lists in order to store episode data
		episode_states = []
		episode_actions = []
		episode_rewards = []
		episode_next_states = []
		episode_return_from_states = []
		
		for time in xrange(timeSteps):
			#Choose action based on current policy
			action = self.choose_action(curr_state)
			#Execute the action in the environment and observe reward
			next_state, reward, done, info = self.env.step(action)
			#Update the total reward
			total_reward += reward
			if done or time >= self.env.spec.timestep_limit :
				# print "Episode {} ended at step {} with total reward {}".format(episodeNumber, time, total_reward)
				break
			
			#Add state, action, reward transitions to containers for episode data
			#[TODO: Store discounted return instead of just return to test]
			curr_state_l = curr_state.tolist()
			next_state_l = next_state.tolist()
			if curr_state_l not in episode_states:
				episode_states.append(curr_state_l)
				episode_actions.append(action)
				episode_rewards.append(reward)
				episode_next_states.append(next_state_l)
				episode_return_from_states.append(reward)
				for i in xrange(len(episode_return_from_states)-1):
					#Here multiply the reward by discount factor raised to the power len(episode_return_from_states)-1-i
					# episode_return_from_states[i] += reward
					episode_return_from_states[i] += pow(self.discount,len(episode_return_from_states)-1-i)*reward
			else:
				#Iterate through the replay memory and update the final return for all states, i.e don't add the 
				#state if it is already there but update reward for other states
				for i in xrange(len(episode_return_from_states)):
					episode_return_from_states[i] += pow(self.discount,len(episode_return_from_states)-i)*reward
		
			curr_state = next_state
		
		#Update the global replay memory
		self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states)
		return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward


	def update_policy(self, advantage_vectors):
		"""Updates the policy weights by running gradient descent on one state at a time"""
		#[TODO: Try out batch gradient descent in this case as well]
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		
		for i in xrange(len(replay_states)):
			
			states = replay_states[i]
			actions = replay_actions[i]
			advantage_vector = advantage_vectors[i]
			for j in xrange(len(states)):
				action = self.to_action_input(actions[j])

				state = np.asarray(states[j])
				state = state.reshape(1,len(self.observation_space.high))

				_, error_value = self.sess.run([self.optim, self.loss], feed_dict={self.x: state, self.action_input: action, self.y: advantage_vector[j]})
	
	
	def softmax_policy(self, state, weights, biases):
		"""Defines softmax policy for tf graph"""
		policy = tf.nn.softmax(tf.matmul(state, weights) + biases)
		return policy

	def choose_action(self, state):
		"""Chooses action from the crrent policy and weights"""
		state = np.asarray(state)
		state = state.reshape(1,len(self.observation_space.high))
		softmax_out = self.sess.run(self.policy, feed_dict={self.x:state})
		action = np.random.choice([0,1], 1, replace = True, p = softmax_out[0])[0] #Sample action from prob density
		return action

	def update_memory(self, episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states):
		"""Updates the global replay memory"""
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		#Using first visit Monte Carlo so total return from a state is calculated from first time it is visited 

		replay_states.append(episode_states)
		replay_actions.append(episode_actions)
		replay_rewards.append(episode_rewards)
		replay_next_states.append(episode_next_states)
		replay_return_from_states.append(episode_return_from_states)

	
	def reset_memory(self):
		"""Resets the global replay memory"""
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]
	
	def to_action_input(self, action):
		"""Utility function to convert action to a format suitable for the neura networ input"""
		action_input = [0]*self.action_space_n
		# print "Action going in: ", action
		action_input[action] = 1
		action_input = np.asarray(action_input)
		action_input = action_input.reshape(1, self.action_space_n)
		return action_input




class Critic:
	"""Defines the critic network and functions to evaluate the current policy using Monte Carlo"""
	def __init__(self, env, discount = 0.90, learning_rate = 0.008):
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.action_space_n = self.action_space.n
		self.n_input = len(self.observation_space.high)
		self.n_hidden_1 = 20
		#Learning Parameters
		self.learning_rate = learning_rate 
		self.discount = discount
		self.num_epochs = 20   
		self.batch_size = 32 
		self.graph = tf.Graph()
		#Neural network is a Multi-Layered perceptron with one hidden layer containing tanh units
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
		"""Constructs the multilayere perceptron model"""
		#First hidden layer
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.tanh(layer_1)
		#Output Layer
		out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
		return out_layer


	def update_value_estimate(self):
		"""Uses mini batch gradient descent to update the value estimate"""
		global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
		#Monte Carlo prediction 
		batch_size = self.batch_size
		if np.ma.size(replay_states) < batch_size:
			batch_size = np.ma.size(replay_states)

		for epoch in xrange(self.num_epochs):
			total_batch = np.ma.size(replay_states)/batch_size
			#Loop over all batches
			for i in xrange(total_batch):
				batch_state_input, batch_return_input = self.get_next_batch(batch_size, replay_states, replay_return_from_states)
				#Fit training data using batch
				self.sess.run(self.optim, feed_dict={self.state_input:batch_state_input, self.return_input:batch_return_input})
	

	def get_advantage_vector(self, states, rewards, next_states):
		"""Returns TD(0) Advantage for particular state and action"""	
		
		advantage_vector = []
		for i in xrange(len(states)):
			state = np.asarray(states[i])
			state = state.reshape(1,len(self.observation_space.high))
			next_state = np.asarray(next_states[i])
			next_state = next_state.reshape(1,len(self.observation_space.high))
			reward = rewards[i]
			state_value = self.sess.run(self.value_pred, feed_dict={self.state_input:state})
			next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input:next_state})
			#This follows directly from the forula for TD(0)
			advantage = reward + self.discount*next_state_value - state_value
			advantage_vector.append(advantage)

		return advantage_vector


	def get_next_batch(self, batch_size, states_data, returns_data):
		"""Return mini-batch of transitions from replay data sampled with replacement"""
		all_states = []
		all_returns = []
		for i in xrange(len(states_data)):
			episode_states = states_data[i]
			episode_returns = returns_data[i]
			for j in xrange(len(episode_states)):
				all_states.append(episode_states[j])
				all_returns.append(episode_returns[j])
		all_states = np.asarray(all_states)
		all_returns = np.asarray(all_returns)
		randidx = np.random.randint(all_states.shape[0], size=batch_size)
		batch_states = all_states[randidx, :]
		batch_returns = all_returns[randidx]
		return batch_states, batch_returns



class ActorCriticLearner:
	def __init__(self, env, max_episodes, episodes_before_update, discount, policy_learning_rate, value_learning_rate):
		self.env = env
		self.actor = Actor(self.env, discount, policy_learning_rate)
		self.critic = Critic(self.env, discount, value_learning_rate)	

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
	#Hyper Parameters
	max_episodes = 300
	episodes_before_update = 2
	discount = 0.85
	policy_learning_rate = 0.01
	value_learning_rate = 0.008


	ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update, discount, policy_learning_rate, value_learning_rate)
	ac_learner.learn()
	env.monitor.close()
	

if __name__=="__main__":
	main()