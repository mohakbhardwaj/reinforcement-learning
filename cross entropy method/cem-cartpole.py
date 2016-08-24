#!/usr/bin/env python
import gym 
import numpy
import operator

def sampleParams(mean, cov, n):
		#Get n samples of parameters from multivariate gaussian with mean and covariance
		return numpy.random.multivariate_normal(mean, cov, n)

def select_action(params, observation):
		#Return 0 or 1 depending on the parameters and observation
		w = params[:-1]
		b = params[-1]
		y = numpy.dot(w,observation) + b
		action = int(y < 0)
		return action


def eval_noisy(params, env, numEpisodes, t):
		#Evaluate the environment for using params for numEpisodes and return average  of noisy rewards obtained
		rewards = []
		for episode in xrange(numEpisodes):
			observation = env.reset()
			total_reward_in_episode = 0
			for time in xrange(t):
				env.render()
				action = select_action(params, observation)
				observation, reward, done, info = env.step(action) # take a random action
				total_reward_in_episode += reward
				if done:
					# print("Episode finished after {} timesteps with reward {}".format(t+1, total_reward_in_episode))
					rewards.append(total_reward_in_episode)
					break;
		return numpy.average(rewards)


def fit_gaussian(successors):
		#Write a function to fit a multivariate gaussian distribution on a set of numbers
		mean = numpy.mean(successors, axis=0)
		cov = numpy.cov(successors, rowvar=0)
		return mean, cov

class cem:
	def __init__(self, mean, cov, n, numIters, p, numEpisodes, env, t):
		self.mean = mean #Initialize the mean of policy parameters
		self.cov = cov #Initialize the covariance of policy parameters distribution
		self.n = n   #Number of samples to be taken from policy parameter distribution
		self.numIters = numIters #Number of iterations to run the cem for
		self.p = p #Top samples to be considered for next population
		self.numEpisodes = numEpisodes
		self.env = env
		self.t = t #Number of timesteps to run one episode for
	

	def elite_set(self):
		#Call sample params, followed by eval_noisy for all params
		#Select top p params and return them as elte set
		sampled_params = sampleParams(self.mean, self.cov, self.n)
		avg_rewards = []
		for params in sampled_params:
			avg_reward_for_params = eval_noisy(params, self.env, self.numEpisodes, self.t)
			avg_rewards.append((params, avg_reward_for_params))
		
		sorted_avg_rewards = sorted(avg_rewards, key=lambda tup: tup[1], reverse=True)
		return sorted_avg_rewards[0:self.p]

	def do_cross_entropy(self):
		#For numIters, call elite_set and update the mean and sigma of the distribution
		#Return the final mean 
		for iter in xrange(self.numIters):
			successors_and_rewards = self.elite_set()
			successors = []
			for successor_and_reward in successors_and_rewards:
				successors.append(successor_and_reward[0])
			self.mean, self.cov = fit_gaussian(successors)
		return self.mean, self.cov



def main():
	env = gym.make('CartPole-v0')
	env.seed(0)
	numpy.random.seed(0)
	env.monitor.start('./cartpole-cem-experiment-5')
	#Initial values for mean, covariance and other paramters for the cross entropy method
	initMean = numpy.array([0, 0, 0, 0, 0])
	initCov = numpy.zeros((5,5), float)
	numpy.fill_diagonal(initCov, 10000)
	n = 100
	p = 20
	numIters = 10
	numEpisodes = 5
	t = 200
	cemObject = cem(initMean, initCov, n, numIters, p, numEpisodes, env, t)
	
	#Let the training begin
	print "Begin Training"
	
	finalMean, finalStdDev = cemObject.do_cross_entropy()
	# print "Training Complete, moving on to testing ..."
	# print finalMean, finalStdDev
	
	# eval_noisy(finalMean, env, 500, t)
	env.monitor.close()
	# for i_episode in range(100):
	# 	observation = env.reset()
	# 	for t in range(100):
	# 	    env.render()
	# 	    # print(observation)
	# 	    action = env.action_space.sample()
	# 	    observation, reward, done, info = env.step(action) # take a random action
	# 	    if done:
	# 	    	print("Episode finished after {} timesteps".format(t+1))




if __name__=="__main__":
	main()
