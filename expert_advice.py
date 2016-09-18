#!/usr/bin/env python
#Implements weighted majority and randomized weighted majority algorithms for prediction under
#expert advice on deterministic, stochastic and adverserial environments with three different experts 
import numpy as np
import matplotlib.pyplot as plt



class Nature:
	def __init__(self, num_experts):
		self.num_experts = num_experts
		self.experts = [Experts("optimistic"), Experts("pessismistic"), Experts("matchbased")]

	def get_observation(self, matchNumber):
		observation = []
		for expert in self.experts:
			#Get advice based on match number +1 as match number 0 is first match
			observation.append(expert.get_advice(matchNumber))
		return observation
	
	def get_label(self):
		label = 0
		return label

	def get_expert_losses(self, label, round_number):
		expert_losses = []
		for expert in self.experts:
			expert_losses.append(expert.expert_loss(label, round_number))
		return expert_losses

class StochasticNature(Nature):
	#Returns a random label from 0 or 1
	def get_label(self):
		random_int = np.random.randint(2)
		return random_int if random_int == 1 else -1

class DeterministicNature(Nature):
	#Returns the same answer at each timestep as of now
	def get_label(self):
		return 1

class AdverserialNature(Nature):
	def get_label(self, weights, observation):
		#Uses current weights and expert advice to give label
		#opposite of what your expected prediction might be
		#Predict according to weighted majority and return prediction opposite of it
		prediction = np.sign(np.sum(np.multiply(weights, observation)))
		return -prediction

#Class characterizing the different kinds of experts
#Currently there are the folowing kinds of experts possible:
#Optimistic (always win), Pessimistic(always lose), Match Based(base on round number)
class Experts:
	def __init__(self, expert_type):
		self.expert_type = expert_type
	def get_advice(self, round_number):
		if self.expert_type == "optimistic":
			return 1
		elif self.expert_type == "pessismistic":
			return -1
		elif self.expert_type == "matchbased":
			return 1 if (round_number+1)%2 == 0 else -1
	def expert_loss(self, true_label, round_number):
		return 0 if(true_label - self.get_advice(round_number)) == 0 else 1



class Learner:
	def __init__(self,nature, prediction_type, adverserial=False): 
		self.adverserial = adverserial
		self.nature = nature
		self.weights = [1]*self.nature.num_experts
		self.prediction_type = prediction_type
		#Learning parameters
		self.ita = 0.2

	#Generic function that can incorporate different types of learners
	def learn(self, total_prediction_rounds):
		#The general online learning paradigm is followed
		#Nature sends observation, learner makes prediction according 
		#to prediction scheme (WMA or RWMA) and nature gives label
		#(might not be true one if adverserial nature) 
		total_learner_loss = 0
		learner_loss_plt = []
		experts_loss_plt = []
		for round_number in xrange(total_prediction_rounds):
			#Ask nature for observation which is expert advice
			observation = self.nature.get_observation(round_number)
			#Here prediction is made based on prediction scheme
			if self.prediction_type == "weighted_majority":
				prediction = self.weighted_majority(observation)
			elif self.prediction_type == "randomized_weighted_majority":
				prediction = self.randomized_weighted_majority(observation, round_number)
			#Ask nature for label which is a diifferent function for adverserial
			if self.adverserial:
				label = self.nature.get_label(self.weights, observation)
			else:
				label = self.nature.get_label()
			#Now that the true label has been observed, calculate the loss 
			#and update weights
			#The loss is 0 if correct prediction has been made
			#otherwise loss is 1
			learner_loss = 0 if (label - prediction) == 0 else 1
			expert_losses = self.nature.get_expert_losses(label, round_number)
			
			#Update the total learner loss
			total_learner_loss += learner_loss
			self.update_weights(expert_losses)
			print "Prediction: {}, Label: {}, Learner Loss: {}, Weights: {}".format(prediction, label, learner_loss, self.weights)
			#Update data structures for plotting
			learner_loss_plt.append(learner_loss)
			experts_loss_plt.append(expert_losses)
		return total_learner_loss, learner_loss_plt, experts_loss_plt
	
	#Function provides prediction according to weighted majority algorithm
	def weighted_majority(self, observation):
		prediction = np.sign(np.sum(np.multiply(self.weights, observation)))
		return prediction
	#Function provides prediciton according to randomized weighted majority algorithm
	def randomized_weighted_majority(self, observation, round_number):
		#Index of expert to follow is sampled from multinomial distribution and 
		#prediction is done according ot the advice from that expert
		sum_weights = np.sum(self.weights)
		#Select index randomly from multinomial distribution by running 1 trial with 
		#weights as the probabilities
		idx = np.argmax(np.random.multinomial(1, np.divide(self.weights, sum_weights))) 
		prediction = self.nature.experts[idx].get_advice(round_number)
		return prediction
	
	#Function that updates weights 
	def update_weights(self,expert_losses):
		#Apply multiplicative update to the wieghts to get new weights		
		new_weights = [0]*self.nature.num_experts
		#Only the experts who make a wrong prediction should be penalized
		for idx, weight in np.ndenumerate(self.weights):			
			decrease_factor = 1 - (self.ita*expert_losses[idx[0]])
			new_weights[idx[0]] = weight*decrease_factor
		self.weights = new_weights


def calculate_average_regret(total_prediction_rounds, learner_loss, experts_loss):
	#First calculate cumulative losses
	experts_loss = np.asarray(experts_loss)
	#print experts_loss
	cumulative_learner_loss = []
	cumulative_experts_loss = []
	cl_experts = [0]*len(experts_loss[0])
	for i in xrange(total_prediction_rounds):
		cumulative_learner_loss.append(sum(learner_loss[0:i+1]))
		for k in xrange(len(experts_loss[0])):
			cl_experts[k] = sum(experts_loss[0:i+1,k])
		cumulative_experts_loss.append(list(cl_experts))
	#Calculate the regrets
	episode_regrets = []
	#print cumulative_learner_loss
	
	for i in xrange(total_prediction_rounds):
		best_expert_loss = min(cumulative_experts_loss[i])
		#print best_expert_loss, cumulative_learner_loss[i]
		episode_regrets.append((cumulative_learner_loss[i] - best_expert_loss)/(i+1.))
	return episode_regrets


#Function that plots loss of learner vs all experts at every timestep for an episode of total_prediction_rounds
def plot_episode_losses(total_prediction_rounds, learner_loss, experts_loss):
	
	#Plot losses during the episode in every step for learner and all the experts
	t = np.linspace(0, total_prediction_rounds-1, total_prediction_rounds)
	#Instantiate figure
	fig, axs = plt.subplots(len(experts_loss[0]),1, figsize=(12,10))
	#Plot the learner loss
	# plt.plot(t, learner_loss)
	#Iterate through the expert losses and plot them
	for i in xrange(len(experts_loss[0])):
		expert_i_loss = np.asarray(experts_loss)[:,[i]]
		axs[i].set_ylim([-0.5, 1.5])
		axs[i].plot(t, expert_i_loss, 'r', label="Expert {} episode loss".format(i))
		axs[i].plot(t, learner_loss, 'k', label="Learner episode loss")
		axs[i].legend()
		axs[i].set_xlabel("Prediction Rounds")
		axs[i].set_ylabel("Loss at timestep, {0,1}")
	plt.draw()

#Function that plots reget of learner vs all experts at every timestep for an episode of total_prediction_rounds
def plot_episode_regrets(total_prediction_rounds, episode_regrets):
	#Go about plotting episode regrets
	#Plot losses during the episode in every step for learner and all the experts
	t = np.linspace(0, total_prediction_rounds-1, total_prediction_rounds)
	#Instantiate figure
	# fig, axs = plt.subplots(len(episode_regrets),1, figsize=(12,10))
	#Iterate through the expert losses and plot them
	# for i in xrange(len(episode_regrets[0])):
	# 	expert_i_regret= np.asarray(episode_regrets)[:,[i]]
	# 	# axs[i].set_ylim([-100, 100])
	# 	axs[i].plot(t, expert_i_regret, 'r', label="Regret vs Expert {}".format(i))
	# 	axs[i].legend()
	# 	axs[i].set_xlabel("Prediction Rounds")
	# 	axs[i].set_ylabel("Average Regret at timestep")
	plt.figure()
	plt.plot(t, episode_regrets, 'k', label = "Average Regret")
	plt.xlabel("Prediction Rounds")
	plt.ylabel("Average Regret")
	plt.draw()


def main():

	total_prediction_rounds = 100
	#Choose the kind of nature and nuber of experts in that nature (3 or 4)
	nature = StochasticNature(3)
	#Choose kind of learner and whether nature is adverserial
	weighted_majority_learner = Learner(nature, "randomized_weighted_majority", False)
	#Lear for one episode of length total_prediction_rounds
	episode_learner_loss, episode_learner_loss_plt, episode_experts_loss_plt = weighted_majority_learner.learn(total_prediction_rounds)
	print "Episode Loss = {}".format(episode_learner_loss)
	#Plot the episode losses
	plot_episode_losses(total_prediction_rounds, episode_learner_loss_plt, episode_experts_loss_plt)
	#Calculae the regrets of learner vs all the experts
	episode_regrets = calculate_average_regret(total_prediction_rounds, episode_learner_loss_plt, episode_experts_loss_plt)
	
	#Plot the rerets of learner vs all the experts
	plot_episode_regrets(total_prediction_rounds, episode_regrets)
	#At the end call plt.show() to ensure that the graph windows don't close
	plt.show()
	
if __name__ == "__main__":
	main()
