#!/usr/bin/env python
#Implements weighted majority and randomized weighted majority algorithms for prediction under
#expert advice on deterministic, stochastic and adverserial environments with three different experts 
import numpy as np
class Nature:
	def __init__(self, num_experts):
		self.num_experts = num_experts
		self.experts = [Experts("optimistic"), Experts("pessismistic"), Experts("matchbased")]

	def get_observation(self, matchNumber):
		observation = []
		for expert in self.experts:
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
		return -1

class AdverserialNature(Nature):
	def get_label(self, weights, observation):
		#Uses current weights and expert advice to give label
		#opposite of what your expected prediction might be
		#Predict according to weighted majority and return prediction opposite of it
		prediction = np.sign(np.sum(np.multiply(weights, observation)))
		return -prediction

class Experts:
	def __init__(self, expert_type):
		self.expert_type = expert_type
	def get_advice(self, round_number):
		if self.expert_type == "optimistic":
			return 1
		elif self.expert_type == "pessismistic":
			return -1
		elif self.expert_type == "matchbased":
			return 1 if round_number%2 == 0 else -1
	def expert_loss(self, true_label, round_number):
		return 0 if(true_label - self.get_advice(round_number)) == 0 else 1


class Learner:
	def __init__(self,nature, prediction_type, adverserial=False): 
		self.adverserial = adverserial
		self.nature = nature
		self.weights = [1]*self.nature.num_experts
		self.prediction_type = prediction_type
		#Learning parameters
		self.ita = 0.1

	#Generic function that can incorporate different types of learners
	def learn(self, total_prediction_rounds):
		#The general online learning paradigm is followed
		#Nature sends observation, learner makes prediction according 
		#to prediction scheme (WMA or RWMA) and nature gives label
		#(might not be true one if adverserial nature) 
		total_learner_loss = 0
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
		return total_learner_loss
	
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


def main():

	total_prediction_rounds = 100
	nature_1 = AdverserialNature(3)
	weighted_majority_learner = Learner(nature_1, "randomized_weighted_majority", True)
	episode_learner_loss = weighted_majority_learner.learn(total_prediction_rounds)
	print "Episode Loss = {}".format(episode_learner_loss)
	
if __name__ == "__main__":
	main()