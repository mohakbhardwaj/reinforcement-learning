# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


<<<<<<< HEAD
import mdp, util, heapq
=======
import mdp, util
>>>>>>> 93ae94a0e26e625c232243463bc9e8388b46a14a

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in xrange(iterations):
          
          values_k = self.values.copy()
          for s in states:
            
            qsa = []

            if not self.mdp.isTerminal(s):
              
              for a in self.mdp.getPossibleActions(s):
                q_action = 0
                sd_psd = self.mdp.getTransitionStatesAndProbs(s, a)
                for tup in sd_psd:
                  sd = tup[0]
                  psd = tup[1]
                  rsasd = self.mdp.getReward(s,a,sd)
                  q_action += psd*(rsasd + self.discount*values_k[sd])
                qsa.append(q_action)
              self.values[s] = max(qsa)


            # #LAO* Code
            # #set initial values of start state to heuristic of manhattan distance
            # start = self.mdp.getStartState()
            # goal = states[-1]
            # self.values[start] = util.manhattanDistance(start,goal)
            # #Initialize fringe with start
            # fringe = [(self.values[start],start, start)]
            
            # graph = [(self.values[start],start,start)]
            # while len(fringe) > 0:
            #   Zset = FIND(fringe,goal) #expand best element in fringe, 
            #                                        #initialize values of new states 
            #                                         #in the expanded graph
            #                                         #return subset states to be revised
            #   fringe = REVISE(Zset)       #Value iteration on the states and 
            #                                         # revise graph



            

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        q_action = 0
        sd_psd = self.mdp.getTransitionStatesAndProbs(state, action)
        for tup in sd_psd:
            sd = tup[0]
            psd = tup[1]

            rsasd = self.mdp.getReward(state,action,sd)
            q_action += psd*(rsasd + self.discount*self.values[sd])
        return q_action
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None
        actions = self.mdp.getPossibleActions(state)
        qsa = []
        for a in actions:
          q_action = self.computeQValueFromValues(state,a)
          qsa.append((a, q_action))
        qsa.sort(key = lambda tup:tup[1])
        a = qsa[-1][0]
        return a

       # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
# <<<<<<< HEAD

#     def FIND(fringe,goal):
#       best_state = fringe.pop()
#       actions = self.mdp.getPossibleActions(best_state)
#       expand_list = []
#       for action in possible_actions:
#         expand_list.append(self.mdp.getTransitionStatesAndProbs(best_state))
#       for elem in expand_list:
#         self.values[elem[0]] = util.manhattanDistance(elem[0], goal)
#       Z = [start]
#       if best_state[1] != start:
#         while best_state[2] !=start:
#           Z.insert(0,best_state[2])
#       return 


#     def REVISE():
# =======
# >>>>>>> 93ae94a0e26e625c232243463bc9e8388b46a14a
