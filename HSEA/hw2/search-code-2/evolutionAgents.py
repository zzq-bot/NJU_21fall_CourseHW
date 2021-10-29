# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
A* search , run the following command:

> python pacman.py -l smallmaze -p EvolutionSearchAgent

"""

from game import Directions
from game import Agent
from game import Actions
from searchAgents import SearchAgent, PositionSearchProblem, FoodSearchProblem
import util
import time
import search


#########################################################
# This portion is written for you, but will only work   #
#     after you fill in parts of EvolutionSearchAgent   #
#########################################################

def positionHeuristic(state):
    '''
    A simple heuristic function for evaluate the fitness in task 1
    :param state:
    :return:
    '''
    return abs(state[0] - 1) + abs(state[1] - 1)


class EvolutionSearchAgent():
    def __init__(self, type='PositionSearchProblem', actionDim=10):
        '''
        This is the EvolutionSearchAgent, you should firstly finish the search.evolutionSearch
        '''
        self.searchType = globals()[type]
        self.actionDim = actionDim # dim of individuals
        self.T = None # iterations for evolution
        self.popSize = None # number of individuals in the population
        self.population = None


    def getFitness(self, state):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        return positionHeuristic(state)

    def mutation(self):
        pass

    def crossover(self):
        pass

    def generateLegalActions(self):
        '''
        generate the individuals with legal actions
        :return:
        '''
        pass

    def getActions(self, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param problem:
        :return: the best individual in the population
        '''
        pass

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.getActions(problem)  # Find a path

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions): # You may need to add some conditions for taking the action
            return self.actions[i]
        else: # You may need to use getAction multiple times
            self.actionIndex = 0
            problem = self.searchType(state)
            self.actions = self.getActions(problem)
            return Directions.STOP
