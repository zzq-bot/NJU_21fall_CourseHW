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
import numpy as np
import torch
from torch.distributions import Categorical


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

id2action = {
    0:Directions.NORTH,
    1:Directions.SOUTH,
    2:Directions.EAST,
    3:Directions.WEST,
    4:Directions.STOP
}

class EvolutionSearchAgent():
    def __init__(self, type='PositionSearchProblem', actionDim=20):
        '''
        This is the EvolutionSearchAgent, you should firstly finish the search.evolutionSearch
        '''
        self.searchType = globals()[type]
        self.actionDim = actionDim # dim of individuals
        self.T = 100 # iterations for evolution
        self.popSize = 40 # number of individuals in the population
        self.population = np.random.randint(low=0, high=4, size=(self.popSize, self.actionDim))
        #self.stop_time = np.ones(self.popSize) * (self.actionDim-1)
        self.fitness_cnt = 0
        self.pm = 0.2
        self.pc = None

    def getFitness(self, problem, individual_idx):
        """
        evaluate the individual
        """
        individual = self.population[individual_idx]
        s = problem.getStartState()
        done = False
        stop_time = self.actionDim-1
        for i, ac in enumerate(individual):
            if problem.isGoalState(s):
                done =True
                break
            ac = id2action[ac]
            x, y = s
            dx, dy = Actions.directionToVector(ac)
            nextx, nexty = int(x + dx), int(y + dy)
            if not problem.walls[nextx][nexty]:
                s = (nextx, nexty)
            else:
                stop_time = i
                break
        f = 1 / (positionHeuristic(s)+1)
        return f, done, stop_time

    def parent_selection(self, fitness):
        """
        choose parents to mutation & crossover to produce offspring
        return: choosen parents idxs
        """
        probs = Categorical(logits=torch.FloatTensor(fitness))
        idxs = probs.sample((self.popSize,))
        idxs = idxs.squeeze()
        return idxs

    def survivor_selection(self, fitness):
        """
        fitness based; only choose best individuals from current and offspring
        return: choosen individuals idxs( of concanecate(parents, offspring)
        """
        idxs = np.argsort(fitness)[-self.popSize:]
        return idxs

    def mutation(self, individual):
        for i in range(len(individual)):
            if np.random.random() < self.pm:
                x = np.random.randint(0, 4)
                individual[i] = x
        return individual

    def crossover(self, x_idx, y_idx, stop_time):
        x = self.population[x_idx]
        y = self.population[y_idx]
        #print(stop_time[x_idx], stop_time[y_idx])
        max_valid_len = min(stop_time[x_idx], stop_time[y_idx])
        if self.pc is None:  # one point
            cross_point = np.random.randint(0, max_valid_len+1)
            new_x = np.concatenate((x[:cross_point], y[cross_point:]))
            new_y = np.concatenate((y[:cross_point], x[cross_point:]))
        return new_x, new_y

    def generateLegalActions(self, individual_idx):
        '''
        当pacman在执行这个动作序列时碰到墙壁，就让它停止
        '''
        pass

    def getActions(self, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param problem:
        :return: the best individual in the population
        '''

        for i in range(self.T):
            population_stop_time = np.ones(self.popSize) * (self.actionDim - 1)
            offsprings_stop_time = np.ones(self.popSize) * (self.actionDim - 1)
            # 1、parent selection
            fitness = []
            done = []
            for individual_idx in range(len(self.population)):
                f, d, s = self.getFitness(problem, individual_idx)
                fitness.append(f)
                done.append(d)
                population_stop_time[individual_idx] = s
            fitness = np.array(fitness, dtype=np.float64)
            done = np.array(done, dtype=np.bool)
            """if np.any(done):
                choosen_actions = self.population[np.random.choice(np.argwhere(done == True).reshape(1, -1)[0])]
                best_actions = [" "] * self.actionDim
                for j, a in enumerate(choosen_actions):
                    best_actions[j] = id2action[a]
                return best_actions"""
            assert len(fitness) == self.popSize

            self.fitness_cnt += 1
            idxs = self.parent_selection(fitness)
            np.random.shuffle(idxs)  # 打乱一下顺序
            assert len(idxs) == self.popSize
            parents = self.population[idxs]
            parents_stop_time = population_stop_time[idxs]

            # 2、mutation
            #print("mutation")
            mu_parents = parents.copy()
            for j, x in enumerate(parents):
                mu_parents[j] = self.mutation(x)

            # 3、crossover

            #print("corssovers")
            offsprings = []
            for j in range(0, len(mu_parents), 2):
                offsprings += list(self.crossover(j, j+1, parents_stop_time))
            offsprings = np.array(offsprings)

            # 4、fitness evaluation

            #print("fitness evaluation")
            off_fitness = []
            for individual_idx in range(len(offsprings)):
                f, _, s = self.getFitness(problem, individual_idx)
                off_fitness.append(f)
                offsprings_stop_time[individual_idx] = s
            off_fitness = np.array(off_fitness, dtype=np.float64)
            together_fitness = np.concatenate((fitness, off_fitness))
            together_actions = np.concatenate((parents, offsprings))

            # 5、survivor selection

            print("survivor selection")
            survivor_idxs = self.survivor_selection(together_fitness)
            self.population = together_actions[survivor_idxs]

        #finally
        fitness = []
        stop_time = []
        for individual_idx in range(len(self.population)):
            f, _, s = self.getFitness(problem, individual_idx)
            fitness.append(f)
            stop_time.append(s)
        best_idx = np.argmax(fitness)
        stop = stop_time[best_idx]
        best_actions = [" "] * self.actionDim
        for j, a in enumerate(self.population[best_idx]):
            best_actions[j] = id2action[a]
        print(best_actions[:stop])
        return best_actions[:stop]

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
