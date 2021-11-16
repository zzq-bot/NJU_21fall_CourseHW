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
import torch

from game import Directions
from game import Agent
from game import Actions
from searchAgents import SearchAgent, PositionSearchProblem, FoodSearchProblem
import util
import time
import search
import numpy as np
import random
import torch
from torch.distributions import Categorical
import math
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
    #return abs(state[0] - 1) + abs(state[1] - 1)
    return int(math.sqrt((state[0]-1)**2)+math.sqrt((state[1]-1)**2))

id2action = {
    0:Directions.NORTH,
    1:Directions.SOUTH,
    2:Directions.EAST,
    3:Directions.WEST,
    #4:Directions.STOP
}

class EvolutionSearchAgent():
    def __init__(self, type='PositionSearchProblem', actionDim=10):
        '''
        This is the EvolutionSearchAgent, you should firstly finish the search.evolutionSearch
        '''
        self.searchType = globals()[type]
        self.actionDim = actionDim # dim of individuals
        self.T = 50 # iterations for evolution
        self.popSize = 20 # number of individuals in the population
        self.population = np.random.randint(low=0, high=4, size=(self.popSize, self.actionDim))
        self.fitness_cnt = 0
        self.pm = 0.2
        self.pc = None
        self.initial_reward = 1000
        self.memory = {}

    def getFitness(self, problem, actions):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        #s, c, idx_stop = search.evolutionSearch(problem, actions)
        s, actions, c = self.generateLegalActions(actions, problem)
        #h = np.mean([simulate(s, problem, 3) for _ in range(30)])
        return positionHeuristic(s) + self.memory[s], actions, positionHeuristic(s) == 0

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

    def crossover(self, x, y):
        if self.pc is None:#one point
            cross_point = np.random.randint(0, self.actionDim - 1)
            new_x = np.concatenate((x[:cross_point], y[cross_point:]))
            new_y = np.concatenate((y[:cross_point], x[cross_point:]))
            #print(new_x, new_y)
        return new_x, new_y

    def generateLegalActions(self, actions, problem):
        '''
        generate the individuals with legal actions
        :return:
        '''
        s = problem.getStartState()
        cost = 0
        for i, action in enumerate(actions):
            cost += 1
            if problem.isGoalState(s):
                break
            action = id2action[action]
            x, y = s
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not problem.walls[nextx][nexty]:
                s = (nextx, nexty)
            else:
                legal = []
                for id, a in id2action.items():
                    dx, dy = Actions.directionToVector(a)
                    nextx, nexty = int(x + dx), int(y + dy)
                    if not problem.walls[nextx][nexty]:
                        legal.append(id)
                choosen_id = np.random.choice(legal)
                dx, dy = Actions.directionToVector(id2action[choosen_id])
                s = (int(x+dx), int(y+dy))
                actions[i] = choosen_id
            if s not in self.memory.keys():
                self.memory[s] = 1
        return s, actions, cost

    def getActions(self, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param problem:
        :return: the best individual in the population
        '''
        for i in range(self.T):
            if i % 10 == 0:
                print("hhhhhh")
            # 1、parent selection
            fitness = []
            done = []
            for j, actions in enumerate(self.population):
                f, a, d = self.getFitness(problem, actions)
                fitness.append(f)
                done.append(d)
                self.population[j] = a
            fitness = np.array(fitness, dtype=np.float64)
            done = np.array(done, dtype=np.bool)
            if np.any(done):
                choosen_actions = self.population[np.random.choice(np.argwhere(done==True).reshape(1, -1)[0])]
                best_actions = [" "] * self.actionDim
                for j, a in enumerate(choosen_actions):
                    best_actions[j] = id2action[a]
                return best_actions
            assert len(fitness) == self.popSize
            #print(np.mean(fitness))
            self.fitness_cnt += 1
            fitness = np.max(fitness) - fitness + 10
            idxs = self.parent_selection(fitness)
            np.random.shuffle(idxs)#打乱一下顺序
            assert len(idxs) == self.popSize
            parents = self.population[idxs]

            #2、mutation
            print("mutation")
            mu_parents = []
            for x in parents:
                mu_parents.append(self.mutation(x))

            #3、crossover

            print("corssovers")
            offsprings = []
            for j in range(0, len(parents), 2):
                x = mu_parents[j]
                y = mu_parents[j+1]
                offsprings += list(self.crossover(x, y))
            offsprings = np.array(offsprings)

            #4、fitness evaluation

            print("fitness evaluation")
            off_fitness = []
            for j, actions in enumerate(offsprings):
                f, a, _ = self.getFitness(problem, actions)
                off_fitness.append(f)
                offsprings[j] = a
            off_fitness = np.array(off_fitness, dtype=np.float64)
            together_fitness = np.concatenate((fitness, off_fitness))
            together_actions = np.concatenate((parents, offsprings))

            #5、survivor selection

            print("survivor selection")
            survivor_idxs = self.survivor_selection(together_fitness)
            self.population = together_actions[survivor_idxs]
        fitness = np.array([self.getFitness(problem, actions)[0] for actions in self.population], dtype=np.float64)
        best_idx = np.argmin(fitness)
        best_actions = [" "] * self.actionDim
        for j, a in enumerate(self.population[best_idx]):
            best_actions[j] = id2action[a]
        print(best_actions)
        return best_actions

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
            a = self.actions[i]
            problem = self.searchType(state)
            s = problem.getStartState()
            dx, dy = Actions.directionToVector(a)
            nxt_x, nxt_y = (s[0]+dx, s[1]+dy)
            if (nxt_x, nxt_y) not in self.memory:
                self.memory[(nxt_x, nxt_y)] = 1
            else:
                self.memory[(nxt_x, nxt_y)] += 1
            return a
        else: # You may need to use getAction multiple times
            self.actionIndex = 0
            problem = self.searchType(state)
            self.actions = self.getActions(problem)
            return Directions.STOP
