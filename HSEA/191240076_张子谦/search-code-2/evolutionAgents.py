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
import random
import torch
from torch.distributions import Categorical
import math
from sklearn.cluster import KMeans


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

def is_back_and_forth(x, y):
    """return True if x and y is back and forth"""
    return (x == 0 and y == 1) or (x == 1 and y == 0) or (x == 2 and y == 3) or (x == 3 and y == 2)

id2action = {
    0:Directions.NORTH,
    1:Directions.SOUTH,
    2:Directions.EAST,
    3:Directions.WEST,
    4:Directions.STOP,
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
        self.fitness_cnt = 0
        self.pm = 0.2
        self.pc = None
        self.memory_pool = {}

    def getFitness(self, problem, individual):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        block = 0
        no_back_forth = 0
        done = False
        stop_idx = []
        path_reward = 0
        s = problem.getStartState()
        if s not in self.memory_pool.keys():
            self.memory_pool[s] = 1
        for i, ac in enumerate(individual):

            if problem.isGoalState(s):
                done =True
            #1、if back and forth ?
            assert len(individual) == self.actionDim, print(individual)
            if i+1 < len(individual) and not is_back_and_forth(individual[i], individual[i+1]):
                no_back_forth += 1

            ac = id2action[ac]
            x, y = s
            dx, dy = Actions.directionToVector(ac)
            nextx, nexty = int(x + dx), int(y + dy)
            #2、if cross the wall?
            if problem.walls[nextx][nexty]:
                block += 1
                stop_idx.append(i)
            else:
                s = (nextx, nexty)
                if s not in self.memory_pool.keys():
                    self.memory_pool[s] = 1
                path_reward += 5 / self.memory_pool[s]
                #path_reward += 10 / self.memory_pool[s] #for big maze
        #f = 1 / (positionHeuristic(s)+1)
        f = 1. * path_reward + 5. * no_back_forth - 10. * block
        f -= 2 * positionHeuristic(s)
        return f, done, stop_idx, s

    def parent_selection(self, fitness):
        """
        choose parents to mutation & crossover to produce offspring
        return: choosen parents idxs
        """
        probs = Categorical(logits=torch.FloatTensor(fitness))
        idxs = probs.sample((self.popSize,))
        idxs = idxs.squeeze()
        return idxs

    def survivor_selection(self, fitness, final_s):
        """
        use KMeans to seperate according to final_state of individual
        """
        cluster_res = KMeans(n_clusters=2).fit(final_s)
        group_res = cluster_res.predict(final_s)
        group0 = np.argwhere(group_res == 0).squeeze()
        group1 = np.argwhere(group_res == 1).squeeze()
        try:
            size0 = int(self.popSize * len(group0) / len(final_s))
            size1 = int(self.popSize * len(group1) / len(final_s))
            if size0 + size1  == self.popSize - 2:
                size0 += 1
                size1 += 1
            elif size0 + size1 == self.popSize - 1:
                size1 += 1
            assert size0 + size1 == self.popSize, print(size0 + size1, self.popSize)
            idxs0 = group0[np.argsort(fitness[group0])[-size0:]]
            idxs1 = group1[np.argsort(fitness[group1])[-size1:]]
            idxs = np.concatenate((idxs0, idxs1))
            return idxs
        except:
            return np.argsort(fitness)[-self.popSize:]

    def mutation(self, individual):
        for i in range(len(individual)):
            if np.random.random() < self.pm:
                x = np.random.randint(0, 4)
                individual[i] = x
        return individual

    def crossover(self, x, y):
        if self.pc is None:  # one point
            cross_point = np.random.randint(0, self.actionDim - 1)
            new_x = np.concatenate((x[:cross_point], y[cross_point:]))
            new_y = np.concatenate((y[:cross_point], x[cross_point:]))
            # print(new_x, new_y)
        return new_x, new_y

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
        for i in range(self.T):
            #print(i)
            # 1、parent selection
            fitness = []
            done = []
            final_s = []
            for j in range(len(self.population)):
                f, d, _, s = self.getFitness(problem, self.population[j])
                fitness.append(f)
                done.append(d)
                final_s.append(s)
            self.fitness_cnt += 1
            fitness = np.array(fitness, dtype=np.float64)
            done = np.array(done, dtype=np.bool)
            final_s = np.array(final_s)

            if np.any(done):
                print(f"find the target, use fitness {self.fitness_cnt} times")
                try:
                    best_idx = np.argwhere(done==True)[0][0]
                    choosen_individual = self.population[best_idx]
                    _, _, stop_idx, _ = self.getFitness(problem, choosen_individual)
                    choosen_individual[stop_idx] = 4
                    best_actions = [" "] * self.actionDim
                    for j, a in enumerate(choosen_individual):
                        best_actions[j] = id2action[a]
                    #print(best_actions)
                    return best_actions
                except:
                    print(np.argwhere(done==True))
                    best_idx = np.argwhere(done == True).squeeze()[0]
                    print(best_idx)
                    _, _, stop_idx, _ = self.getFitness(problem, choosen_individual)
                    choosen_individual[stop_idx] = 4
                    print(choosen_individual)
                    assert 0

            assert len(fitness) == self.popSize, print(len(fitness), self.popSize, len(self.population))
            # print(np.mean(fitness))
            self.fitness_cnt += 1
            idxs = self.parent_selection(fitness)
            idxs = np.array(idxs)
            np.random.shuffle(idxs)  # 打乱一下顺序
            assert len(idxs) == self.popSize
            parents = self.population[idxs]
            parents_fitness = fitness[idxs]
            parents_final_s = final_s[idxs]

            # 2、mutation
            mu_parents = []
            for x in parents:
                mu_parents.append(self.mutation(x))


            # 3、crossover
            offsprings = []
            for j in range(0, len(parents), 2):
                x = mu_parents[j]
                y = mu_parents[j + 1]
                offsprings += list(self.crossover(x, y))
            offsprings = np.array(offsprings)


            # 4、fitness evaluation
            off_fitness = []
            off_final_s = []
            for j in range(len(offsprings)):
                f, d, _, s = self.getFitness(problem, offsprings[j])
                off_fitness.append(f)
                off_final_s.append(s)
            self.fitness_cnt += 1
            off_fitness = np.array(off_fitness, dtype=np.float64)
            off_final_s = np.array(off_final_s)
            together_fitness = np.concatenate((parents_fitness, off_fitness))
            together_actions = np.concatenate((parents, offsprings))
            together_final_s = np.concatenate((parents_final_s, off_final_s))

            # 5、survivor selection
            survivor_idx = self.survivor_selection(together_fitness, together_final_s)
            self.population = together_actions[survivor_idx]

        #finally
        fitness = []
        stop_idxs = []
        for individual_idx in range(len(self.population)):
            f, _, stop, _ = self.getFitness(problem, self.population[individual_idx])
            fitness.append(f)
            stop_idxs.append(stop)
        best_idx = np.argmax(fitness)
        choosen_individual = self.population[best_idx]
        stop_idx = stop_idxs[best_idx]
        choosen_individual[stop_idx] = 4
        best_actions = [" "] * self.actionDim
        for j, a in enumerate(choosen_individual):
            best_actions[j] = id2action[a]
        #print(best_actions)
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
            problem = self.searchType(state)
            s = problem.getStartState()
            if s not in self.memory_pool.keys():
                self.memory_pool[s] = 1
            else:
                self.memory_pool[s] += 1
            return self.actions[i]
        else: # You may need to use getAction multiple times
            self.actionIndex = 0
            problem = self.searchType(state)
            self.actions = self.getActions(problem)
            return Directions.STOP


class FoodEvolutionSearchAgent(EvolutionSearchAgent):
    def __init__(self, actionDim=20):
        '''
        EvolutionSearchAgent for FoodSearchProblem
        '''
        super().__init__(actionDim=actionDim)
        self.searchType = FoodSearchProblem

    def getFitness(self, state, problem, individual):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        food = state.getFood()
        foodlist = food.asList()
        food_cnt = 0
        block = 0
        no_back_forth = 0
        done = False
        stop_idx = []
        s = problem.getStartState()[0]
        for i, ac in enumerate(individual):
            #if problem.isGoalState(s):
            if len(foodlist) == 0:
                done = True
                break
            # 1、if back and forth ?
            assert len(individual) == self.actionDim, print(individual)
            if i + 1 < len(individual) and not is_back_and_forth(individual[i], individual[i + 1]):
                no_back_forth += 1
            ac = id2action[ac]
            x, y = s
            dx, dy = Actions.directionToVector(ac)
            nextx, nexty = int(x + dx), int(y + dy)
            # 2、if cross the wall ?
            if problem.walls[nextx][nexty]:
                block += 1
                stop_idx.append(i)
            else:
                s = (nextx, nexty)
                if foodlist:
                    assert isinstance(s, type(foodlist[0]))
                    if s in foodlist:
                        food_cnt += 1
                        foodlist.remove(s)

        max_dis = 0
        if len(foodlist) > 0:
            distances = []
            for foodpos in foodlist:
                dist = abs(s[0] - foodpos[0]) + abs(s[1] - foodpos[1])
                distances.append(dist)
            max_dis = max(distances)
        f = 5 * food_cnt + 2 * no_back_forth - 2 * block - max_dis
        return f, done, stop_idx, s

    def survivor_selection_simple(self, fitness):
        return np.argsort(fitness)[-self.popSize:]

    def getActions(self, state, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param state, problem
        :return: the best individual in the population
        '''
        for i in range(self.T):
            # 1、parent selection
            fitness = []
            done = []
            final_s = []
            for j in range(len(self.population)):
                f, d, _, s = self.getFitness(state, problem, self.population[j])
                fitness.append(f)
                done.append(d)
                final_s.append(s)
            self.fitness_cnt += 1
            fitness = np.array(fitness, dtype=np.float64)
            done = np.array(done, dtype=np.bool)
            final_s = np.array(final_s)

            if np.any(done):
                print(f"find the target, use fitness {self.fitness_cnt} times")
                try:
                    best_idx = np.argwhere(done==True)[0][0]
                    choosen_individual = self.population[best_idx]
                    _, _, stop_idx, _ = self.getFitness(state, problem, choosen_individual)
                    choosen_individual[stop_idx] = 4
                    best_actions = [" "] * self.actionDim
                    for j, a in enumerate(choosen_individual):
                        best_actions[j] = id2action[a]
                    #print(best_actions)
                    return best_actions
                except:
                    print(np.argwhere(done==True))
                    best_idx = np.argwhere(done == True).squeeze()[0]
                    print(best_idx)
                    _, _, stop_idx, _ = self.getFitness(state, problem, choosen_individual)
                    choosen_individual[stop_idx] = 4
                    print(choosen_individual)
                    assert 0

            assert len(fitness) == self.popSize, print(len(fitness), self.popSize, len(self.population))
            # print(np.mean(fitness))
            self.fitness_cnt += 1
            idxs = self.parent_selection(fitness)
            idxs = np.array(idxs)
            np.random.shuffle(idxs)  # 打乱一下顺序
            assert len(idxs) == self.popSize
            parents = self.population[idxs]
            parents_fitness = fitness[idxs]
            parents_final_s = final_s[idxs]

            # 2、mutation
            mu_parents = []
            for x in parents:
                mu_parents.append(self.mutation(x))


            # 3、crossover
            offsprings = []
            for j in range(0, len(parents), 2):
                x = mu_parents[j]
                y = mu_parents[j + 1]
                offsprings += list(self.crossover(x, y))
            offsprings = np.array(offsprings)


            # 4、fitness evaluation
            off_fitness = []
            off_final_s = []
            for j in range(len(offsprings)):
                f, d, _, s = self.getFitness(state, problem, offsprings[j])
                off_fitness.append(f)
                off_final_s.append(s)
            self.fitness_cnt += 1
            off_fitness = np.array(off_fitness, dtype=np.float64)
            off_final_s = np.array(off_final_s)
            together_fitness = np.concatenate((parents_fitness, off_fitness))
            together_actions = np.concatenate((parents, offsprings))
            together_final_s = np.concatenate((parents_final_s, off_final_s))

            # 5、survivor selection
            #survivor_idx = self.survivor_selection(together_fitness, together_final_s)
            survivor_idx = self.survivor_selection_simple(together_fitness)
            self.population = together_actions[survivor_idx]

        #finally
        fitness = []
        stop_idxs = []
        for individual_idx in range(len(self.population)):
            f, _, stop, _ = self.getFitness(state, problem, self.population[individual_idx])
            fitness.append(f)
            stop_idxs.append(stop)
        best_idx = np.argmax(fitness)
        choosen_individual = self.population[best_idx]
        stop_idx = stop_idxs[best_idx]
        choosen_individual[stop_idx] = 4
        best_actions = [" "] * self.actionDim
        for j, a in enumerate(choosen_individual):
            best_actions[j] = id2action[a]
        #print(best_actions)
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
        self.actions = self.getActions(state, problem)  # Find a path

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
            problem = self.searchType(state)
            s = problem.getStartState()
            return self.actions[i]
        else: # You may need to use getAction multiple times
            self.actionIndex = 0
            problem = self.searchType(state)
            self.actions = self.getActions(state, problem)
            return Directions.STOP

class ClassicEvolutionSearchAgent(FoodEvolutionSearchAgent):
    def __init__(self, actionDim=5):
        '''
        EvolutionSearchAgent for ClassicProblem
        '''
        super().__init__(actionDim=actionDim)
        #next for
        #self.T = 50
        """self.popSize = 20
        self.population = np.random.randint(low=0, high=4, size=(self.popSize, self.actionDim))"""

    def getFitness(self, state, problem, individual):
        state_copy = state.deepCopy()
        food = state_copy.getFood()
        foodlist = food.asList()
        food_cnt = 0
        capsuleslist = state_copy.getCapsules()
        capsules_cnt = 0
        block = 0
        no_back_forth = 0
        done = False
        stop_idx = []
        location = problem.getStartState()[0]
        f = 0
        ghost_pos = state_copy.getGhostPositions()
        #dis_threshold = 3
        dis_threshold = 5
        near_cnt = 0
        start_score = state_copy.getScore()
        for i, ac in enumerate(individual):
            for ghost_p in ghost_pos:
                dis2ghost = abs(location[0]-ghost_p[0]) + abs(location[1]-ghost_p[1])
                if dis2ghost <= dis_threshold:
                    near_cnt += 1
            if state_copy.isWin():
                done = True
                break
            elif state_copy.isLose():
                f = -100000
                break
            assert len(individual) == self.actionDim, print(individual)
            if i + 1 < len(individual) and not is_back_and_forth(individual[i], individual[i + 1]):
                no_back_forth += 1
            ac = id2action[ac]
            x, y = location
            dx, dy = Actions.directionToVector(ac)
            nextx, nexty = int(x + dx), int(y + dy)
            if problem.walls[nextx][nexty]:
                block += 1
                stop_idx.append(i)
            else:
                state_copy = state_copy.generateSuccessor(0, ac)
                location = (nextx, nexty)
                if foodlist:
                    assert isinstance(location, type(foodlist[0]))
                    if location in foodlist:
                        food_cnt += 1
                        foodlist.remove(location)
                if capsuleslist:
                    assert isinstance(location, type(capsuleslist[0]))
                    if location in capsuleslist:
                        capsules_cnt += 1
                        capsuleslist.remove(location)
        min_dis = 0
        if len(foodlist) > 0:
            distances = []
            for foodpos in foodlist:
                dist = abs(location[0] - foodpos[0]) + abs(location[1] - foodpos[1])
                distances.append(dist)
            min_dis = min(distances)
        if f != -100000:
            f = 5 * food_cnt + 5 * capsules_cnt + 2 * no_back_forth - 2 * block - 10 * min_dis + 10.* (state_copy.getScore() - start_score) - 50 * near_cnt
        return f, done, stop_idx, location

    def getActions(self, state, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param state, problem
        :return: the best individual in the population
        '''
        for i in range(self.T):
            # 1、parent selection
            fitness = []
            done = []
            final_s = []
            for j in range(len(self.population)):
                f, d, _, s = self.getFitness(state, problem, self.population[j])
                fitness.append(f)
                done.append(d)
                final_s.append(s)
            self.fitness_cnt += 1
            fitness = np.array(fitness, dtype=np.float64)
            done = np.array(done, dtype=np.bool)
            final_s = np.array(final_s)

            if np.any(done):
                print(f"find the target, use fitness {self.fitness_cnt} times")
                try:
                    best_idx = np.argwhere(done==True)[0][0]
                    choosen_individual = self.population[best_idx]
                    _, _, stop_idx, _ = self.getFitness(state, problem, choosen_individual)
                    choosen_individual[stop_idx] = 4
                    best_actions = [" "] * self.actionDim
                    for j, a in enumerate(choosen_individual):
                        best_actions[j] = id2action[a]
                    #print(best_actions)
                    return best_actions
                except:
                    print(np.argwhere(done==True))
                    best_idx = np.argwhere(done == True).squeeze()[0]
                    print(best_idx)
                    _, _, stop_idx, _ = self.getFitness(state, problem, choosen_individual)
                    choosen_individual[stop_idx] = 4
                    print(choosen_individual)
                    assert 0

            assert len(fitness) == self.popSize, print(len(fitness), self.popSize, len(self.population))
            # print(np.mean(fitness))
            self.fitness_cnt += 1
            idxs = self.parent_selection(fitness)
            idxs = np.array(idxs)
            np.random.shuffle(idxs)  # 打乱一下顺序
            assert len(idxs) == self.popSize
            parents = self.population[idxs]
            parents_fitness = fitness[idxs]
            parents_final_s = final_s[idxs]

            # 2、mutation
            mu_parents = []
            for x in parents:
                mu_parents.append(self.mutation(x))


            # 3、crossover
            offsprings = []
            for j in range(0, len(parents), 2):
                x = mu_parents[j]
                y = mu_parents[j + 1]
                offsprings += list(self.crossover(x, y))
            offsprings = np.array(offsprings)


            # 4、fitness evaluation
            off_fitness = []
            off_final_s = []
            for j in range(len(offsprings)):
                f, d, _, s = self.getFitness(state, problem, offsprings[j])
                off_fitness.append(f)
                off_final_s.append(s)
            self.fitness_cnt += 1
            off_fitness = np.array(off_fitness, dtype=np.float64)
            off_final_s = np.array(off_final_s)
            together_fitness = np.concatenate((parents_fitness, off_fitness))
            together_actions = np.concatenate((parents, offsprings))
            together_final_s = np.concatenate((parents_final_s, off_final_s))

            # 5、survivor selection
            survivor_idx = self.survivor_selection(together_fitness, together_final_s)
            #survivor_idx = self.survivor_selection_simple(together_fitness)
            self.population = together_actions[survivor_idx]

        #finally
        fitness = []
        stop_idxs = []
        for individual_idx in range(len(self.population)):
            f, _, stop, _ = self.getFitness(state, problem, self.population[individual_idx])
            fitness.append(f)
            stop_idxs.append(stop)
        best_idx = np.argmax(fitness)
        choosen_individual = self.population[best_idx]
        stop_idx = stop_idxs[best_idx]
        choosen_individual[stop_idx] = 4
        best_actions = [" "] * self.actionDim
        for j, a in enumerate(choosen_individual):
            best_actions[j] = id2action[a]
        #print(best_actions)
        return best_actions
