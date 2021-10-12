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

> python pacman.py -p SearchAgent -a fn=astar

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
from search import Node
import numpy as np
#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='astar', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)
        #self.actionIndex = 0

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

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
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE ***"

    position, foodGrid = state
    foodlist = foodGrid.asList()
    if not foodlist:
        return 0
    distances = []
    for foodpos in foodlist:
        dist = abs(position[0] - foodpos[0]) + abs(position[1] - foodpos[1])
        #dist = uniformSearch(position, foodpos, problem.startingGameState)
        #dist = bfs(position, foodpos, problem.walls)
        distances.append(dist)
    return max(distances) + foodGrid.count()
    """def bfs_nearest():
        frontier = util.Queue()
        reached = set()
        frontier.push((position, 0))
        reached.add(position)
        while not frontier.isEmpty():
            pos, cost = frontier.pop()
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(pos[0] + dx), int(pos[1] + dy)
                if not problem.walls[next_x][next_y] and not (next_x, next_y) in reached:
                    if (next_x, next_y) in foodlist:
                        return (next_x, next_y), cost + 1
                    frontier.push(((next_x, next_y), cost + 1))
                    reached.add((next_x, next_y))

    nearest_food, nearest_cost = bfs_nearest()
    dist = 0
    for food in foodlist:
        dist = max(dist, abs(food[0] - nearest_food[0]) + abs(food[1] - nearest_food[1]))
    return nearest_cost + dist + foodGrid.count()"""

def uniformSearch(pos1, pos2, startingGameState):
    problem = PositionSearchProblem(startingGameState, goal=pos2, start=pos1, warn=False, visualize=False)
    return len(search.astar(problem))

def bfs(pos1, pos2, walls):
    if pos1 == pos2:
        return 0
    frontier = util.Queue()
    reached = set()
    frontier.push((pos1, 0))
    reached.add(pos1)
    while not frontier.isEmpty():
        pos, cost = frontier.pop()
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(pos[0] + dx), int(pos[1] + dy)
            if not walls[next_x][next_y] and not (next_x, next_y) in reached:
                if (next_x, next_y) == pos2:
                    return cost + 1
                frontier.push(((next_x, next_y), cost+1))
                reached.add((next_x, next_y))


class OneStepFoodSearchAgent(SearchAgent):
    def registerInitialState(self, state):
        #self.max_depth = 1000
        self.max_depth = 50      #设定探索astar探索次数
        self.actions = []
        self.find_best = False    #如果探索中astar已经找到最优解(吃掉所有食物，则从actions中直接获取
        self._expand = 0
        #self.heuristic = OneStepFoodHeuristic
        self.heuristic = task3Heuristic
        self.eps = 5e-2
    def getAction(self, state):
        """根据当前state进行一定深度的探索，得到"""
        if self.heuristic is OneStepFoodHeuristic and self.find_best and self.actions:
            self.actions.pop(0)
            if len(self.actions) == 1:
                print("Search nodes expanded: ", self._expand)
            return self.actions[0]
        if self.eps is not None and np.random.random()<self.eps:
            return np.random.choice(state.getLegalActions(0))
        self.find_best, self.actions = self.limited_depth_astar_Search(state)
        return self.actions[0]

    def limited_depth_astar_Search(self, state):
        node = (state.deepCopy(), 0, [])
        original_node = node
        frontier = util.PriorityQueue()
        frontier.update(node, self.heuristic(node[0]))
        reached = set()
        i = 0
        while not frontier.isEmpty() and i < self.max_depth:
            i += 1
            node = frontier.pop()
            if node[0].getFood().count() == 0:
                return True, node[2]
            if node[0] in reached:
                continue
            reached.add(node[0])
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                try:
                    nxt_state = node[0].deepCopy().generateSuccessor(0, action)
                    score = 0
                    cost = node[1] + 1
                    actions = node[2] + [action]
                    frontier.update((nxt_state, cost, actions), cost + self.heuristic(nxt_state))
                    self._expand += 1
                except:
                    continue
        return False, node[2]

def OneStepFoodHeuristic(state):
    pos = state.getPacmanPosition()
    food = state.getFood()

    foodlist = food.asList()
    if not foodlist:
        return 0
    distances = []
    for foodpos in foodlist:
        dist = abs(pos[0] - foodpos[0]) + abs(pos[1] - foodpos[1])
        distances.append(dist)
    return max(distances) + food.count()

def task3Heuristic(state):
    if state.isWin():
        return -1000000
    if state.isLose():
        return 1000000
    score = 0
    score += state.getNumFood()
    score += 100 * len(state.getCapsules())
    return -(state.getScore())