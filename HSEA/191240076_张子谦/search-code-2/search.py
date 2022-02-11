# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def simulate(state, problem, penalty):
    goal_x, goal_y = problem.goal
    now_x, now_y = state
    direction_x = 1 if goal_x - now_x > 0 else -1
    direction_y = 1 if goal_y - now_y > 0 else -1
    cost = abs(goal_x - now_x) + abs(goal_y - now_y)
    while now_x != goal_x and now_y != goal_y:
        r = random.randint(0, 1)
        if r == 0:
            now_x += direction_x
        else:
            now_y += direction_y
        if problem.walls[now_x][now_y]:
            cost += penalty
    while now_x != goal_x:
        now_x += direction_x
        if problem.walls[now_x][now_y]:
            cost += penalty
    while now_y != goal_y:
        now_y += direction_y
        if problem.walls[now_x][now_y]:
            cost += penalty
    return cost

def manhattanHeuristic(state, problem):
    assert problem is not None
    #print("use manhattan heuristic func")
    goal_x, goal_y = problem.goal
    now_x, now_y = state
    return abs(goal_x - now_x) + abs(goal_y - now_y)

def myHeuristic(state, problem=None):
    """
        you may need code other Heuristic function to replace  NullHeuristic
        """
    "*** YOUR CODE HERE ***"
    return 1.0 * manhattanHeuristic(state, problem)

class Node:
    def __init__(self, state, cost, actions):
        self.state = state
        self.cost = cost
        self.actions = actions


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first.

        Your search algorithm needs to return a list of actions that reaches the
        goal. Make sure to implement a graph search algorithm.

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print("Start:", problem.getStartState())
        print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        print("Start's successors:", problem.getSuccessors(problem.getStartState()))
        """
    "*** YOUR CODE HERE ***"

    random.seed(7)

    node = Node(problem.getStartState(), 0., [])
    frontier = util.PriorityQueue()
    frontier.update(node, heuristic(node.state, problem))
    reached = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            return node.actions
        if node.state in reached:
            continue
        reached.add(node.state)
        for child in problem.getSuccessors(node.state):
            s, act, c = child
            cost = node.cost + c
            actions = node.actions + [act]
            frontier.update(Node(s, cost, actions), cost + heuristic(s, problem))

from game import Directions
from game import Actions
id2action = {
    0:Directions.NORTH,
    1:Directions.SOUTH,
    2:Directions.EAST,
    3:Directions.WEST,
    4:Directions.STOP
}

def evolutionSearch(problem, actions):
    """execute actions and return cost"""
    s = problem.getStartState()
    cost = 0
    idxs_stop = []
    for i, action in enumerate(actions):
        if problem.isGoalState(s):
            break
        action = id2action[action]
        x, y = s
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if not problem.walls[nextx][nexty]:
            s = (nextx, nexty)
        else:
            idxs_stop.append(i)
        cost += 1
    return s, cost, idxs_stop

# Abbreviations
astar = aStarSearch