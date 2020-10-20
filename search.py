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
import heapq
from util import Stack
from util import Queue

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



def depthFirstSearch(problem):
   
    "*** YOUR CODE HERE ***"
    stack = Stack()

    visited_node = [] # Visited states
    path = [] 

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Start from the beginning node ais there any exists node because we are searching by DFS #
    stack.push((problem.getStartState(),[]))

    while(True):
        # empty stack#
        if stack.isEmpty():
            return []
        # Get informations of current state #
        sk,path = stack.pop() # Take position and path
        visited_node.append(sk)
        #  reaching the goal #
        if problem.isGoalState(sk):
            return path
        # Get successors of current state #
        succ = problem.getSuccessors(sk)
        # Add new states in stack and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited_node:
                    newPath = path + [item[1]] # Calculate new path
                    stack.push((item[0],newPath))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # using queue for BFS
    queue = Queue()

    visited_node = [] # Visited nodes are stored here
    path = []

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    queue.push((problem.getStartState(),[]))

    while(True):

        if queue.isEmpty():
            return []

        qq,path = queue.pop() # Take position and path
        visited_node.append(qq)
        if problem.isGoalState(qq):
            return path

        succ = problem.getSuccessors(qq)

        if succ:
            for item in succ:
                if item[0] not in visited_node and item[0] not in (state[0] for state in queue.list):
                    newPath = path + [item[1]] # Calculate new path to visit all the paths
                    queue.push((item[0],newPath))  

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def _update_util(Frontier, item, priority):
        for index, (p, c, i) in enumerate(Frontier.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del Frontier.heap[index]
                Frontier.heap.append((priority, c, item))
                heapq.heapify(Frontier.heap)
                break
        else:
            Frontier.push(item, priority)

    Frontier = util.PriorityQueue()
    Visited_node = []
    Frontier.push( (problem.getStartState(), []), 0 )
    Visited_node.append( problem.getStartState() )

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()

        if problem.isGoalState(state):
            return actions

        if state not in Visited_node:
            Visited_node.append( state )

        for next in problem.getSuccessors(state):
            state_next = next[0]
            direction = next[1]
            if state_next not in Visited_node:
                _update_util( Frontier, (state_next, actions + [direction]), problem.getCostOfActions(actions+[direction]) )
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def _update_util(Frontier, item, priority):
        for index, (p, c, i) in enumerate(Frontier.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del Frontier.heap[index]
                Frontier.heap.append((priority, c, item))
                heapq.heapify(Frontier.heap)
                break
        else:
            Frontier.push(item, priority)

    Frontier = util.PriorityQueue()
    Visited_node = []
    Frontier.push( (problem.getStartState(), []), heuristic(problem.getStartState(), problem) )
    Visited_node.append( problem.getStartState() )

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        #print state
        if problem.isGoalState(state):
            #print 'Find Goal'
            return actions

        if state not in Visited_node:
            Visited_node.append( state )

        for i in problem.getSuccessors(state):
            state_next = i[0]
            direction = i[1]
            if state_next not in Visited_node:
                _update_util( Frontier, (state_next, actions + [direction]), \
                    problem.getCostOfActions(actions+[direction])+heuristic(state_next, problem) )
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
