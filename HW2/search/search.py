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
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    
    # Initialize with (state, hitWalls, actions)
    stack = Stack()
    stack.push((problem.getStartState(), [], 0))  # (state, path, wall_hits)
    visited = set()
    visited.add((problem.getStartState(), 0))
    
    while not stack.isEmpty():
        state, path, wall_hits = stack.pop()
        
        if wall_hits > 2:
            continue
        
        if problem.isGoalState(state) and (2>=wall_hits >= 1):
            return path
        
        for successor, action, cost in problem.getSuccessors(state):
                next_wall_hits = wall_hits + 1 if problem.isWall(successor) else wall_hits
                nextState = (successor, next_wall_hits)
                if (nextState[0], nextState[1]) not in visited:
                    stack.push((nextState[0], path + [action], nextState[1]))
                    visited.add((nextState[0], nextState[1]))
            
            
    return []

    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queue = Queue()
    queue.push((problem.getStartState(), [], 0))
    visited = set()
    visited.add((problem.getStartState(), 0))
    
    while not queue.isEmpty():
        state, path, wall_hits = queue.pop()
        
        if wall_hits > 2:
            continue
        
        if problem.isGoalState(state) and (2>= wall_hits >=1):
            return path

        for successor, action, _ in problem.getSuccessors(state):
            next_wall_hits = wall_hits + 1 if problem.isWall(successor) else wall_hits
            nextState = (successor, next_wall_hits)
            if (nextState[0], nextState[1]) not in visited:
                queue.push((nextState[0], path + [action], nextState[1]))  
                visited.add((nextState[0], nextState[1]))
    return []
    #util.raiseNotDefined()
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    
    # Initialize with (state, hitWalls, actions) and cost=0
    pq = PriorityQueue()
    startState = problem.getStartState()
    pq.push((startState, 0, []), 0)
    
    visited = set()
    visited.add((startState, 0))

    while True:
        if pq.isEmpty():
            return []
            
        currentState, hitWalls, currentActions = pq.pop()
        
        if hitWalls > 2:
            continue
            
        if problem.isGoalState(currentState) and (1 <= hitWalls <= 2):
            return currentActions
        
        for successor, action, stepCost in problem.getSuccessors(currentState):
            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)
            else:
                nextState = (successor, hitWalls)
                
            if (nextState[0], nextState[1]) not in visited:
                newActions = currentActions + [action]
                newCost = problem.getCostOfActions(newActions)
                pq.push((nextState[0], nextState[1], newActions), newCost)
                visited.add((nextState[0], nextState[1]))

    return []
    # from util import PriorityQueue
    
    # # Initialize with (state, hitWalls, actions) and cost=0
    # pq = PriorityQueue()
    # pq.push((problem.getStartState(), 0, []), 0)  # (state, path, cost, wall_hits)
    # visited = set()
    # visited.add((problem.getStartState(), 0))
    
    # while not pq.isEmpty():
    #     state, wall_hits, path = pq.pop()
        
    #     if wall_hits > 2:
    #         continue
        
    #     if problem.isGoalState(state) and (2>= wall_hits >=1):
    #         return path
        
    #     for successor, action, stepCost in problem.getSuccessors(state):
    #         next_wall_hits = wall_hits + 1 if problem.isWall(successor) else wall_hits

    #         nextState = (successor, next_wall_hits)
    #         if (nextState[0], nextState[1]) not in visited:
    #             newActions = path + [action]
    #             newCost = problem.getCostOfActions(newActions)
    #             # pq.push((nextState[0], nextState[1], newActions), newCost)
    #             # visited.add((nextState[0], nextState[1]))  
    #             # if (nextState[0], nextState[1]) not in visited or newCost < visited(nextState[0]):
    #             pq.push((nextState[0], nextState[1], path + [action]), newCost)
    #             visited.add((nextState[0], newCost))        
    # return []
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    
    # Initialize with (state, hitWalls, actions) and f(n)=h(n)
    # pq = PriorityQueue()
    # pq.push((problem.getStartState(), 0, []), 0)  # (state, path, cost, wall_hits)
    # visited = set()
    # visited.add((problem.getStartState(), 0))
    
    # while not pq.isEmpty():
    #     state, wall_hits, path = pq.pop()
        
    #     if wall_hits > 2:
    #         continue
        
    #     if problem.isGoalState(state) and (2>= wall_hits >=1):
    #         return path
        
    #     for successor, action, stepCost in problem.getSuccessors(state):
    #         next_wall_hits = wall_hits + 1 if problem.isWall(successor) else wall_hits
    #         nextState = (successor, next_wall_hits)
    #         newActions = path + [action]
    #         new_cost = problem.getCostOfActions(newActions)
    #         priority = new_cost + heuristic(successor, problem)
    #         if (nextState[0], nextState[1]) not in visited: # or priority < visited(nextState, 0):
    #             pq.push((nextState[0], path + [action], nextState[1]), priority)
    #             visited.add(nextState[0], priority)            
            
    # return []
    pq = PriorityQueue()
    startState = problem.getStartState()
    pq.push((startState, 0, []), heuristic(startState, problem))
    
    visited = set()
    visited.add((startState, 0))

    while True:
        if pq.isEmpty():
            return []
            
        currentState, hitWalls, currentActions = pq.pop()
        
        if hitWalls > 2:
            continue
            
        if problem.isGoalState(currentState) and (1 <= hitWalls <= 2):
            return currentActions
            
        for successor, action, stepCost in problem.getSuccessors(currentState):
            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)
            else:
                nextState = (successor, hitWalls)
                
            if (nextState[0], nextState[1]) not in visited:
                newActions = currentActions + [action]
                g_n = problem.getCostOfActions(newActions)
                h_n = heuristic(nextState[0], problem)
                f_n = g_n + h_n
                pq.push((nextState[0], nextState[1], newActions), f_n)
                visited.add((nextState[0], nextState[1]))

    return []
    # util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
