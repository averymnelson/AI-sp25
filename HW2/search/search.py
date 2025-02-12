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
    
    # Initialize the s using a stack
    s = Stack()
    startState = problem.getStartState()
    s.push((startState, [], 0))  # (state, path, wallHits)

    v = set()

    while not s.isEmpty():
        state, path, wallHits = s.pop()

        # Check if goal is reached with 1 or 2 wall hits
        if problem.isGoalState(state) and (1 <= wallHits and wallHits <= 2):
            print(wallHits)
            return path

        # Mark state as v with wallHits
        if (state, wallHits) in v:
            continue
        v.add((state, wallHits))

        # Expand successors
        for nextState, action, cost in problem.getSuccessors(state):
            newWallHits = wallHits

            # If nextState is a wall, count the hit but do NOT change nextState
            if problem.isWall(nextState):  
                if wallHits < 2:
                    newWallHits += 1
                else:
                    continue  # Ignore paths that exceed wall limits

            # Push valid states onto the stack
            s.push((nextState, path + [action], newWallHits))

    return []
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    q = Queue()
    
    startState = problem.getStartState()
    q.push((startState, [], 0))

    v = set()
    v.add(startState)

    while not q.isEmpty():
        state, actions, wallHits = q.pop()
        
        if wallHits > 2:
            continue
            
        if problem.isGoalState(state) and (wallHits >= 1 and wallHits <=2):
            print(wallHits)
            return actions
        
        successors = problem.getSuccessors(state)
        for nextState, action, cost in successors:
            currentWallHits = wallHits
            if problem.isWall(nextState):
                currentWallHits = wallHits + 1

            if nextState not in v:
                q.push((nextState, actions + [action], currentWallHits))
                v.add(nextState)
                
    return []
    #util.raiseNotDefined()
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    
    pq = PriorityQueue()
    startState = problem.getStartState()
    pq.push((startState, [], 0), 0)  # (state, path, wallHits), priority = cost

    visited = set()

    while not pq.isEmpty():
        state, path, wallHits = pq.pop()

        # Goal check with valid wall hits
        if problem.isGoalState(state) and 1 <= wallHits <= 2:
            return path

        if (state, wallHits) in visited:
            continue
        visited.add((state, wallHits))

        for nextState, action, stepCost in problem.getSuccessors(state):
            newWallHits = wallHits

            if problem.isWall(nextState):  
                if wallHits < 2:
                    newWallHits += 1
                else:
                    continue  # Ignore invalid paths

            newPath = path + [action]
            newCost = problem.getCostOfActions(newPath)

            pq.push((nextState, newPath, newWallHits), newCost)

    return []
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
    
    pq = PriorityQueue()
    startState = problem.getStartState()
    pq.push((startState, [], 0), heuristic(startState, problem))

    visited = set()

    while not pq.isEmpty():
        state, path, wallHits = pq.pop()

        if problem.isGoalState(state) and 1 <= wallHits <= 2:
            return path

        if (state, wallHits) in visited:
            continue
        visited.add((state, wallHits))

        for nextState, action, stepCost in problem.getSuccessors(state):
            newWallHits = wallHits

            if problem.isWall(nextState):  
                if wallHits < 2:
                    newWallHits += 1
                else:
                    continue  

            newPath = path + [action]
            g_n = problem.getCostOfActions(newPath)  # Cost to get here
            h_n = heuristic(nextState, problem)  # Estimated cost to goal
            f_n = g_n + h_n  # Total estimated cost

            pq.push((nextState, newPath, newWallHits), f_n)

    return []
    # util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
