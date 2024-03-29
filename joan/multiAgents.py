# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        food = currentGameState.getFood().asList()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total_score = 0
        for position in food:
            d = manhattanDistance(position, newPos)
            total_score += 100 if d == 0 else 1.0 / ( (d ** 2))
            print(total_score)
        for ghost in newGhostStates:
            d = manhattanDistance(ghost.getPosition(), newPos)
            if d > 1:
                continue
            total_score += 2000 if ghost.scaredTimer != 0 else -200
        return total_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def terminal_test(self, state, depth):
        return depth == 0 or state.isWin() or state.isLose()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        actions = []
        for action in gameState.getLegalActions(agentIndex=0):
            succ = gameState.getNextState(agentIndex=0, action=action)
            u = self.min_value(
                succ, agent=1, depth=self.depth
            )
            if u == v:
                actions.append(action)
            elif u > v:
                v = u
                actions = [action]
        # return random.choice(actions)
        return actions[0]
    
    def min_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("inf")
        for action in gameState.getLegalActions(agentIndex=agent):
            succ = gameState.getNextState(agent, action=action)
            if agent == gameState.getNumAgents() - 1:
                v = min(
                    v, self.max_value(succ, agent=0, depth=depth -1)
                )
            else:
                v = min(
                    v, self.min_value(succ, agent=agent +1, depth=depth)
                )
        return v

    def max_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        for action in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=action)
            v = max(
                v, self.min_value(succ, agent=1, depth=depth)
            )
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        pruning = {
            "alpha": float("-inf"),
            "beta": float("inf")
        }
        actions = []
        for action in gameState.getLegalActions(agentIndex=0):
            succ = gameState.getNextState(agentIndex=0, action=action)
            u = self.min_value(
                succ, agent=1, depth=self.depth, pruning=pruning
            )
            """
            We removed this part sa we don't need a list of actions, we want the first one
            as we would have explored a more complete version of the tree
            if u == v:
                actions.append(action)
            
            """
            if u > v:
                v = u
                actions = [action]
            pruning["alpha"] = max(pruning["alpha"], v)
        return actions[0]

    def min_value(self, gameState, agent, depth, pruning):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("inf")
        _pruning = pruning.copy()
        for action in gameState.getLegalActions(agentIndex=agent):
            succ = gameState.getNextState(agent, action=action)
            if agent == gameState.getNumAgents() - 1:
                v = min(
                    v, self.max_value(succ, agent=0, depth=depth -1, pruning=_pruning)
                )
            else:
                v = min(
                    v, self.min_value(succ, agent=agent +1, depth=depth, pruning=_pruning)
                )
            
            if v < pruning["alpha"]:
                return v
            _pruning["beta"] = min(_pruning["beta"], v)

        return v

    def max_value(self, gameState, agent, depth, pruning):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        _pruning=pruning.copy()
        for action in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=action)
            v = max(
                v, self.min_value(succ, agent=1, depth=depth, pruning=_pruning)
            )

            if v > pruning["beta"]:
                return v
            _pruning["alpha"] = max(_pruning["alpha"], v)
        return v
        

class IterativeMinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

class ExpectiMaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        actions = []
        for action in gameState.getLegalActions(agentIndex=0):
            succ = gameState.getNextState(agentIndex=0, action=action)
            u = self.min_value(
                succ, agent=1, depth=self.depth
            )
            if u == v:
                actions.append(action)
            elif u > v:
                v = u
                actions = [action]
        # return random.choice(actions)
        return actions[0]
    
    def min_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = []
        for action in gameState.getLegalActions(agentIndex=agent):
            succ = gameState.getNextState(agent, action=action)
            if agent == gameState.getNumAgents() - 1:
                v += [self.max_value(succ, agent=0, depth=depth -1)]
            else:
                v += [self.min_value(succ, agent=agent +1, depth=depth)]
        
        avg_value = float(sum(v) / len(v))

        return avg_value

    def max_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        for action in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=action)
            v = max(
                v, self.min_value(succ, agent=1, depth=depth)
            )
        return v



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # compute position score

    # compute ghost hunting

    # compute ghost positioning

# Abbreviation
better = betterEvaluationFunction
