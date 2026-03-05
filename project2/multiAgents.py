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

import sys

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        def has_wall(position):
            walls = successorGameState.getWalls()
            return walls[position[0]][position[1]]

        north = (newPos[0], newPos[1] + 1)
        south = (newPos[0], newPos[1] - 1)
        east  = (newPos[0] + 1, newPos[1])
        west  = (newPos[0] - 1, newPos[1])

        score = successorGameState.getScore()
        # Avoid corners (adjacent squares have 2 walls)
        if (has_wall(north) and has_wall(east)) or (has_wall(north) and has_wall(west)) or (has_wall(south) and has_wall(west)) or (has_wall(south) and has_wall(east)):
            score -= 10

        "*** YOUR CODE HERE ***"
        min_distance = sys.float_info.max
        for food_pos in newFood.asList():
            distance = manhattanDistance(newPos, food_pos)
            if distance < min_distance:
                min_distance = distance

        score += 10 / min_distance

        # Penalty for every food left, so go quickly
        score -= 4 * len(newFood.asList())

        for ghost_state in newGhostStates:
            ghost_pos = ghost_state.getPosition()
            distance = manhattanDistance(newPos, ghost_pos)
            if distance < 3 and distance != 0:
                score -= 40 / distance # Heart attack...

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(state: GameState, depth: int, agent_index: int):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            next_depth = depth

            # Depth only decreases after all agents have moved
            if agent_index == state.getNumAgents() - 1:
                next_depth -= 1

            next_agent = (agent_index + 1) % state.getNumAgents()

            if agent_index == 0:
                best = float('-inf')

                for action in state.getLegalActions(agent_index):
                    next_state = state.generateSuccessor(agent_index, action)
                    best = max(minimax(next_state, next_depth, next_agent), best)
                return best
            else:
                best = float('inf')

                for action in state.getLegalActions(agent_index):
                    next_state = state.generateSuccessor(agent_index, action)
                    best = min(minimax(next_state, next_depth, next_agent), best)
                return best

        pacman_actions = gameState.getLegalActions(0)

        best_action = None
        best_score = float('-inf')

        for action in pacman_actions:
            next_state = gameState.generateSuccessor(0, action)
            score = minimax(next_state, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minimax_alpha_beta(state: GameState, depth: int, agent_index: int, alpha: float, beta: float):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            next_depth = depth

            # Depth only decreases after all agents have moved
            if agent_index == state.getNumAgents() - 1:
                next_depth -= 1

            next_agent = (agent_index + 1) % state.getNumAgents()

            if agent_index == 0:
                best = float('-inf')

                for action in state.getLegalActions(agent_index):
                    next_state = state.generateSuccessor(agent_index, action)

                    score = minimax_alpha_beta(next_state, next_depth, next_agent, alpha, beta)
                    best = max(score, best)

                    if score > beta:
                        break
                    alpha = max(best, alpha)
                return best
            else:
                best = float('inf')

                for action in state.getLegalActions(agent_index):
                    next_state = state.generateSuccessor(agent_index, action)

                    score = minimax_alpha_beta(next_state, next_depth, next_agent, alpha, beta)
                    best = min(score, best)

                    if score < alpha:
                        break
                    beta = min(best, beta)
                return best

        pacman_actions = gameState.getLegalActions(0)

        best_action = None
        best_score = float('-inf')

        alpha = float('-inf')

        for action in pacman_actions:
            next_state = gameState.generateSuccessor(0, action)
            score = minimax_alpha_beta(next_state, self.depth, 1, alpha, float('inf'))
            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state: GameState, depth: int, agent_index: int):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            next_depth = depth

            # Depth only decreases after all agents have moved
            if agent_index == state.getNumAgents() - 1:
                next_depth -= 1

            next_agent = (agent_index + 1) % state.getNumAgents()

            if agent_index == 0:
                best = float('-inf')

                for action in state.getLegalActions(agent_index):
                    next_state = state.generateSuccessor(agent_index, action)
                    best = max(expectimax(next_state, next_depth, next_agent), best)
                return best
            else:
                result = 0
                for action in state.getLegalActions(agent_index):
                    next_state = state.generateSuccessor(agent_index, action)
                    result += expectimax(next_state, next_depth, next_agent)
                return result / len(state.getLegalActions(agent_index))

        pacman_actions = gameState.getLegalActions(0)

        best_action = None
        best_score = float('-inf')

        for action in pacman_actions:
            next_state = gameState.generateSuccessor(0, action)
            score = expectimax(next_state, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    result = 0

    def inner_eval(state: GameState):
        # Useful information you can extract from a GameState (pacman.py)
        pacman_pos = state.getPacmanPosition()
        food = state.getFood()
        ghost_states = state.getGhostStates()

        def has_wall(position):
            walls = state.getWalls()
            return walls[position[0]][position[1]]

        north = (pacman_pos[0], pacman_pos[1] + 1)
        south = (pacman_pos[0], pacman_pos[1] - 1)
        east  = (pacman_pos[0] + 1, pacman_pos[1])
        west  = (pacman_pos[0] - 1, pacman_pos[1])

        score = state.getScore()
        # Avoid corners (adjacent squares have 2 walls)
        if (has_wall(north) and has_wall(east)) or (has_wall(north) and has_wall(west)) or (has_wall(south) and has_wall(west)) or (has_wall(south) and has_wall(east)):
            score -= 10

        "*** YOUR CODE HERE ***"
        min_distance = sys.float_info.max
        for food_pos in food.asList():
            distance = manhattanDistance(pacman_pos, food_pos)
            if distance < min_distance:
                min_distance = distance

        score += 10 / min_distance

        # Penalty for every food left, so go quickly
        score -= 4 * len(food.asList())

        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            distance = manhattanDistance(pacman_pos, ghost_pos)
            if distance < 3 and distance != 0 and ghost_state.scaredTimer == 0:
                score -= 10 / distance # Heart attack...

        return score

    def expectimax(state: GameState, depth: int, agent_index: int):
        if depth == 0 or state.isWin() or state.isLose():
            return inner_eval(state)

        next_depth = depth

        # Depth only decreases after all agents have moved
        if agent_index == state.getNumAgents() - 1:
            next_depth -= 1

        next_agent = (agent_index + 1) % state.getNumAgents()

        if agent_index == 0:
            best = float('-inf')

            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                best = max(expectimax(next_state, next_depth, next_agent), best)
            return best
        else:
            result = 0
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                result += expectimax(next_state, next_depth, next_agent)
            return result / len(state.getLegalActions(agent_index))

    result = expectimax(currentGameState, 2, 0)

    return result

# Abbreviation
better = betterEvaluationFunction
