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

import math
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from typing import NamedTuple, Union, Any, Sequence
from collections import namedtuple



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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        #print(f'scores: {scores},{bestScore},{bestIndices},{chosenIndex}')

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        #newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostDistances = []
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                distance = manhattanDistance(newPos, ghostState.configuration.pos)
                ghostDistances.append(distance)

        nearestGhost = 100 if not ghostDistances else min(ghostDistances)

        if nearestGhost == 0:
            return -math.inf

        if successorGameState.getNumFood() == 0:
            return math.inf

        food = currentGameState.getFood()
        if food[newPos[0]][newPos[1]]:
            nearestFood = 0

        else:
            foodDistances = []
            for row in range(food.width):
                for col in range(food.height):
                    if food[row][col] == True:
                        foodDistances.append(manhattanDistance(newPos, (row, col)))

            nearestFood = min(foodDistances)

        # Tuned values: 0.6 and 0.9
        risk = 1 / (nearestGhost - 0.6)
        benefit = 1 / (nearestFood + 0.9)
        score = -risk + benefit
        return score



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

decision = namedtuple('decision',['score','action'])

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        actions = gameState.getLegalActions()
        nextStates = [gameState.generateSuccessor(0, act) for act in actions]
        scores = list(map(lambda x: self.minMaxHelper(x, 1, self.depth), nextStates))
        ind = max(range(len(scores)), key=scores.__getitem__, default=-1)
        return actions[ind]

    def minMaxHelper(self, state, agentIndex, curDepth):

        if curDepth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        bestScoreFun = max if agentIndex == 0 else min
        newAgent = (agentIndex + 1) % state.getNumAgents()
        newDepth = (curDepth - 1) if newAgent == 0 else curDepth

        legalActions = state.getLegalActions(agentIndex)
        nextStates = [state.generateSuccessor(agentIndex, act) for act in legalActions]
        scores = list(map(lambda x: self.minMaxHelper(x, newAgent, newDepth), nextStates))
        return bestScoreFun(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBetaHelper(gameState, 0, self.depth,
                             a=-math.inf, b=math.inf).action

    def alphaBetaHelper(self, state, agentIdx, curDepth, a, b):
        if curDepth == 0 or state.isWin() or state.isLose():
            return decision(self.evaluationFunction(state), action=None)

        return self.findMinMax(state, agentIdx, curDepth, a, b)

    def findMinMax(self, state, agentIdx, depth, a, b):

        logic = "max" if agentIdx == 0 else "min"
        finalScore = -math.inf if logic == "max" else math.inf
        finalAction = None
        nextAgent = (agentIdx + 1) % state.getNumAgents()
        nextDepth = depth if nextAgent > 0 else depth - 1

        for act in state.getLegalActions(agentIdx):
            nextStates = state.generateSuccessor(agentIdx, act)
            curScore = self.alphaBetaHelper(nextStates, nextAgent, nextDepth, a, b).score

            if logic == "max":
                if curScore > b:
                    return decision(curScore, act)
                if curScore > finalScore:
                    finalScore = curScore
                    finalAction = act
                    if curScore > a:
                        a = curScore

            else:
                if curScore < a:
                    return decision(curScore, act)
                if curScore < finalScore:
                    finalScore = curScore
                    finalAction = act
                    if curScore < b:
                        b = curScore

        return decision(finalScore, finalAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        actions = gameState.getLegalActions()
        nextStates = [gameState.generateSuccessor(0, act) for act in actions]
        scores = list(map(lambda x: self.getExpectedScore(x, 1, self.depth), nextStates))
        ind = max(range(len(scores)), key=scores.__getitem__, default=-1)
        return actions[ind]

    # def getIndex(self, gameState, agentInd, depth):
    #
    #     actions = gameState.getLegalActions()
    #     nextStates = []
    #     scores = []
    #     for act in actions:
    #         nextStates.append(gameState.generateSuccessor(agentInd, act))
    #     for sucessor in nextStates:
    #         scores.append(self.getExpectedScore(sucessor, agentInd+1, depth))
    #     return scores

    def getExpectedScore(self, state, agentIdx, depth):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIdx)
        nextAgent = (agentIdx + 1) % state.getNumAgents()
        nextDepth = depth if nextAgent > 0 else depth - 1

        nextStates = [state.generateSuccessor(agentIdx, act) for act in legalActions]
        scores = list(map(lambda x: self.getExpectedScore(x, nextAgent, nextDepth), nextStates))
        if agentIdx == 0:  # pacman turn
            return max(scores)
        else:  # ghost turn
            return sum(scores) / len(legalActions)
