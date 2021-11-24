# multiAgents1.py
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
import time
from ghostAgents import RandomGhost
from ghostAgents import DirectionalGhost


from util import manhattanDistance, Counter
from game import Directions
import random, util
import math
import time
from game import Agent
from ghostAgents import DirectionalGhost
from ghostAgents import RandomGhost
from featureExtractors import SimpleExtractor



# class MultiAgentSearchAgent(Agent):
#     """
#       This class provides some common elements to all of your
#       multi-agent searchers.  Any methods defined here will be available
#       to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
#
#       You *do not* need to make any changes here, but you can if you want to
#       add functionality to all your adversarial search agents.  Please do not
#       remove anything, however.
#
#       Note: this is an abstract class: one that should not be instantiated.  It's
#       only partially specified, and designed to be extended.  Agent (game.py)
#       is another abstract class.
#     """
#
#     number_of_nodes = []
#     depth_of_tree = []
#     time_per_moves = []
#
#     def __init__(self, evalFn = 'scoreEvaluationFunction', depth='4'):
#         self.index = 0 # Pacman is always agent index 0
#         self.evaluationFunction = util.lookup(evalFn, globals())
#         self.depth = int(depth)
#


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
        newFood = successorGameState.getFood()
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
    number_of_nodes = []
    depth_of_tree = []
    time_per_moves = []

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class ScoredAction(NamedTuple):
    score: Union[int, float]
    action: Any

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

        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions()
        successors = (gameState.generateSuccessor(0, action) for action in legalActions)
        scores = [self._minimaxScore(successor, 1, self.depth) for successor in successors]
        i = self.getIndexOfMax(scores)
        return legalActions[i]

    def _minimaxScore(self, state, agentIndex: int, depth: int) -> ScoredAction:
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # pacman turn
            selectBestScore = max
            nextAgent = 1
            nextDepth = depth
        else:  # ghost turn
            selectBestScore = min
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = (depth - 1) if nextAgent == 0 else depth

        legalActions = state.getLegalActions(agentIndex)
        successors = (state.generateSuccessor(agentIndex, action) for action in legalActions)
        scores = [self._minimaxScore(successor, nextAgent, nextDepth)
                  for successor in successors]
        return selectBestScore(scores)

    def getIndexOfMax(self, values: Sequence, default=-1):
        return max(range(len(values)), key=values.__getitem__, default=default)


    def getIndexOfMin(self, values: Sequence, default=-1):
        return min(range(len(values)), key=values.__getitem__, default=default)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class Node():
    """
     This class provides a data structure to store nodes in our search tree.
    """

    node_id = 0  # Unique node ID for debugging

    def __init__(self, state, action, parent, agent_index=0):
        """
         action: the action taken to arrive at this state
         state: problem-specific state representation
         parent: parent Node
         agent_index: agent modeled by this mode (Pacman only, currently)
        """

        self.state = state
        self.action = action
        self.parent = parent
        self.agent_index = agent_index
        self.times_explored = 0  # Number of times this node appears in a simulation
        self.num_wins = 0  # Number of simulation wins that include this node
        self.score_sum = 0  # Sum of scores over all simulations involving this node
        self.children = []  # Children expanded in the search tree

        self.node_id = Node.node_id  # Unique node ID assigned to this node for debugging
        Node.node_id += 1

    def best_score_selection(self):
        """Returns child with the best average score over simulations"""
        # Should we consider wins?
        scores = [1.0 * child.score_sum / child.times_explored if child.times_explored else 0.0 for child in
                  self.children]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return self.children[chosenIndex]

    def best_win_potential_selection(self):
        """Returns child with the best average wins."""
        scores = [1.0 * child.num_wins / child.times_explored if child.times_explored else 0.0 for child in
                  self.children]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return self.children[chosenIndex]

    def best_win_and_score_selection(self):
        """Returns the child with highest metric that balanced score and survivability."""
        # TODO: adjust this function so it works better. The agent still chooses to run into Pacman when it doesn't need to.
        best_score = -float('inf')
        bestIndices = []
        for current_child in self.children:
            if current_child.times_explored:
                average_win = current_child.num_wins / current_child.times_explored
                if average_win > 0.1:
                    current_score = (average_win * current_child.score_sum) / current_child.times_explored
                else:
                    current_score = average_win
            else:
                current_score = -float('inf')

            if current_score > best_score:
                bestIndices = [current_child]
                best_score = current_score
            elif current_score == best_score:
                bestIndices.append(current_child)
        return random.choice(bestIndices)  # Pick randomly among the best

    def explore_exploit_selection(self, explore_algorithm='ucb', explore_variable=''):
        if explore_algorithm == 'ucb':
            if explore_variable == '':
                return self.upper_confidence_bound()
            else:
                return self.upper_confidence_bound(float(explore_variable))
        else:
            if explore_variable == '':
                return self.epsilon_greed_search()
            else:
                return self.epsilon_greed_search(float(explore_variable))

    def epsilon_greed_search(self, exploit_weight=0.8):
        """Weights random exploration vs. exploitation"""
        if random.random() < exploit_weight:
            # "EXPLOIT"
            scores = [1.0 * child.score_sum / child.times_explored if child.times_explored else 0.0 for child in
                      self.children]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        else:
            # print "EXPLORE"
            chosenIndex = random.choice(range(len(self.children)))
        return self.children[chosenIndex]

    def upper_confidence_bound(self, c=150.0):
        """Returns the child with the highest upper confidence bound score."""

        scores = [(1.0 * child.score_sum / child.times_explored) + (
                    c * (math.log(self.times_explored) / child.times_explored)) if child.times_explored else float(
            'inf') for child in self.children]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return self.children[chosenIndex]

    def most_visited_selection(self):
        """Returns child that has been visited the most"""
        scores = [child.times_explored for child in self.children]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return self.children[chosenIndex]

    def get_action(self, best_child_algorithm='best_combination'):
        """After simulations, choose the best action by following the path along the nodes with the best average score over simulationss"""
        if best_child_algorithm == 'best_win':
            best_child = self.best_win_potential_selection()
        elif best_child_algorithm == 'best_combination':
            best_child = self.best_win_and_score_selection()
        elif best_child_algorithm == 'most_visited':
            best_child = self.most_visited_selection()
        else:
            best_child = self.best_score_selection()
        return best_child.action

    def gen_children(self):
        """Generate all possible child nodes in the game tree for the given agent_index"""
        children = []
        legalMoves = self.state.getLegalActions(self.agent_index)
        for i in range(len(legalMoves)):
            action = legalMoves[i]
            child_state = self.state.generateSuccessor(self.agent_index, action)
            new_child = Node(child_state, action, parent=self,
                             agent_index=(self.agent_index + 1) % self.state.getNumAgents())
            children.append(new_child)
        self.children = children

    def print_tree(self, tab=0):
        """Helper function for debugging"""
        print(" " * tab + "ID", self.node_id)
        if self.parent:
            print(" " * tab + "Parent", self.parent.node_id)
        else:
            print(" " * tab + "ROOT")
        print(" " * tab + "Agent index", self.agent_index)
        print(" " * tab + "Wins", self.num_wins)
        print(" " * tab + "Score", self.score_sum)
        print(" " * tab + "Explored", self.times_explored)
        for child in self.children:
            child.print_tree(tab + 2)

    def update_score(self, win, score):
        self.times_explored += 1
        if self.agent_index == 1:

            self.num_wins += float(win)

            self.score_sum += score
        else:
            self.num_wins -= float(not win)
            self.score_sum -= score


class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):
    """
      Monte Carlo Tree Search agent from R&N Chapter 5
    """

    current_tree = None
    current_number_of_nodes = 0

    def __init__(self, steps='500', reuse='True', simDepth='10', chooseAlg='best_combination', exploreAlg='ucb',
                 exploreVar='',
                 randSim='False', pacmanEps='0.9', earlyStop='True', tillBored='80', optimism='0.2', panics='True',
                 simRelevance='0.1', dangerZone='0.2'):
        # TODO: Add to command line options

        self.steps_allowed = int(steps)  # Number of iterations of MCTS to do per timestep
        self.reuse_tree = reuse == 'True'  # Whether to reuse the tree created last time this class was called
        self.simulation_depth = int(
            simDepth)  # Depth to play out a simulation before using a heuristic to approximate the score
        self.action_exploration = exploreAlg  # Chosen algorithm to pick the best next action to explore.
        self.explore_algorithm_variable = exploreVar  # Parameter value used to balance exploration and exploitation.
        self.random_simulation_moves = randSim == 'True'  # Whether Pacman's moves in the simulation will be random.
        self.epsilon_pacman_simluation = float(
            pacmanEps)  # Epsilon value for epsilon greedy search if Pacman's simulation values aren't random.
        self.early_stop = earlyStop == 'True'
        self.steps_till_bored = int(tillBored)
        self.featExtractor = SimpleExtractor()
        self.choose_action_algo = chooseAlg
        self.weights = Counter(
            {'eats-food': 326.615053847113, 'closest-food': -22.920237767606736, 'bias': 0.6124765039597753,
             '#-of-ghosts-1-step-away': -2442.2537145683605})  # weights to use in rollout policy based on RL with features from Project 4
        self.simulation_ghost_epsilon = float(optimism)
        self.panics = panics == 'True'  # Whether the agent will avoid early stopping if the win rate is to low.
        self.last_simulation_weight = float(
            simRelevance)  # What percentage of the combined win rate will the last simulation make up.
        self.panic_percent = float(
            dangerZone)  # Win rates below this percentage will prevent the agent from early stopping.

    def getAction(self, gameState):
        """
          Returns the action chosen by MC Tree Search Agent
          Simulations are implemented with all random moves
        """

        def random_transition(state, agent_index):
            """Return a randomly selected (state, action) tuple for the given agent_index
               Return None if there are no possible moves from this state """
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(agent_index)
            if legalMoves:
                # Choose random action
                chosenAction = random.choice(legalMoves)
                return state.generateSuccessor(agent_index, chosenAction), chosenAction
            else:  # EndState - no more moves
                return None

        def q_learning_policy(state):
            # Learned in Project 4

            legalMoves = state.getLegalActions(0)
            if legalMoves:
                maxScore = self.weights * self.featExtractor.getFeatures(state, legalMoves[0])
                maxMoves = [legalMoves[0]]
                for currentMove in legalMoves[1:]:
                    currentScore = self.weights * self.featExtractor.getFeatures(state, currentMove)
                    if maxScore < currentScore:
                        maxMoves = [currentMove]
                        maxScore = currentScore
                    elif maxScore == currentScore:
                        maxMoves.append(currentMove)
                chosenAction = random.choice(maxMoves)
                return state.generateSuccessor(0, chosenAction), chosenAction
            else:
                return None

        def epsilon_greedy_policy(state, epsilon=0.9, agent_index=0):
            """
              Return action that would result in best score with probability epsilon,
              otherwise, return random action
            """
            legalMoves = state.getLegalActions(agent_index)
            if legalMoves:
                if random.random() < epsilon:
                    # TODO: better policy
                    scores = [state.generateSuccessor(agent_index, a).getScore() for a in legalMoves]
                    # scores = [state_heuristic1(state.generateSuccessor(agent_index, a))[1] for a in legalMoves]
                    bestIndices = [index for index in range(len(scores)) if scores[index] == max(scores)]
                    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
                    chosenAction = legalMoves[chosenIndex]
                    return state.generateSuccessor(agent_index, chosenAction), chosenAction
                else:
                    # Choose random action
                    chosenAction = random.choice(legalMoves)
                    return state.generateSuccessor(agent_index, chosenAction), chosenAction
            else:  # EndState - no more moves
                return None

        def select(tree):
            """Selects a leaf node to expand in the search tree"""
            if not tree.children:  # Leaf
                return tree
            # TODO: Write a better SELECT method
            # best_child = tree.best_score_selection()
            best_child = tree.explore_exploit_selection(self.action_exploration, self.explore_algorithm_variable)
            return select(best_child)

        def expand(leaf):
            """Expands all children of the leaf node"""
            leaf.gen_children()
            self.current_number_of_nodes += len(leaf.children)

        def backpropagate(result, node):
            """Update stats of all nodes traversed in current simulation"""
            win, score = result
            node.update_score(win, score)
            if node.parent is None:
                return
            backpropagate(result, node.parent)

        def state_heuristic(state):
            """Returns the heuristic of the current state."""

            # The current heuristic rewards the agent for being in a not losing state and having a higher score
            # while punishing the agent slightly for being too far away from food.
            Pos = state.getPacmanPosition()
            Food = state.getFood()
            closest_food = float('inf')
            for current_food in Food.asList():
                distance = manhattanDistance(Pos, current_food)
                if closest_food > distance:
                    closest_food = distance
            return 0.5, (0.5 * state.getScore()) + 400 - (0.25 * closest_food)

        def state_heuristic1(state):
            """Returns the heuristic of the current state."""
            caps = state.getCapsules()
            Pos = state.getPacmanPosition()
            Food = state.getFood()
            GhostStates = state.getGhostStates()
            ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
            GhostPositions = [g.getPosition() for g in GhostStates]
            # Get the food
            score = -1 * Food.count()
            score += -1 * len(caps)
            # Stay away from ghosts or try to eat them if they're scared
            ghost_dists = [util.manhattanDistance(Pos, gPos) for gPos in GhostPositions]
            for i in range(len(ghost_dists)):
                if ScaredTimes[i] > ghost_dists[i]:
                    score += 1
                else:
                    if ghost_dists[i] < 2:
                        score += -100

            # Get closer to food
            if Food.count() > 0:
                FoodPos = Food.asList()
                min_food_dist = min([util.manhattanDistance(Pos, fPos) for fPos in FoodPos])
                score += -1 * min_food_dist
            return 0.5, score + state.getScore()

        def learned_heuristic(state):
            # Learned in Project 4
            weights = Counter({'closest-food': -2.9590833461811363, 'bias': 205.60863391209026,
                               '#-of-ghosts-1-step-away': -119.89950003939676, 'eats-food': 270.2008225113668})

            legalMoves = state.getLegalActions(0)
            if legalMoves:
                score = max([weights * self.featExtractor.getFeatures(state, a) for a in legalMoves])
                return 0.5, score
            else:
                return state.isWin(), state.getScore()

        def tree_depth(node, current_depth=0):
            max_depth = current_depth
            if len(node.children) > 0:
                max_depth = tree_depth(node.children[0], current_depth + 1)
                for current_child in range(1, len(node.children)):
                    child_depth = tree_depth(node.children[current_child], current_depth + 1)
                    if child_depth > max_depth:
                        max_depth = child_depth
            return max_depth

        def simulate(node, agent_index=0, random_moves=False, heuristic_fn=state_heuristic):
            """Simulate game until end state starting at a given node and choosing all random actions"""
            if random_moves:
                agent_index = 1
                state = node.state

                for current_turn in range(self.simulation_depth):
                    while agent_index < state.getNumAgents():
                        if state.isWin() or state.isLose():
                            return state.isWin(), state.getScore()
                        state, _ = random_transition(state, agent_index)

                        agent_index += 1
                    agent_index = 0
                return heuristic_fn(state)
            else:
                state = node.state

                if random.random() < self.simulation_ghost_epsilon:
                    ghosts = [RandomGhost(i + 1) for i in range(state.getNumAgents())]
                else:
                    ghosts = [DirectionalGhost(i + 1) for i in range(state.getNumAgents())]
                for current_turn in range(self.simulation_depth):
                    while agent_index < state.getNumAgents():
                        if state.isWin() or state.isLose():  # or count == max_steps:
                            return state.isWin(), state.getScore()
                        if agent_index == 0:
                            state, action = q_learning_policy(state)
                            # state, action = epsilon_greedy_policy(state, agent_index=0)
                        else:
                            ghost = ghosts[agent_index - 1]
                            state = state.generateSuccessor(agent_index, ghost.getAction(state))

                        agent_index += 1
                    agent_index = 0

                return heuristic_fn(state)

        def find_state(current_node, search_target, depth=0):
            found_state = None
            if current_node.agent_index == 0 and depth > 0:
                if search_target == current_node.state:
                    found_state = current_node

            else:
                for current_child in current_node.children:
                    found_state = find_state(current_child, search_target, 1)
                    if found_state is not None:
                        break
            return found_state

        ####################
        # MC tree search   #
        #                  #
        # 1. Select        #
        # 2. Expand        #
        # 3. Simulate      #
        # 4. Backpropagate #
        ####################

        start_time = time.time()

        # Instantiate root node
        if MonteCarloTreeSearchAgent.current_tree is not None and self.reuse_tree:
            tree = find_state(MonteCarloTreeSearchAgent.current_tree, gameState, 0)
        else:
            tree = None

        if tree is None:
            tree = Node(gameState, action=None, parent=None)
        else:
            tree.parent = None
        tree = Node(gameState, action=None, parent=None)

        # Count number of iterations
        bored_counter = 0
        counter = 0

        last_top_action = -1
        current_win_rate = 1

        while counter < self.steps_allowed:
            leaf = select(tree)
            expand(leaf)
            if leaf.children:
                child = random.choice(leaf.children)
                result = simulate(child, child.agent_index + 1)
                backpropagate(result, child)
            else:  # End state
                result = leaf.state.isWin(), leaf.state.getScore()
                backpropagate(result, leaf)
            counter += 1

            if self.early_stop:
                if self.panics:
                    current_win_rate = ((1 - self.last_simulation_weight) * current_win_rate) + (
                                self.last_simulation_weight * result[0])
                    if current_win_rate <= self.panic_percent:
                        bored_counter = -1
                current_top_action = tree.get_action(best_child_algorithm=self.choose_action_algo)

                if current_top_action == last_top_action:
                    bored_counter += 1
                    if bored_counter >= self.steps_till_bored:
                        break
                else:
                    last_top_action = current_top_action
                    bored_counter = 0

        # debugging
        # tree.print_tree()
        Node.node_id = 0

        # Select action from child with best simulation stats

        MonteCarloTreeSearchAgent.current_tree = tree
        action = tree.get_action(best_child_algorithm=self.choose_action_algo)
        end_time = time.time()
        MultiAgentSearchAgent.time_per_moves.append(end_time - start_time)
        MultiAgentSearchAgent.number_of_nodes.append(self.current_number_of_nodes)
        MultiAgentSearchAgent.depth_of_tree.append(tree_depth(tree))
        self.current_number_of_nodes = 0
        return action
