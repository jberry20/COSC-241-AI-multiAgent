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

class Node:
    def __init__(self, state=None, pathcost=None, parent=None, action=None):
        self.state = state
        self.pathcost = pathcost
        self.action = action
        self.parent = parent
    def getParent(self):
        return self.parent
    def getState(self):
        return self.state
    def getPathCost(self):
        return self.pathcost
    def getAction(self):
        return self.action

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        currentFood = currentGameState.getFood()
        currentCapsules = currentGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]        

        x, y = newPos        
        MAX, MIN = 100, -100
        scores = []        
        for ghost in newGhostStates:
            new_distance = manhattanDistance(ghost.getPosition(), newPos)
            score = 1
            if ghost.scaredTimer is not 0 and ghost.scaredTimer >= new_distance:
                #attack ghost
                score = MAX - new_distance
            elif ghost.scaredTimer is 0 and new_distance < 2:
                #run from ghost
                scores.append(MIN + new_distance)
                continue
            if newPos in currentCapsules:
                #eat capsule
                score = MAX + 1
            elif currentFood[x][y] == True:
                #eat food
                score = MAX
            else:
               if (currentCapsules):
                   capsuleDistance = min([manhattanDistance(newPos, x) for x in currentCapsules])
                   score = (max(MAX - breadthFirstSearch(successorGameState), MAX - capsuleDistance)) 
               else:
                   score = (MAX - breadthFirstSearch(successorGameState))
            scores.append(score)
        return scores

def breadthFirstSearch(gameState):

    frontier = util.Queue()
    explored = set()
    steps = 0    
    frontier.push(gameState)
    explored.add(gameState.getPacmanPosition())
    
    while(True):
        if frontier.isEmpty():
            return steps
        
        successorGameState = frontier.pop()
        if (successorGameState.isWin()):
            return steps
        elif (successorGameState.isLose()):
            continue
        
        x, y = successorGameState.getPacmanPosition()
        if (gameState.hasFood(x,y)):
            return steps
        
        steps += 1                                    
        for action in successorGameState.getLegalActions():
            successor = successorGameState.generatePacmanSuccessor(action)
            successorPosition = successor.getPacmanPosition()           
            if successorPosition not in explored:
                frontier.push(successor)
            explored.add(successorPosition)
                
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
        """
        maxLevel, minLevel = 0, 0
        value, action = self.maxValue(gameState, maxLevel, minLevel)
        return action

    def maxValue(self, gameState, maxLevel, minLevel):

        if gameState.isWin() or gameState.isLose() or maxLevel == self.depth:
            return self.evaluationFunction(gameState)
        
        v = -9999.0, None
        for action in gameState.getLegalActions(self.index):
            tmp_v, tmp_a = v
            successorGameState = gameState.generateSuccessor(self.index, action)
            agent = 0
            value = self.minValue(successorGameState, maxLevel + 1, minLevel, agent + 1)           
            if value > tmp_v:
                v = value, action
        return v
                    
    def minValue(self, gameState, maxLevel, minLevel, agent):
                
        numberOfAgents = gameState.getNumAgents()
        v = 9999.0
        if minLevel == (numberOfAgents - 1) * self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)            
            if (numberOfAgents - 1) != agent:        
                value = self.minValue(successorGameState, maxLevel, minLevel + 1, agent + 1)
            else:
                value = self.maxValue(successorGameState, maxLevel, minLevel)
                
            if type(value) is tuple:                    
                tmp_v, tmp_a = value                     
            else:                    
                tmp_v = value                
            v = min(v, tmp_v)            
        return v
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """        
        maxLevel, minLevel = 0, 0
        alpha, beta = -9999, 9999
        value, action = self.maxValue(gameState, maxLevel, minLevel, alpha, beta)
        return action
    
    def maxValue(self, gameState, maxLevel, minLevel, alpha, beta):


        if gameState.isWin() or gameState.isLose() or maxLevel == self.depth:
            return self.evaluationFunction(gameState)

        v = -9999.0, None
        for action in gameState.getLegalActions(self.index):           
            tmp_v, tmp_a = v
            successorGameState = gameState.generateSuccessor(self.index, action)
            agent = 0
            value = self.minValue(successorGameState, maxLevel + 1, minLevel, agent + 1, alpha, beta)
            if value > tmp_v:
                v = value, action            
            tmp_v, tmp_a = v
            if tmp_v > beta:
                return v
            alpha = max(alpha, tmp_v)                
        return v
            
        
    def minValue(self, gameState, maxLevel, minLevel, agent, alpha, beta):
        
        
        numberOfAgents = gameState.getNumAgents()
        v = 9999

        if minLevel == (numberOfAgents - 1) * self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)
 
            if (numberOfAgents - 1) != agent:        
                value = self.minValue(successorGameState, maxLevel, minLevel + 1, agent + 1, alpha, beta)
            else:
                value = self.maxValue(successorGameState, maxLevel, minLevel, alpha, beta)
                
            if type(value) is tuple:                    
                tmp_v, tmp_a = value                     
            else:                    
                tmp_v = value 
            v = min(v, tmp_v)

            if v < alpha:
                return v            
            beta = min(beta, v)                
        return v


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
        maxLevel, minLevel = 0, 0
        value, action = self.maxValue(gameState, maxLevel, minLevel)
        if action is None:
            print(action)
        return action

    def maxValue(self, gameState, maxLevel, minLevel):

        if gameState.isWin() or gameState.isLose() or maxLevel == self.depth:
            return self.evaluationFunction(gameState)

        v = -9999, None
        for action in gameState.getLegalActions(self.index):      
            tmp_v, tmp_a = v
            successorGameState = gameState.generateSuccessor(self.index, action)
            agent = 0
            value = self.minValue(successorGameState, maxLevel + 1, minLevel, agent + 1)
            
            if value > tmp_v:
                v = value, action
        return v
            
        
    def minValue(self, gameState, maxLevel, minLevel, agent):
                
        numberOfGhostAgents = gameState.getNumAgents() - 1
        if minLevel == (numberOfGhostAgents * self.depth) or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        v = 0.0
        number_of_actions = 0.0
        for action in gameState.getLegalActions(agent):
            number_of_actions += 1.0
            successorGameState = gameState.generateSuccessor(agent, action)
            if numberOfGhostAgents != agent:
                value = self.minValue(successorGameState, maxLevel, minLevel + 1, agent + 1)
            else:
                value = self.maxValue(successorGameState, maxLevel, minLevel)

            if type(value) is tuple:                    
                tmp_v, tmp_a = value             
            elif type(value) is str:
                print("terminal", value)                        
            else:
                tmp_v = value
            v += tmp_v                            
            
        return (v / number_of_actions)

def averageFoodDistance(currentGameState):
    
    currentFood = currentGameState.getFood()
    currentPosition = currentGameState.getPacmanPosition()
    foodDistance = []
    for x, row in enumerate(currentFood):
        for y, column in enumerate(currentFood[x]):
            if currentFood[x][y]:
                foodDistance.append(manhattanDistance(currentPosition, (x,y)))
    avg = sum(foodDistance)/float(len(foodDistance)) if (foodDistance and sum(foodDistance) != 0) else 1
    return avg

def minGhostDistance(currentGameState):
    
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    pacmanPosition = currentGameState.getPacmanPosition()    
    minGhostDistance = 9999
    for ghost in currentGhostStates:
        ghostDistance = manhattanDistance(pacmanPosition, ghost.getPosition())
        if ghostDistance < minGhostDistance:
            minGhostDistance = ghostDistance
    if minGhostDistance == 0:
        minGhostDistance = 1
    return minGhostDistance

def minCapsuleDistance(currentGameState):

    pacmanPosition = currentGameState.getPacmanPosition()
    currentCapsules = currentGameState.getCapsules()        
    if (currentCapsules):
        minCapsuleDistance = 9999
        for capsule in currentCapsules:
            capsuleDistance = manhattanDistance(pacmanPosition,capsule)
            if capsuleDistance < minCapsuleDistance:
                minCapsuleDistance = capsuleDistance
        return minCapsuleDistance
    return 0

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Evaluation Function is a function of five features:
            1. Current game score
            2. Average food distance
            3. Ghost distance
            4. Number of food remaining
            5. Distance to closest capsule (if it still exists)
    """

    currentGameScore = scoreEvaluationFunction(currentGameState)
    ghostDistance = minGhostDistance(currentGameState)
    capsuleDistance = minCapsuleDistance(currentGameState)
            
    return (currentGameScore - (averageFoodDistance(currentGameState)) + (2.0/ghostDistance) - currentGameState.getNumFood() - (capsuleDistance))

# Abbreviation
better = betterEvaluationFunction

    

