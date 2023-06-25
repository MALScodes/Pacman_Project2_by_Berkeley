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
#         successorGameState = currentGameState.generatePacmanSuccessor(action)
#         newPos = successorGameState.getPacmanPosition()
#         newFood = successorGameState.getFood()
#         newGhostStates = successorGameState.getGhostStates()
#         newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        ghostPositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
######################Initlizng#########################
        pacmanScore = successorGameState.getScore() #-----> initialzing our pacman score 
        posInfinity= float("+inf") #---> we will be using positive inf through out our code
        negInfinity= float("-inf") #---> we will be using negative inf through out our code

        if successorGameState.isWin(): # ---> as soon as we reach the winning state (get all food)
            return posInfinity # ---> we will return posInf.
        
######################Ghosts#########################
 
        nearGhost = posInfinity #----> Initlizng our nearest ghost to positive inf, 
                                       #we will next modify it by looking at all distances of ghosts in our state
        for ghost in ghostPositions: #----> checking all the ghosts position in our state
            newGhost=util.manhattanDistance(newPos, ghost)#----> checking all the pacman->ghosts disistances in our state using
                                                                                                                       #manhattanDistance
            if newGhost<nearGhost:#---> finding the minumem distance of the nearest ghost
                nearGhost=newGhost#----> modifying the nearGhost var.
        for runawayGhosts in newScaredTimes: #----> for go through the dots for scary times 
            runawayGhosts= runawayGhosts + runawayGhosts #---> adding all the scared ghosts to each other so that we could get the best outcome when it comes to eating the ghosts
        if runawayGhosts > 0 : #----> if the ghost is frightend / eaten our pacman will gain and be motivated by points
            pacmanScore = pacmanScore + 250
        pacmanScore = pacmanScore + (2 * nearGhost)

######################Food#########################

        food = posInfinity #---> Initialize food to positive infinity
        for dots in newFood.asList(): #----> Loop through all the food dots in the game
            newClosestDot=util.manhattanDistance(newPos, dots)#---> Calculate the manhattan distance between Pacman's position and the current food dot
            if newClosestDot<food: #----> Update the closest food dot if it's closer than the current closest
                food=newClosestDot
        pacmanScore = pacmanScore - (2 * food) #---> Subtract twice the distance to the closest food dot from Pacman's score
        if(successorGameState.getNumFood() < currentGameState.getNumFood()):#----> If Pacman has eaten a food dot, add 10 points to his score

            pacmanScore = pacmanScore + 10
        

        if action is Directions.STOP: #----> If Pacman stops moving, subtract negative infinity from his score

            pacmanScore = pacmanScore+ negInfinity

        return pacmanScore

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        posInfinity= float("+inf") #---> we will be using positive inf through out our code
        negInfinity= float("-inf") #---> we will be using negative inf through out our code
        
        def MaxME(gameState, depth, agentME):
            availableMeActions=gameState.getLegalActions(agentME)
            if depth == self.depth: #---> Terminal Case 1: when reached to the depth final return our score
                return (gameState.getScore(), None)
            elif (gameState.isWin()==True): #----> Terminal Case 2: when we reach the winning state return our score
                return (gameState.getScore(), None)
            elif (gameState.isLose()==True):#----> Terminal Case 3: when we reach the losing state return our score
                return (gameState.getScore(), None)
            listofnodesMAX=[] #--> initilazing our list if max nodes visted
            agentME=0 #--> making our agent 0 which means we play against pacman
            node = negInfinity #----> v= -inf
            
            for actionME in availableMeActions : #--> loop through all the available action for pacman
                state = gameState.generateSuccessor(agentME, actionME)#---> genertaing Successors for our pacman agent
                min_node = MinGhost(state, depth,1)[0] #---> calling the minghost (next node) with paramters of state, depth, and 1 (ghost agent)
                listofnodesMAX.append(min_node) #--> appending the min node retrieved from MinGhost
                if min_node > node: #---> if min node is bigger than our past node
                    action= actionME#---> imp. our current action 
                    node= min_node#---> our next node is now the min node
            return (node, action) #return the action and node

        def MinGhost(gameState, depth, agentGhost):
            availableGhActions=gameState.getLegalActions(agentGhost)
            if depth == self.depth: #---> Terminal Case 1: when reached to the depth final return our score
                return (gameState.getScore(), None)
            elif (gameState.isWin()==True): #----> Terminal Case 2: when we reach the winning state return our score
                return (gameState.getScore(), None)
            elif (gameState.isLose()==True):#----> Terminal Case 3: when we reach the losing state return our score
                return (gameState.getScore(), None)
            listofnodes=[]#--> initilazing our list if min nodes visted
            node = posInfinity#----> v= +inf
            for actionGhost in availableGhActions:#--> loop through all the available action for ghost
                state = gameState.generateSuccessor(agentGhost, actionGhost)#---> genertaing Successors for our ghost agent
                if (agentGhost+1 is gameState.getNumAgents()): #---> if the next node is our last node(depth-wise)
                    eventualNode = MaxME(state, depth+1,0)[0]#---> calling the Max (next node) with paramters of state, depth, and 1 (ghost agent)
                elif (agentGhost+1 is not gameState.getNumAgents()):#---> if our next node is not the last node (depth-wise)
                    eventualNode = MinGhost(state,depth, agentGhost+1)[0]#---> calling the minghost (next node) with paramters of state, depth, and 1 (ghost agent)
                    listofnodes.append(eventualNode)#---> appending the min node to our list
                if eventualNode < node: #---> if our current node is bigger than our next node
                    action= actionGhost #---> imp. our current action 
                    node= eventualNode #---> our next node is now the min node
                    print(listofnodes)
            return (node, action)
#         return MaxME(gameState, 0,0)[0]
        return MaxME(gameState, 0,0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        posInfinity= float("+inf") #---> we will be using positive inf through out our code
        negInfinity= float("-inf") #---> we will be using negative inf through out our code
        a = negInfinity
        b = posInfinity


        def MaxMEprune(gameState, depth, a, b):
            agentME=0
            availableMeActions=gameState.getLegalActions(agentME)
            if depth == self.depth: #---> Terminal Case 1: when reached to the depth final return our score
                return (gameState.getScore(), None)
            elif (gameState.isWin()==True): #----> Terminal Case 2: when we reach the winning state return our score
                return (gameState.getScore(), None)
            elif (gameState.isLose()==True):#----> Terminal Case 3: when we reach the losing state return our score
                return (gameState.getScore(), None)
            
            node = negInfinity #----> v= -inf
            
            for actionME in availableMeActions :#--> loop through all the available action for pacman
                state = gameState.generateSuccessor(agentME, actionME)#---> genertaing Successors for our pacman agent
                min_node = MinGhostprune(state, depth, 1, a, b)[0]
                if min_node > node: #---> if min node is bigger than our past node
                    actionTaken= actionME#---> imp. our current action 
                    node= min_node#---> our next node is now the min node
                if (b < node):#---> if our beta is less than the current node
                    return (node, actionTaken)#---> we will return the node and it's action
                else:
                    if node > a: #---> else if our node is bigger than alpha
                        a = node #---> our new alpha value will be our current node
                    else:
                        a = a #---> else alpha value doesn't change
            return (node, actionTaken)

        def MinGhostprune(gameState, depth, agentGhost, a, b):
            availableGhActions=gameState.getLegalActions(agentGhost)
            if depth == self.depth: #---> Terminal Case 1: when reached to the depth final return our score
                return (gameState.getScore(), None)
            elif (gameState.isWin()==True): #----> Terminal Case 2: when we reach the winning state return our score
                return (gameState.getScore(), None)
            elif (gameState.isLose()==True):#----> Terminal Case 3: when we reach the losing state return our score
                return (gameState.getScore(), None)
            
            node = posInfinity #----> v= inf
            for actionGhost in availableGhActions: #--> loop through all the available action for ghost
                state = gameState.generateSuccessor(agentGhost, actionGhost) #---> genertaing Successors for our ghost agent
                if (agentGhost+1 is gameState.getNumAgents()): #---> if the next node is our last node(depth-wise)
                    eventualNode = MaxMEprune(state, depth + 1, a, b)[0]#---> call Max node one last time
                elif (agentGhost+1 is not gameState.getNumAgents()): #---> if the next node is not our last node(depth-wise)
                    eventualNode = MinGhostprune(state, depth, agentGhost + 1, a, b)[0]#--->get our ghost node
                if eventualNode < node:# --> if our node is less than the old (max) node 
                    actionTaken= actionGhost#---> we will store the action taken
                    node= eventualNode#----> we will store the current node to node (replace)
                if (a > node): #--->if alphs is bigger than our node
                    print("beta is pruned", node, a)
#                     print("Prune beta and return")
                    return (node, actionTaken) #prune beta and retruen
                else:
                    if node < b: #---> if beta is bigger than our node
                         b = node#---> re-evalutae our beta to be the value of our node
                    else:
                         b = b #----> else, beta doesn't change
#                     print("b",b)
            return (node, actionTaken)

        return MaxMEprune(gameState, 0, a, b)[1]
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
        ##########Initilazing############
        posInfinity= float("+inf") #---> we will be using positive inf through out our code
        negInfinity= float("-inf") #---> we will be using negative inf through out our code
        points = negInfinity
        availableMeActions = gameState.getLegalActions(0)
        MultiAgentSearchAgent = None
        
       ###################################
        def MaxNODE(gameState,depth):
            agentME=0
            availableMeActions=gameState.getLegalActions(agentME)
            dapth = depth + 1#--> incrementing our depth by 1
            if dapth == self.depth: #---> Terminal Case 1: when reached to the depth final return our score
                return (self.evaluationFunction(gameState))
            elif (gameState.isWin()==True): #----> Terminal Case 2: when we reach the winning state return our score
                return (self.evaluationFunction(gameState))
            elif (gameState.isLose()==True):#----> Terminal Case 3: when we reach the losing state return our score
                return (self.evaluationFunction(gameState))
            node = negInfinity
            for actionME in availableMeActions :#--> loop through all the available action for pacman
                state = gameState.generateSuccessor(agentME, actionME)#---> genertaing Successors for our pacman agent
                node = max(node,expectNODE(state,dapth,1))
            return node
        
        def expectNODE(gameState,depth, agentIndex):
            availableMeActions = gameState.getLegalActions(agentIndex)
            agentIndexalpha=agentIndex+1
            if (gameState.isWin()==True): #----> Terminal Case 1: when we reach the winning state return our score
                return (self.evaluationFunction(gameState))
            elif (gameState.isLose()==True):#----> Terminal Case 2: when we reach the losing state return our score
                return (self.evaluationFunction(gameState))
            expVAL = 0 #---> initalizng our expected node value to 0
            actionsAchieved = len(availableMeActions) #---> count all of our actions
            listofExp=[] #---> initialzing our list to follow up with our expected values
            for expActions in availableMeActions:#---> go through all actions availbe to agent(exp)
                statesucc= gameState.generateSuccessor(agentIndex,expActions)#--->generate state successors
                if (agentIndex+1 is (gameState.getNumAgents())):#---> if our next node is the last node (depth-wise)
                    value = MaxNODE(statesucc,depth)#---> calling the Max (next node) with paramters of state, depth
                elif (agentIndex+1 is not (gameState.getNumAgents())): #---> if our next node is not the last node (depth-wise)
                    value = expectNODE(statesucc,depth,agentIndexalpha)
                expVAL = expVAL + value #---> we incrment our Expect value by the value retireved from next node
                listofExp.append(expVAL) #----> appending the expVAL to follow our expected values
                ExpPER=(expVAL)/(actionsAchieved)
            print(ExpPER)
            return ExpPER
        

        for takenAction in availableMeActions:#---> go through all actions availbe to agent(pacman)
            statesucc = gameState.generateSuccessor(0,takenAction)#---->generate state successors
            resaultExecuteExp = expectNODE(statesucc,0,1)#---> start our programe and call expectNODE (first node to be visited)
            if (resaultExecuteExp > points) : #----> update the highest score and the best action so far
                MultiAgentSearchAgent, points = takenAction, resaultExecuteExp#---> update our return statment and points
        return MultiAgentSearchAgent#---> return our best score

    
    
    
    
    
    
    
    
    
    
    

    
