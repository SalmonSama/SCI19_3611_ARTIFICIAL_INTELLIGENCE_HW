# minimax.py

from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState

def evaluation_function(state: GameState):
    """
    A simple evaluation function that returns the current score of the game state.
    A higher score is better for Pacman.
    """
    return state.getScore()

class PacmanAgent(Agent):
    """
    A Pacman Agent that uses the depth-limited Minimax algorithm.
    """

    def __init__(self, depth='2'):
        """
        Initializes the agent.
        - depth: The number of plies (one move by Pacman and one by each ghost)
                 the algorithm will search ahead.
        """
        super().__init__()
        self.depth = int(depth)

    def get_action(self, state: GameState):
        """
        The main function that decides the best action to take.
        It calls the minimax algorithm to find the action that maximizes Pacman's utility.
        """
        # Find the action that yields the highest minimax value.
        best_action = None
        max_value = -float('inf')

        # Iterate through all legal actions for Pacman to find the best one.
        for action in state.getLegalActions(0):  # 0 is the agent index for Pacman
            successor_state = state.generateSuccessor(0, action)
            # Start the search from the ghost's turn (agent index 1) at depth 0.
            value = self.minimax_value(successor_state, 0, 1)

            if value > max_value:
                max_value = value
                best_action = action
        
        return best_action

    def minimax_value(self, state: GameState, current_depth, agent_index):
        """
        The main recursive minimax function.
        It alternates between max_value (Pacman's turn) and min_value (ghost's turn).
        """
        # --- Terminal State Check (Base Case) ---
        # 1. The maximum search depth is reached.
        # 2. The game has been won or lost.
        if current_depth == self.depth or state.isWin() or state.isLose():
            return evaluation_function(state)

        num_agents = state.getNumAgents()
        
        # If it's Pacman's turn (agent_index 0), find the max value.
        if agent_index == 0:
            return self.max_value(state, current_depth, agent_index)
        # If it's a ghost's turn (agent_index > 0), find the min value.
        else:
            return self.min_value(state, current_depth, agent_index)

    def max_value(self, state: GameState, current_depth, agent_index):
        """
        Calculates the best value for the maximizer (Pacman).
        """
        v = -float('inf')
        for action in state.getLegalActions(agent_index):
            successor_state = state.generateSuccessor(agent_index, action)
            # The next agent is the ghost (agent_index + 1).
            v = max(v, self.minimax_value(successor_state, current_depth, agent_index + 1))
        return v

    def min_value(self, state: GameState, current_depth, agent_index):
        """
        Calculates the best value for the minimizer (ghosts).
        """
        v = float('inf')
        num_agents = state.getNumAgents()
        
        # Determine the next agent and depth.
        # If this is the last ghost, the next turn is Pacman's, and the depth increases.
        next_agent_index = (agent_index + 1) % num_agents
        next_depth = current_depth
        if next_agent_index == 0:
            next_depth += 1

        for action in state.getLegalActions(agent_index):
            successor_state = state.generateSuccessor(agent_index, action)
            v = min(v, self.minimax_value(successor_state, next_depth, next_agent_index))
        return v