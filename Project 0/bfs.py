from pacman_module.game import Agent
from pacman_module.pacman import Directions
from collections import deque  # Import deque

def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.
    """
    return (state.getPacmanPosition(), state.getFood(), tuple(state.getCapsules()))

class PacmanAgent(Agent):
    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        if not self.moves:
            self.moves = self.bfs(state)

        try:
            return self.moves.pop(0)

        except IndexError:
            return Directions.STOP

    def bfs(self, state):
        fringe = deque([(state, [])])  # Use deque and an empty path list
        closed = set()

        while fringe:  # Loop until fringe is empty
            current_state, path = fringe.popleft() # Use popleft() to get the first element (FIFO)

            if current_state.isWin():
                return path

            current_key = key(current_state)

            if current_key not in closed:
                closed.add(current_key)

                for next_state, action in current_state.generatePacmanSuccessors():
                    # Add new states to the right end of the deque
                    new_path = path + [action]
                    fringe.append((next_state, new_path))
        
        return []  # Return empty list if no solution is found