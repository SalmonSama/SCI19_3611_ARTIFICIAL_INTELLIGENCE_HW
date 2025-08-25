from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import PriorityQueue

def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.
    """
    return (state.getPacmanPosition(), state.getFood(), tuple(state.getCapsules()))

def heuristic(state):
    """
    Returns the estimated cost from the current state to the goal.
    """
    position = state.getPacmanPosition()
    food_grid = state.getFood()
    
    if food_grid.count() == 0:
        return 0

    min_distance = float('inf')
    for x in range(food_grid.width):
        for y in range(food_grid.height):
            if food_grid[x][y]:
                distance = abs(position[0] - x) + abs(position[1] - y)
                min_distance = min(min_distance, distance)
    
    return min_distance

class PacmanAgent(Agent):
    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        if not self.moves:
            self.moves = self.astar(state)

        try:
            return self.moves.pop(0)

        except IndexError:
            return Directions.STOP

    def astar(self, state):
        # fringe stores tuples of (priority, item)
        fringe = PriorityQueue()
        
        # We push a tuple of (state, path) as the item.
        fringe.push((state, []), 0)  

        # closed set stores visited states to avoid cycles
        closed = set()

        while not fringe.isEmpty():
            # Pop returns the item with the lowest priority.
            # It seems your PriorityQueue.pop() returns a tuple of (priority, item)
            # so we'll unpack it correctly here.
            priority, current_item = fringe.pop()
            current_state, path = current_item

            current_key = key(current_state)
            if current_key in closed:
                continue
            
            closed.add(current_key)

            if current_state.isWin():
                return path
            
            for next_state, action in current_state.generatePacmanSuccessors():
                g_cost = len(path) + 1
                h_cost = heuristic(next_state)
                f_cost = g_cost + h_cost
                
                new_path = path + [action]
                
                # Push the new state and path with the calculated f_cost as priority
                fringe.push((next_state, new_path), f_cost)
        
        return [] # failure