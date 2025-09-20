# hminimax.py

from pacman_module.game import Agent
from pacman_module.pacman import GameState

def score_evaluation_function(current_game_state: GameState):
    return current_game_state.getScore()

class PacmanAgent(Agent):
    """
    Agent ที่ใช้อัลกอริทึม H-Minimax (Minimax with Alpha-Beta Pruning)
    """

    def __init__(self, depth='2', eval_fn=score_evaluation_function):
        super().__init__()
        self.depth = int(depth)
        self.evaluation_function = eval_fn

    def get_action(self, game_state: GameState):
        """
        คืนค่า Action ที่ดีที่สุดโดยใช้ Alpha-Beta Pruning
        """
        alpha = -float('inf')
        beta = float('inf')
        # เริ่มต้นค้นหาจาก Pacman (agent 0)
        _, best_action = self.get_value(game_state, 0, 0, alpha, beta)
        return best_action

    def get_value(self, state: GameState, current_depth, agent_index, alpha, beta):
        """
        ฟังก์ชัน Alpha-Beta แบบ Recursive หลัก
        """
        num_agents = state.getNumAgents()
        
        # Base Case: ถ้าเกมจบ หรือถึงความลึกสูงสุด
        if state.isWin() or state.isLose() or current_depth == self.depth:
            return self.evaluation_function(state), None

        # ถ้าเป็นตาของ Pacman (Maximizer)
        if agent_index == 0:
            return self.max_value(state, current_depth, agent_index, alpha, beta)
        # ถ้าเป็นตาของผี (Minimizer)
        else:
            return self.min_value(state, current_depth, agent_index, alpha, beta)

    def max_value(self, state: GameState, current_depth, agent_index, alpha, beta):
        max_val = -float('inf')
        best_action = None
        
        successors = state.generatePacmanSuccessors()
        if not successors:
            return self.evaluation_function(state), None

        next_agent_index = 1

        for successor_state, action in successors:
            val, _ = self.get_value(successor_state, current_depth, next_agent_index, alpha, beta)
            if val > max_val:
                max_val = val
                best_action = action
            
            if max_val > beta:
                return max_val, best_action # Pruning
            
            alpha = max(alpha, max_val)
            
        return max_val, best_action

    def min_value(self, state: GameState, current_depth, agent_index, alpha, beta):
        min_val = float('inf')
        
        successors = state.generateGhostSuccessors(agent_index)
        if not successors:
            return self.evaluation_function(state), None

        num_agents = state.getNumAgents()
        next_agent_index = (agent_index + 1) % num_agents
        next_depth = current_depth
        if next_agent_index == 0:
            next_depth += 1

        for successor_state, action in successors:
            val, _ = self.get_value(successor_state, next_depth, next_agent_index, alpha, beta)
            min_val = min(min_val, val)

            if min_val < alpha:
                return min_val, None # Pruning
            
            beta = min(beta, min_val)

        return min_val, None