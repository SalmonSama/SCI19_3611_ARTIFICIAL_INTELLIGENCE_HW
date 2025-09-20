# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions
import numpy as np
from pacman_module.util import manhattanDistance

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.
        """
        # XXX: Your code here to obtain bonus
        
        legal_moves = state.getLegalPacmanActions()
        if Directions.STOP in legal_moves:
            legal_moves.remove(Directions.STOP)

        if not legal_moves:
            return Directions.STOP

        best_move = Directions.STOP
        max_score = -float('inf')

        # หาตำแหน่งที่น่าจะเป็นที่สุดของผีแต่ละตัว
        ghost_positions = []
        for belief in belief_state:
            if np.sum(belief) > 0:
                # unravel_index ใช้เปลี่ยน flat index ไปเป็น coordinate
                max_prob_pos = np.unravel_index(np.argmax(belief, axis=None), belief.shape)
                ghost_positions.append(max_prob_pos)
        
        # ถ้าไม่มีผีเหลืออยู่ ก็ให้เดินหาอาหารแทน
        if not ghost_positions:
             # (อาจจะเพิ่มโค้ดหาอาหารที่นี่)
             return random.choice(legal_moves)

        pacman_pos = state.getPacmanPosition()

        # ประเมินคะแนนของแต่ละ action ที่ทำได้
        for move in legal_moves:
            successor_state = state.generatePacmanSuccessor(move)
            if successor_state is None:
                continue

            successor_pos = successor_state.getPacmanPosition()
            score = 0
            
            # คำนวณระยะห่างไปยังผีที่ใกล้ที่สุด
            min_dist_to_ghost = float('inf')
            if ghost_positions:
                distances = [manhattanDistance(successor_pos, g_pos) for g_pos in ghost_positions]
                min_dist_to_ghost = min(distances)

            # ให้คะแนนสูงถ้าเข้าใกล้ผี
            # ยิ่งใกล้ คะแนนยิ่งสูง (ใช้ 1/distance)
            score -= min_dist_to_ghost 
            
            # เพิ่มคะแนนถ้ากินอาหารได้
            if state.hasFood(successor_pos[0], successor_pos[1]):
                score += 10
                
            # เพิ่มคะแนนถ้ากินแคปซูล
            if successor_pos in state.getCapsules():
                score += 50

            if score > max_score:
                max_score = score
                best_move = move

        # XXX: End of your code here to obtain bonus
        return best_move