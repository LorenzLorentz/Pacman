import numpy as np
from enum import Enum
from core.gamedata import *

class AIState(Enum):
    COLLECT = "COLLECT"
    ESCAPE = "ESCAPE"
    BONUS = "BONUS"
    GETOUT = "GETOUT"

def is_valid_position(self, pos, game_state: GameState):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < game_state.board_size and 0 <= y < game_state.board_size:
            if self.current_state != AIState.GETOUT:
                if game_state.board[x][y] == Space.PORTAL.value:
                    return False
            if game_state.board[x][y] != Space.WALL.value:
                return True
        return False

def get_valid_moves_pacman(self, pos, game_state):
    moves = []
    directions = [
        (np.array(list(Update.UP.value)), Direction.UP.value),  # UP
        (np.array(list(Update.LEFT.value)), Direction.LEFT.value),  # LEFT
        (np.array(list(Update.DOWN.value)), Direction.DOWN.value),  # DOWN
        (np.array(list(Update.RIGHT.value)), Direction.RIGHT.value),  # RIGHT
    ]
    for direction, move_num in directions:
        new_pos = pos + direction
        if self.is_valid_position(new_pos, game_state):
            moves.append(move_num)
    return moves

def get_valid_moves_ghost(self, pos, game_state):
    moves = []
    directions = [
        (np.array(list(Update.UP.value)), Direction.UP.value),  # UP
        (np.array(list(Update.LEFT.value)), Direction.LEFT.value),  # LEFT
        (np.array(list(Update.DOWN.value)), Direction.DOWN.value),  # DOWN
        (np.array(list(Update.RIGHT.value)), Direction.RIGHT.value),  # RIGHT
    ]
    for direction1, move_num1 in directions:
        for direction2, move_num2 in directions:
            for direction3, move_num3 in directions:
                new_pos = pos[0] + direction1
                if self.is_valid_position(new_pos, game_state):
                    new_pos = pos[1] + direction2
                    if self.is_valid_position(new_pos, game_state):
                        new_pos = pos[2] + direction3
                        if self.is_valid_position(new_pos, game_state):
                            moves.append([move_num1, move_num2, move_num3])
    return moves