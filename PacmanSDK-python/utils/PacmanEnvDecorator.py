from core.GymEnvironment import *
from utils.ghostact_int2list import *

class PacmanEnvDecorator:
    def __init__(self, env:PacmanEnv=None):
        if env:
            self.env=env
        else:
            self.env = PacmanEnv("local")

    def reset(self, mode="local"):
        self.env.reset(mode=mode)

    def restore(self, state:GameState):
        self.env.ai_reset(state.gamestate_to_statedict())

    def step(self, pacmanAction, ghostAction, state=None):
        if state:
            self.restore(state)
        return self.env.step(pacmanAction, ghostact_int2list(ghostAction))
    
    def game_state(self):
        return self.env.game_state()
    
    def is_eaten(self) -> bool:
        state=self.env.game_state()
        pos_pacman=state.pacman_pos
        pos_ghosts=state.ghosts_pos

        EATEN = False
        for _ in range(3):
            if pos_pacman == pos_ghosts[_]:
                EATEN = True
                break
        
        if EATEN:
            if state.pacman_skill_status[3]>0:
                EATEN = False

        return False
    
    def is_gone(self):
        state=self.env.game_state()
        pos_pacman=state.pacman_pos
        pos_portal=state.portal_coord
        if pos_portal == pos_pacman and state.portal_available:
            return True
        return False