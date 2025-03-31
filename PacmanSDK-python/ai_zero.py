from core.gamedata import GameState
from core.GymEnvironment import PacmanEnv
from utils.PacmanEnvDecorator import *
from utils.state_dict_to_tensor import *

from model import *
from mcts import *
from ai_ghost import GhostAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pacman_zero:
    def __init__(self):
        self.env=PacmanEnvDecorator(PacmanEnv())
        self.env.reset()
        self.pacman=PacmanAgent(load_series='03290333')
        self.ghost=GhostAI() # GhostAgent()

    def __call__(self, state):
        self.env.restore(state=state)
        mcts = MCTS_pacman(self.env, pacman=self.pacman, ghost=self.ghost, c_puct=1.5, n_search=3)
        action, _, _ = mcts.run()
        return [action]
    
class Ghost_zero:
    def __init__(self):
        self.env=PacmanEnvDecorator(PacmanEnv())
        self.env.reset()
        self.pacman=PacmanAgent(load_series='03280537')
        self.ghost=GhostAgent()

    def __call__(self, state):
        self.env.restore(state=state)
        mcts = MCTS_ghost(self.env, ghost=self.ghost, pacman=self.ghost, c_puct=2.5, n_search=10)
        action, _, _ = mcts.run()
        return ghostact_int2list(action)

ai_func=Pacman_zero()
__all__ = ["ai_func"]

if __name__=='__main__':
    env=PacmanEnv()
    env.reset()
    state=env.game_state()
    
    import time
    t=time.time()
    out=ai_func(state)
    t=time.time()-t
    print(out, t)