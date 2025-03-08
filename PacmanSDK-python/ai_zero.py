from core.gamedata import GameState
from core.GymEnvironment import PacmanEnv
from model_zero import *
from utils.state_dict_to_tensor import *
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pacman_zero:
    def __init__(self):
        self.env=PacmanEnvDecorator(PacmanEnv())
        self.env.reset()
        self.c_puct=2.5
        self.search_time=4
        self.pacman=PacmanAgent(series='02281615')
        self.ghost=GhostAgent(series='02281615')

    def __call__(self, state):
        self.env.restore(state)
        mcts=MCTS(self.env, self.pacman, self.ghost, self.c_puct, num_simulations=self.search_time)
        return mcts.play_game_pacman()

t=time.time()
ai_func=Pacman_zero()
t=time.time()-t
print(t)
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