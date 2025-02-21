from core.gamedata import GameState
from core.GymEnvironment import PacmanEnv
from model import *
from train import state_dict_to_tensor
import sys  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PacmanAI:
    def __init__(self, device=device):
        print(device, file=sys.stderr)
        print("ai init start", file=sys.stderr)
        self.device = device
        self.pacman_net = PacmanNet(4, 5, 40)
        self.pacman_net.load_state_dict(torch.load("pacman.pth",map_location=device))
        self.pacman_net.to(self.device)
        self.pacman_net.eval()
        print("ai init done1", file=sys.stderr)

    def __call__(self, game_state: GameState):
        state = game_state.gamestate_to_statedict()
        state_tensor, extra = state_dict_to_tensor(state)
        print("ai init done2", file=sys.stderr)
        print(game_state.level,game_state.round,game_state.board_size,game_state.board.shape,file=sys.stderr)
        with torch.no_grad():
            op = (self.pacman_net(state_tensor.to(self.device), extra.to(self.device)).argmax(1).cpu())
            print(op,file=sys.stderr)
        return [op.item()]

ai_func = PacmanAI().__call__
__all__ = ["ai_func"]

if __name__ == "__main__":
    ai = PacmanAI()
    env = PacmanEnv()
    env.reset()
    state = env.game_state()

    out = ai(state)
    print(out)