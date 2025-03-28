from core.gamedata import GameState
from core.GymEnvironment import PacmanEnv
from utils.PacmanEnvDecorator import *
from utils.state_dict_to_tensor import *

from model import *
# from mcts import *
# from alpha_zero import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
def test():
    env=PacmanEnvDecorator()
    env.reset(mode="local")
    pacman = PacmanAgent()
    ghost = GhostAgent()

    print("TEST of net predict")
    t=time.time()
    action1, value1 = pacman.predict(env.game_state())
    action2, value2 = ghost.predict(env.game_state())
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of mcts")
    print("Running search:")
    env.reset()
    t=time.time()
    mcts = MCTS(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, num_simulations=16)
    mcts.run()
    t=time.time()-t
    print(f"time:{t}")
    print("Running batch:")
    env.reset()
    t=time.time()
    mcts = MCTS(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, num_simulations=16)
    # mcts.run_batch()
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of self play")
    SEARCH_TIME=16
    env.reset()
    t=time.time()
    trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, search_time=SEARCH_TIME)
    # trainer.play()
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of agent train")
    traj = []
    for _ in range(78):
        state=env.game_state()
        env.reset(mode="local")

        prob_pacman = torch.rand(5).to(device)
        # prob_pacman = np.random.rand(5)
        prob_pacman = prob_pacman / prob_pacman.sum()
        value_pacman = torch.rand(1).to(device)
        #value_pacman = np.random.rand(1)

        prob_ghost = torch.rand(125).to(device)
        # prob_ghost = np.random.rand(125)
        prob_ghost = prob_ghost / prob_ghost.sum()
        value_ghost = torch.rand(1).to(device)
        # value_ghost = np.random.rand(1)

        dummy1 = 0.0
        dummy2 = 0.0
        
        traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, dummy1, dummy2))
    
    print("Running train:")
    t=time.time()
    pacman.train(traj)
    ghost.train(traj)
    t=time.time()-t
    print(f"time:{t}")
    print("Running batch:")
    t=time.time()
    pacman.train_batch(traj)
    ghost.train_batch(traj)
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of alphazero train")
    SEARCH_TIME=5
    env.reset()
    t=time.time()
    trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, iterations=1, episodes=3, check_time=2, search_time=SEARCH_TIME)
    trainer.train()
    t=time.time()-t
    print(f"time:{t}")
"""

class Pacman_zero:
    def __init__(self):
        self.env=PacmanEnvDecorator(PacmanEnv())
        self.env.reset()
        self.c_puct=2.5
        self.search_time=4
        self.pacman=PacmanAgent(load_series='03280537')
        # self.ghost=GhostAgent(series='03280537')

    def __call__(self, state):
        action, prob, value = self.pacman.predict(self.env.game_state())
        return action

        # self.env.restore(state)
        # mcts=MCTS(self.env, self.pacman, self.ghost, self.c_puct, num_simulations=self.search_time)
        # return mcts.play_game_pacman()

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