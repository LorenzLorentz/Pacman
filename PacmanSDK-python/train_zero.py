from core.GymEnvironment import PacmanEnv
from model_zero import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = PacmanEnv("local")
env.reset()

pacman=PacmanAgent()
ghost=GhostAgent()

SEARCH_TIME=40
trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, search_time=SEARCH_TIME)

trainer.train()