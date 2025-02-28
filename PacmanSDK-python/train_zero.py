from model_zero import *

env = PacmanEnvDecorator()
env.reset()

pacman=PacmanAgent()
ghost=GhostAgent()

SEARCH_TIME=32
trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=2.5, search_time=SEARCH_TIME)

trainer.train()