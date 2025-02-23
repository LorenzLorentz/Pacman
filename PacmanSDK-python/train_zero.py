from model_zero import *

env = PacmanEnvDecorator()
env.reset()

pacman=PacmanAgent()
ghost=GhostAgent()

SEARCH_TIME=40
trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, search_time=SEARCH_TIME)

trainer.train()