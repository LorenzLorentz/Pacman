from core.GymEnvironment import *

class PacmanEnvDecorator:
    def __init__(self, env=None):
        if env:
            self.env=env
        else:
            self.env = PacmanEnv("local")

    def reset(self, mode="local"):
        self.env.reset(mode=mode)

    def restore(self, state):
        self.env.ai_reset(state.gamestate_to_statedict())

    def step(self, pacmanAction, ghostAction, state=None):
        if state:
            self.restore(state)
        return self.env.step(pacmanAction, ghostAction)
    
    def game_state(self):
        return self.env.game_state()