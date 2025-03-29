import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from core.gamedata import *
from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.ghostact_int2list import *
from utils.PacmanEnvDecorator import *
from utils.seed import *

from data import *
from model import *
from mcts import *
from train_bc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlphaZeroTrainer:
    def __init__(self, env:PacmanEnv, pacman:Agent, ghost:Agent, c_puct:int, n_search:int, temp:float, iterations:int, episodes:int, logger:logging=None, writer:SummaryWriter=None):
        self.env=PacmanEnvDecorator(env)

        self.pacman=pacman
        self.ghost=ghost
        
        self.episodes=episodes
        self.iterations=iterations
        
        self.c_puct=c_puct
        self.n_search=n_search
        self.temp=temp

        self.logger=logger
        self.writter=writer

    def decide(self):
        mcts_pacman = MCTS_pacman(env=self.env, pacman=self.pacman, ghost=self.ghost, c_puct=self.c_puct, n_search=self.n_simulations, temp=self.temp, det=False)
        mcts_ghost = MCTS_ghost(env=self.env, ghost=self.ghost,pacman=self.pacman, c_puct=self.c_puct, n_search=self.n_search, temp=self.temp, det=False)

        return mcts_pacman.run(), mcts_ghost.run()

    def selfplay(self):
        traj=[]

        while True:
            decision_pacman, decision_ghost = self.decide()
            selected_action_pacman, action_prob_pacman, value_pacman = decision_pacman
            selected_action_ghost, action_prob_ghost, value_ghost = decision_ghost
            dict, reward_pacman, reward_ghost, done, eatAll = self.env.step(selected_action_pacman, selected_action_ghost)
            state=self.env.game_state()
            traj.append((state, action_prob_pacman, value_pacman, action_prob_ghost, value_ghost, self.env.is_eaten(), self.env.is_gone()))
            
            if done:
                print("game end")
                break
        
        return traj
    
    def train_from_json(self):
        batch_size = 512
        num_epochs = 20

        train_dataset_pacman = Dataset("data/train_dataset_pacman.pt")
        val_dataset_pacman = Dataset("data/val_dataset_pacman.pt")
        test_dataset_pacman = DataLoader("data/test_data_pacman.pt")
        train_loader_pacman = DataLoader(train_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_pacman = DataLoader(val_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader_pacman = DataLoader(test_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        train_dataset_ghost = Dataset("data/train_dataset_ghost.pt")
        val_dataset_ghost = Dataset("data/val_dataset_ghost.pt")
        test_dataset_ghost = DataLoader("data/test_data_ghost.pt")
        train_loader_ghost = DataLoader(train_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_ghost = DataLoader(val_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader_ghost = DataLoader(test_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        trainer = BCTrainer(self.pacman, train_loader_pacman, val_loader_pacman, test_loader_pacman, num_epochs=num_epochs, logger=self.logger)
        trainer.train()
        self.pacman.save_model()

        trainer = BCTrainer(self.ghost, train_loader_ghost, val_loader_ghost, test_loader_ghost, num_epochs=num_epochs, logger=self.logger)
        trainer.train()
        self.ghost.save_model()

    def train_from_selfplay(self, trajs):
        batch_size = 128
        num_epochs = 5

        dataset_synthesize_from_traj(trajs, batch_size=batch_size)

        train_dataset_pacman = PacmanDataset("selfplay/train_dataset_pacman.pt")
        val_dataset_pacman = PacmanDataset("selfplay/val_dataset_pacman.pt")
        test_dataset_pacman = PacmanDataset("selfplay/test_data_pacman.pt")
        train_loader_pacman = DataLoader(train_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_pacman = DataLoader(val_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader_pacman = DataLoader(test_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        train_dataset_ghost = PacmanDataset("selfplay/train_dataset_ghost.pt")
        val_dataset_ghost = PacmanDataset("selfplay/val_dataset_ghost.pt")
        test_dataset_ghost = PacmanDataset("selfplay/test_data_ghost.pt")
        train_loader_ghost = DataLoader(train_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_ghost = DataLoader(val_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader_ghost = DataLoader(test_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        trainer = BCTrainer(self.pacman, train_loader_pacman, val_loader_pacman, test_loader_pacman, num_epochs=num_epochs, logger=self.logger)
        trainer.train()
        self.pacman.save_model()

        trainer = BCTrainer(self.ghost, train_loader_ghost, val_loader_ghost, test_loader_ghost, num_epochs=num_epochs, logger=self.logger)
        trainer.train()
        self.ghost.save_model()

    def pipeline(self):
        for ite in range(self.iterations):
            trajs=[]
            
            self.env.reset()
            inistate=self.env.game_state()
            
            for epi in range(self.episodes):
                self.env.restore(inistate)
                traj = self.selfplay()
                trajs.append(traj)
            
            self.train_from_selfplay(trajs)

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    logger = get_logger(name="PacmanZeroLog", seed=SEED, log_file="log/train_zero_{}.log".format(datetime.datetime.now().strftime("%m%d%H%M")))

    env = PacmanEnvDecorator()
    env.reset()

    pacman=PacmanAgent()
    ghost=GhostAgent()

    SEARCH_TIME=10
    trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=2.5, n_search=SEARCH_TIME, temp=1, iterations=100, episodes=10)

    trainer.pipeline()