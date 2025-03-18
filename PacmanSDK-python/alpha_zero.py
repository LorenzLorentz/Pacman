import os
import sys
import time
import datetime

import numpy as np
import torch

from core.gamedata import *
from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.ghostact_int2list import *
from utils.PacmanEnvDecorator import *

from data import *
from model import *
from mcts import *
from train_bc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_short = datetime.datetime.now().strftime('%H%M')
time_long = datetime.datetime.now().strftime('%m%d%H%M')
# sys.stdout = open(f'log_zero/output_{time_short}.log', 'w')

class AlphaZeroTrainer:
    def __init__(self, env:PacmanEnv, pacman:Agent, ghost:Agent, c_puct:int, n_simulations:int, n_search:int, temp:float, iterations:int, episodes:int):
        self.env=PacmanEnvDecorator(env)

        self.pacman=pacman
        self.ghost=ghost
        
        self.episodes=episodes
        self.iterations=iterations
        
        self.c_puct=c_puct
        self.n_simulations=n_simulations
        self.n_search=n_search
        self.temp=temp

    def decide(self):
        mcts_pacman = MCTS(self.env, self.pacman, self.ghost, self.c_puct, self.n_simulations, self.n_search, self.temp, det=False)
        mcts_ghost = MCTS(self.env, self.ghost, self.pacman, self.c_puct, self.n_simulations, self.n_search, self.temp, det=False)

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
        train_loader_pacman = DataLoader(train_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_pacman = DataLoader(val_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader_pacman = DataLoader(test_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)

        train_dataset_ghost = Dataset("data/train_dataset_ghost.pt")
        val_dataset_ghost = Dataset("data/val_dataset_ghost.pt")
        test_dataset_ghost = DataLoader("data/test_data_ghost.pt")
        train_loader_ghost = DataLoader(train_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_ghost = DataLoader(val_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader_ghost = DataLoader(test_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
        
        trainer = Trainer(self.pacman, train_loader_pacman, val_loader_pacman, test_loader_pacman, num_epochs=num_epochs)
        trainer.train()
        self.pacman.save_model()

        trainer = Trainer(self.ghost, train_loader_ghost, val_loader_ghost, test_loader_ghost, num_epochs=num_epochs)
        trainer.train()
        self.ghost.save_model()

    def train_from_selfplay(self, trajs):
        batch_size = 128
        num_epochs = 5

        dataset_synthesize_from_traj(trajs, batch_size=batch_size)

        train_dataset_pacman = Dataset("selfplay/train_dataset_pacman.pt")
        val_dataset_pacman = Dataset("selfplay/val_dataset_pacman.pt")
        test_dataset_pacman = DataLoader("selfplay/test_data_pacman.pt")
        train_loader_pacman = DataLoader(train_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_pacman = DataLoader(val_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader_pacman = DataLoader(test_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)

        train_dataset_ghost = Dataset("selfplay/train_dataset_ghost.pt")
        val_dataset_ghost = Dataset("selfplay/val_dataset_ghost.pt")
        test_dataset_ghost = DataLoader("selfplay/test_data_ghost.pt")
        train_loader_ghost = DataLoader(train_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_ghost = DataLoader(val_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader_ghost = DataLoader(test_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)

        trainer = Trainer(self.pacman, train_loader_pacman, val_loader_pacman, test_loader_pacman, num_epochs=num_epochs)
        trainer.train()
        self.pacman.save_model()

        trainer = Trainer(self.ghost, train_loader_ghost, val_loader_ghost, test_loader_ghost, num_epochs=num_epochs)
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