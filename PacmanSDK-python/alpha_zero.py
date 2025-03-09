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

from model import *
from mcts import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_short = datetime.datetime.now().strftime('%H%M')
time_long = datetime.datetime.now().strftime('%m%d%H%M')
# sys.stdout = open(f'log_zero/output_{time_short}.log', 'w')

class AlphaZeroTrainer:
    def __init__(self, env, pacman, ghost, c_puct, iterations=10, episodes=32, check_time=5, search_time=32):
        self.env=env

        self.pacman=pacman
        self.ghost=ghost
        self.c_puct=c_puct

        self.iterations=iterations
        self.episodes=episodes
        self.check_time=check_time
        self.search_time=search_time

        self.best_score=0.0

    def decide(self):
        mcts=MCTS(self.env, self.pacman, self.ghost, self.c_puct, num_simulations=self.search_time)
        return mcts.run()

    def play(self):
        traj=[]
        reward_pacman=0.0
        reward_ghost=0.0
        # step=0

        print("\n")
        while True:
            decision_pacman, decision_ghost = self.decide()
            selected_action_pacman, action_prob_pacman, value_pacman = decision_pacman
            selected_action_ghost, action_prob_ghost, value_ghost = decision_ghost
            dict, reward_pacman, reward_ghost, done, eatAll = self.env.step(selected_action_pacman, ghostact_int2list(selected_action_ghost))
            state=self.env.game_state()
            traj.append((state, action_prob_pacman, value_pacman, action_prob_ghost, value_ghost, reward_pacman, reward_ghost))
            print(f"pacman action: {selected_action_pacman}, ghost action {ghostact_int2list(selected_action_ghost)} \n")
            
            if done:
                print("game end")
                break
        return traj, reward_pacman, reward_ghost

    def learn(self, trajs):
        for traj in trajs:
            loss_pacman=self.pacman.train_batch(traj)
            loss_ghost=self.ghost.train_batch(traj)
        # self.env.reset()
        return (loss_pacman, loss_ghost)

    def train(self):
        for ite in range(self.iterations):
            print(f"ite {ite}")
            t=time.time()
            trajs=[]
            self.env.reset()
            inistate=self.env.game_state()
            for epi in range(self.episodes):
                self.env.restore(inistate)
                traj, _, _ = self.play()
                trajs.append(traj)
            t=time.time()-t
            print(f"self_play time {t}")
            t=time.time()
            loss_pacman, loss_ghost = self.learn(trajs)
            t=time.time()-t
            print(f"learning time {t}")

            score_pacman=0.0
            score_ghost=0.0
            for check in range(self.check_time):
                self.env.reset()
                _, reward_pacman, reward_ghost = self.play()
                score_pacman+=reward_pacman
                score_ghost+=sum(reward_ghost)
            score_pacman/=self.check_time
            score_ghost/=self.check_time
            
            print(f"Iteration: {ite+1}/{self.iterations}, loss_pacman = {loss_pacman}, loss_ghost = {loss_ghost}, score_pacman = {score_pacman}, score_ghost = {score_ghost}")

            if(score_pacman+score_ghost>self.best_score):
                self.best_score=score_ghost+score_pacman
                print(f"NEW  BEST with score {score_pacman+score_ghost}")
                self.pacman.save_model()
                self.ghost.save_model()