import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler

from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, env):
        self.GAMMA=0.99
        self.LAMBDA=0.95
        self.EPS=0.2
        self.UPDATE_EPOCH=40

        self.env=PacmanEnvDecorator(env)
        
        self.pacman=PacmanAgent()
        self.ghost=GhostAgent()

        self.scaler = GradScaler('cuda')

    def play(self):
        states, actions_pacman, actions_ghost, values_pacman, next_values_pacman, values_ghost, next_values_ghost, rewards_pacman, rewards_ghost, dones = [], [], [], [],[], [],[], [],[], [],
        self.env.reset()
        while True:
            state=self.env.game_state()
            selected_action_pacman, _, value_pacman = self.pacman.predict(state)
            selected_action_ghost, _, value_ghost = self.ghost.predict(state)
            dict, reward_pacman, reward_ghost, done, eatAll=self.env.step(selected_action_pacman, ghostact_int2list(selected_action_ghost))
            next_state=self.env.game_state()
            _, _, next_value_pacman=self.pacman.predict(next_state)
            _, _, next_value_ghost=self.ghost.predict(next_state)
            
            states.append(state)
            
            actions_pacman.append(selected_action_pacman)
            actions_ghost.append(selected_action_ghost)
            values_pacman.append(value_pacman)
            values_ghost.append(value_ghost)
            next_values_pacman.append(next_value_pacman)
            next_values_ghost.append(next_value_ghost)
            rewards_pacman.append(reward_pacman)
            rewards_ghost.append(reward_ghost)
            dones.append(done)
            
            if done:
                break
        
        actions_pacman=torch.stack(actions_pacman).to(device)
        actions_ghost=torch.stack(actions_ghost).to(device)
        values_pacman=torch.stack(values_pacman).to(device)
        values_ghost=torch.stack(values_ghost).to(device)
        next_values_pacman=torch.stack(next_values_pacman).to(device)
        next_values_ghost=torch.stack(next_values_ghost).to(device)
        rewards_pacman=torch.stack(rewards_pacman).to(device)
        rewards_ghost=torch.stack(rewards_ghost).to(device)

        return states, actions_pacman, actions_ghost, values_pacman, next_values_pacman, values_ghost, next_values_ghost, rewards_pacman, rewards_ghost, dones

    def compute_advantage(self, td_delta):
        advantages=[]
        for delta in reversed(td_delta):
            advantage=self.GAMMA*self.LAMBDA*advantage+delta
            advantages.append(advantage)
        advantages.reverse()
        return torch.tensor(advantages, dtype=torch.float32)
    
    def ppo_train(self):
        states, actions_pacman, actions_ghost, values_pacman, next_values_pacman, values_ghost, next_values_ghost, rewards_pacman, rewards_ghost, dones = self.play()
        
        states_tensor=[]
        for state in states:
            state_tensor=state2tensor(state)
            states_tensor.append(state_tensor)
            pos = state.gamestate_to_statedict()["pacman_coord"]
            legal_action_mask = get_valid_moves_pacman_mask(pos, state)
            legal_actions_mask.append(legal_action_mask)
        states_tensor=torch.stack(states_tensor).to(device)
        legal_actions_mask=torch.stack(legal_actions_mask).to(device)

        _, old_prob_pacman, _ =self.pacman.predict_batch(states_tensor, legal_actions_mask)
        old_prob_pacman = old_prob_pacman.gather(actions_pacman)
        
        _, old_prob_ghost, _ =self.ghost.predict_batch(states_tensor, legal_actions_mask)
        old_prob_ghost=old_prob_ghost.gather(actions_ghost)
        
        td_target_pacman=rewards_pacman+self.GAMMA*next_values_pacman*(1-dones)
        td_delta_pacman=td_target_pacman-values_pacman
        advantages_pacman=self.compute_advantage(td_delta_pacman)
        
        td_target_ghost=rewards_ghost+self.GAMMA*next_values_ghost*(1-dones)
        td_delta_ghost=td_target_ghost-values_ghost
        advantages_ghost=self.compute_advantage(td_delta_ghost)
        
        for _ in range(self.UPDATE_EPOCH):
            self.pacman.ppo_train(states_tensor, legal_actions_mask, old_prob_pacman, actions_pacman, td_target_pacman, advantages_pacman, self.EPS)
            self.ghost.ppo_train(states_tensor, legal_actions_mask, old_prob_ghost, actions_ghost, td_target_pacman, advantages_ghost, self.EPS)