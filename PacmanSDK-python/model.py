import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.amp import autocast, GradScaler
from torch.cuda.amp import autocast, GradScaler

from core.gamedata import *
from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.ghostact_int2list import *
from utils.PacmanEnvDecorator import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ValueNet(nn.Module):

    # Feature: Conv -> BN -> ReLU + ResidualBlock
    # Policy Head: Conv(2 filters) -> BN -> ReLU -> Flatten -> FC -> (log)softmax
    # Value Head: Conv(1 filter) -> BN -> ReLU -> Flatten -> FC -> ReLU -> FC -> Tanh

    def __init__(
        self,
        in_channels=7,
        board_size=42,
        num_filters=128,
        num_res_blocks=14,
        policy_out_dim=5,
        if_Pacman=True
    ):
        super(ValueNet, self).__init__()

        if(if_Pacman):
            policy_out_dim=5
        else:
            policy_out_dim=5*5*5

        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_out_dim)

        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.softmax(p, dim=1)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

class PacmanAgent:
    def __init__(self, batch_size=32, load_series=None, save_series=None):
        self.load_series=load_series
        self.save_series=save_series
        
        self.ValueNet=ValueNet(if_Pacman=True)
        # self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.optimizer=optim.SGD(self.ValueNet.parameters(), lr=5e-2)
        # self.scaler = GradScaler('cuda')
        self.scaler = GradScaler()
        
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
        self.batch_size=batch_size
    
    def init_weight(self, model, load_name=None):
        if load_name:
            model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
            return
        if self.load_series:
            name = f"model/pacman_zero_{self.load_series}.pth"
            if os.path.exists(name):
                model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
                return
        print("No checkpoint found. Training from scratch.")
        model.init_weights()

    def save_model(self, save_name=None):
        if save_name:
            torch.save(self.ValueNet.state_dict(), save_name)
            return
        if self.save_series:
            torch.save(self.ValueNet.state_dict(), f"model/pacman_zero_{self.save_series}.pth")
            return
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.ValueNet.state_dict(), f"model/pacman_zero_{time}.pth")
    
    def predict(self, state):
        # input: state as type: gamestate
        # output: selected_action, action_prob, value
        # in mcts version, actual selected_action is decided other way
        pos = state.gamestate_to_statedict()["pacman_coord"]
        legal_action = get_valid_moves_pacman(pos, state)

        act_probs, value = self.ValueNet(state2tensor(state))
        
        act_probs = act_probs.squeeze()
        value = value.squeeze()

        act_probs_legal=torch.zeros_like(act_probs)
        act_probs_legal[legal_action]=act_probs[legal_action]

        selected_action = torch.multinomial(act_probs_legal, 1).item()

        return selected_action, act_probs_legal, value
    
    def predict_batch(self, states_tensor, legal_actions_mask):
        act_probs, values = self.ValueNet(states_tensor)
        act_probs_legal = act_probs * legal_actions_mask
        selected_actions = torch.multinomial(act_probs_legal, 1)
        return selected_actions, act_probs_legal, values
    
    def ppo_train(self, states_tensor, legal_actions_mask, old_prob, actions, td_target, advantages, eps):
        # with autocast('cuda'):
        with autocast():
            _, new_prob, new_values=self.predict_batch(states_tensor, legal_actions_mask)
            new_prob=new_prob.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio=new_prob/old_prob

            loss_actor=-torch.mean(torch.min(ratio*advantages, torch.clamp(ratio, 1-eps, 1+eps)*advantages))
            loss_critic=F.mse_loss(td_target, new_values.squeeze())
            
            loss=loss_actor+loss_critic

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        print(f"    loss_pacman:{loss}")

    def zero_train(self, traj):
        # traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, reward_pacman, reward_ghost))
        # loss = mse + cross_entopy
        for point in traj:
            state, prob_pacman, value_pacman, _, _, _, _ = point
            # prob_pacman = torch.from_numpy(prob_pacman, device = device)
            # with autocast('cuda'):
            with autocast():
                _, prob_pacman_predict, value_pacman_predict = self.predict(state)
                loss_value = (value_pacman - value_pacman_predict)**2
                loss_prob = -sum(prob_pacman[action] * torch.log(prob_pacman_predict[action]) for action in range(5))
                loss = loss_value + loss_prob
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def zero_train_batch(self, traj):
        loss_return = []

        for i in range(0, len(traj), self.batch_size):
            batch = traj[i:i + self.batch_size]
            states = []
            probs_pacman = []
            values_pacman = []

            with torch.no_grad():
                for point in batch:
                    state, prob_pacman, value_pacman, _, _, _, _ = point
                    states.append(state2tensor(state))
                    probs_pacman.append(prob_pacman)
                    values_pacman.append(value_pacman)
                
                state_batch = torch.cat(states)
                prob_pacman = torch.stack(probs_pacman).to(device).view(-1, 5)
                value_pacman = torch.stack(values_pacman).to(device).view(-1, 1)
            
            # with autocast('cuda'):
            with autocast():
                prob_pacman_predict, value_pacman_predict = self.ValueNet(state_batch)
                loss_value = F.mse_loss(value_pacman_predict, value_pacman)
                loss_prob = F.kl_div(torch.log(prob_pacman_predict + 1e-8), prob_pacman, reduction='batchmean')
                loss = loss_value + loss_prob

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_return.append(loss)
        
        return loss_return[-1]
    
class GhostAgent:
    def __init__(self, batch_size=32, load_series=None, save_series=None):
        self.save_series=save_series
        self.load_series=load_series
        
        self.ValueNet=ValueNet(if_Pacman=False)
        # self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.optimizer=optim.SGD(self.ValueNet.parameters(), lr=5e-2)
        # self.scaler = GradScaler('cuda')
        self.scaler = GradScaler()
        
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
        self.batch_size=batch_size
    
    def init_weight(self, model, load_name=None):
        if load_name:
            model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
            return
        if self.load_series:
            name = f"model/ghost_zero_{self.load_series}.pth"
            if os.path.exists(name):
                model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
                return
        print("No checkpoint found. Training from scratch.")
        model.init_weights()

    def save_model(self, save_name=None):
        if save_name:
            torch.save(self.ValueNet.state_dict(), save_name)
            return
        if self.save_series:
            torch.save(self.ValueNet.state_dict(), f"model/ghost_zero_{self.save_series}.pth")
            return
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.ValueNet.state_dict(), f"model/ghost_zero_{time}.pth")

    def predict(self, state):
        pos = state.gamestate_to_statedict()["ghosts_coord"]
        legal_actions = get_valid_moves_ghost(pos, state)

        act_prob, value = self.ValueNet(state2tensor(state))

        act_prob = act_prob.squeeze()
        value = value.squeeze()

        act_prob_legal=torch.zeros_like(act_prob)
        act_prob_legal[legal_actions]=act_prob[legal_actions]

        seleted_action = torch.multinomial(act_prob_legal, 1).item()

        return seleted_action, act_prob, value
    
    def predict_batch(self, states_tensor, legal_actions_mask):
        act_probs, values = self.ValueNet(states_tensor)
        act_probs_legal = act_probs * legal_actions_mask
        selected_actions = torch.multinomial(act_probs_legal, 1)
        return selected_actions, act_probs_legal, values
    
    def ppo_train(self, states_tensor, legal_actions_mask, old_prob, actions, td_target, advantages, eps):
        # with autocast('cuda'):
        with autocast():
            _, new_prob, new_values=self.predict_batch(states_tensor, legal_actions_mask)
            new_prob=new_prob.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio=new_prob/old_prob
            
            loss_actor=-torch.mean(torch.min(ratio*advantages, torch.clamp(ratio, 1-eps, 1+eps)*advantages))
            loss_critic=F.mse_loss(td_target, new_values.squeeze())
            
            loss=loss_actor+loss_critic

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        print(f"    loss_ghost:{loss}")

    def zero_train(self, traj):
        for point in traj:
            state, _, _, prob_ghost, value_ghost, _, _ = point
            # prob_ghost = torch.from_numpy(prob_ghost, device = device)
            # with autocast('cuda'):
            with autocast():
                _, prob_ghost_predict, value_ghost_predict = self.predict(state)
                loss_value = (value_ghost - value_ghost_predict)**2
                loss_prob = -sum(prob_ghost[action] * torch.log(prob_ghost_predict[action]) for action in range(15))
                loss = loss_value + loss_prob   
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def zero_train_batch(self, traj, batch_size=32):
        loss_return = []
        for i in range(0, len(traj), batch_size):
            batch = traj[i:i + batch_size]
            states = []
            probs_ghost = []
            values_ghost = []

            with torch.no_grad():
                for point in batch:
                    state, _, _, prob_ghost, value_ghost, _, _ = point
                    states.append(state2tensor(state))
                    probs_ghost.append(prob_ghost)
                    values_ghost.append(value_ghost)
                
                state_batch = torch.cat(states)
                prob_ghost = torch.stack(probs_ghost).view(-1, 125)
                value_ghost = torch.stack(values_ghost).view(-1, 1)
            
            # with autocast('cuda'):
            with autocast():
                prob_ghost_predict, value_ghost_predict = self.ValueNet(state_batch)
                loss_value = F.mse_loss(value_ghost_predict, value_ghost)
                loss_prob = F.kl_div(torch.log(prob_ghost_predict + 1e-8), prob_ghost, reduction='batchmean')
                loss = loss_value + loss_prob

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            loss_return.append(loss)

        return loss_return[-1]