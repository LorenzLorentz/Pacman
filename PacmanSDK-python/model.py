import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
# from torch.cuda.amp import autocast, GradScaler

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
    
class GlobalFeature(nn.Module):
    def __init__(self, in_channels=14, out_channels=512, num_res_blocks=6):
        super(GlobalFeature, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.encoder = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_res_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class LocalFeature(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_res_blocks=2):
        super(LocalFeature, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.encoder = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_res_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class ExtraFeature(nn.Module):
    def __init__(self, in_channels, in_features, out_features=64):
        super(ExtraFeature, self).__init__()
        self.fc = nn.Linear(in_features*in_channels, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
    
class InputParser(nn.Module):
    def __init__(self,
                 board_embedding_num=10,
                 board_embedding_dim=14,
                 extrainfo_embedding=7):
        super(InputParser, self).__init__()

        self.board_embedding = nn.Embedding(num_embeddings=board_embedding_num, embedding_dim=board_embedding_dim)
        self.extrainfo_embedding = nn.Embedding(num_embeddings=2*board_embedding_num, embedding_dim=extrainfo_embedding)

    def forward(self, x):
        board = x[:, 0, :, :].long()
        board_emb = self.board_embedding(board)
        global_input = board_emb.permute(0, 3, 1, 2)

        nearby = []
        for i in range(5):
            nearby.append(x[:, 1, 0:7, 7*i:7*(i+1)])
        local_input = torch.stack(nearby, dim=1)

        extra_info = x[:, 1, 0:1, 35:35+7].view(x.shape[0], -1).long()
        extra_input = self.extrainfo_embedding(extra_info)
        extra_input = extra_input.permute(0, 2, 1)
        
        return global_input, local_input, extra_input

class ValueNet(nn.Module):
    def __init__(self, 
                 global_in_channels=14,
                 local_in_channels=5,
                 extra_in_channels=7,
                 extra_in_features=7,
                 global_filters=512,
                 local_filters=128,
                 extra_out_features=64,
                 if_Pacman = True):
        
        super(ValueNet, self).__init__()
        self.if_Pacman = if_Pacman
        
        self.parser = InputParser(board_embedding_dim=global_in_channels, extrainfo_embedding=extra_in_channels)
        
        self.global_feature = GlobalFeature(in_channels=global_in_channels, out_channels=global_filters, num_res_blocks=6)
        self.local_feature = LocalFeature(in_channels=local_in_channels, out_channels=local_filters, num_res_blocks=4)
        self.extra_feature = ExtraFeature(in_channels=extra_in_channels, in_features=extra_in_features, out_features=extra_out_features)

        combined_dim = global_filters + local_filters + extra_out_features
        self.bn_global = nn.BatchNorm1d(global_filters)
        self.bn_local = nn.BatchNorm1d(local_filters)
        self.bn_extra = nn.BatchNorm1d(extra_out_features)
        self.fc1_combined = nn.Linear(combined_dim, global_filters)
        self.bn1_combined = nn.BatchNorm1d(global_filters)
        self.fc2_combined = nn.Linear(global_filters, global_filters)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        
        out_dim = 5 if if_Pacman else 125
        self.policy_head = nn.Linear(global_filters, out_dim)
        self.value_head = nn.Linear(global_filters, 1)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                with torch.no_grad():
                    one_hot = torch.eye(m.num_embeddings, m.embedding_dim)
                    noise = torch.randn_like(one_hot) * 0.1
                    m.weight.copy_(one_hot + noise)
        
    def forward(self, x):
        global_input, local_input, extra_input = self.parser(x)

        # global_input shape: [batch, global_in_channels, board_size, board_size]
        gf = self.global_feature(global_input)  # [batch, global_filters]
        
        # local_input shape: [batch, local_in_channels, local_size, local_size]
        lf = self.local_feature(local_input)    # [batch, local_filters]
        
        # extra_input shape: [batch, extra_in_channels, extra_in_features]
        ef = self.extra_feature(extra_input)    # [batch, extra_features]

        gf = self.bn_global(gf)
        lf = self.bn_local(lf)
        ef = self.bn_extra(ef)

        combined = torch.cat([gf, lf, ef], dim=1)
        combined = self.gelu(self.bn1_combined(self.fc1_combined(combined)))
        combined = self.dropout(combined)
        combined = self.gelu(self.fc2_combined(combined))

        if torch.isnan(gf).any() or torch.isinf(gf).any():
            raise ValueError("NaN or Inf detected in global feature")
        
        if torch.isnan(lf).any() or torch.isinf(lf).any():
            raise ValueError("NaN or Inf detected in local feature")
        
        if torch.isnan(ef).any() or torch.isinf(ef).any():
            raise ValueError("NaN or Inf detected in extra feature")
        
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            raise ValueError("NaN or Inf detected in combined")

        # torch.clamp(self.policy_head(combined), min=-10, max=10)
        policy = self.policy_head(combined) # F.softmax(self.policy_head(combined), dim=1)
        
        value = self.value_head(combined)
        if not self.if_Pacman:
            value = F.tanh(value)
        
        return policy, value

class Agent:
    def __init__(self, is_pacman:bool, batch_size:int=32, load_series:str=None, save_series:str=None):
        self.isPacman = is_pacman

        self.load_series=load_series
        self.save_series=save_series
        
        self.ValueNet=ValueNet(if_Pacman=is_pacman)
        self.optimizer=optim.Adam(self.ValueNet.parameters(), lr=1e-5)
        # self.optimizer=optim.SGD(self.ValueNet.parameters(), lr=5e-2)
        if torch.cuda.is_available():
            self.scaler = GradScaler('cuda')
         # self.scaler = GradScaler()
        
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
        self.batch_size=batch_size

    def is_pacman(self) -> bool:
        return self.isPacman
    
    def name(self) -> str:
        return 'pacman' if self.is_pacman() else 'ghost'

    def init_weight(self, model:ValueNet, load_name:str=None) -> None:
        if load_name:
            model.load_state_dict(torch.load(load_name, map_location=device, weights_only=True))
            return
        if self.load_series:
            name = f"model/{self.name()}_{self.load_series}.pth"
            if os.path.exists(name):
                model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
                return
        print("No checkpoint found. Training from scratch.")
        model.init_weights()
    
    def save_model(self, save_name:str=None):
        if save_name:
            torch.save(self.ValueNet.state_dict(), save_name)
            return
        if self.save_series:
            torch.save(self.ValueNet.state_dict(), f"model/{self.name()}_{self.save_series}.pth")
            return
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.ValueNet.state_dict(), f"model/{self.name()}_{time}.pth")
    
    def predict(self, state:GameState):
        raise NotImplementedError
    
    def predict_batch(self, states_tensor:torch.tensor, legal_actions_mask:torch.tensor):
        act_probs, values = self.ValueNet(states_tensor)
        act_probs_legal = act_probs * legal_actions_mask
        selected_actions = torch.multinomial(act_probs_legal, 1)
        return selected_actions, act_probs_legal, values

class PacmanAgent(Agent):
    def __init__(self, batch_size:int=32, load_series:str=None, save_series:str=None):
        super().__init__(is_pacman = True, batch_size=batch_size, load_series=load_series, save_series=save_series)
    
    def predict(self, state:GameState):
        # input: state as type: gamestate
        # output: selected_action, action_prob, value
        # in mcts version, actual selected_action is decided other way
        self.ValueNet.eval()

        pos = state.gamestate_to_statedict()["pacman_coord"]
        legal_action = get_valid_moves_pacman(pos, state)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    act_probs, value = self.ValueNet(state2tensor(state))
            else:
                act_probs, value = self.ValueNet(state2tensor(state))
            
            act_probs = torch.softmax(act_probs, dim=1)
            act_probs = act_probs.squeeze()
            value = value.squeeze()

            act_probs_legal=torch.zeros_like(act_probs)
            act_probs_legal[legal_action]=act_probs[legal_action]

            selected_action = torch.multinomial(act_probs_legal, 1).item()

        return selected_action, act_probs_legal, value
    
class GhostAgent(Agent):
    def __init__(self, batch_size:int=32, load_series:str=None, save_series:str=None):
        super().__init__(is_pacman = False, batch_size=batch_size, load_series=load_series, save_series=save_series)

    def predict(self, state):
        self.ValueNet.eval()

        pos = state.gamestate_to_statedict()["ghosts_coord"]
        legal_actions = get_valid_moves_ghost(pos, state)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    act_probs, value = self.ValueNet(state2tensor(state))
            else:
                act_probs, value = self.ValueNet(state2tensor(state))

            act_probs = torch.softmax(act_probs, dim=1)
            act_probs = act_probs.squeeze()
            value = value.squeeze()

            act_prob_legal=torch.zeros_like(act_probs)
            act_prob_legal[legal_actions]=act_probs[legal_actions]

            seleted_action = torch.multinomial(act_prob_legal, 1).item()

        return seleted_action, act_probs, value