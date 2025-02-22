import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from core.gamedata import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.data_process import *

device = torch.device("cuda" if torch.cuda() else "cpu")

class MCTSNode:
    def __init__(self, env, done=True, parent=None):
        self.env=copy.deepcopy(env)
        self.state=env.game_state()
        self.state_dict=self.state.gamestate_to_statedict()
        self.done=done

        self.parent=parent
        self.children={}   # dict: action -> child

        self.N=0
        self.P_pacman={}
        self.P_ghost={}    # dict: action -> prob
        self.W_pacman=0.0  # accum prob
        self.W_ghost=0.0
        self.Q_pacman=0.0
        self.Q_ghost=0.0   # ave prob

    def is_terminal(self):
        return self.done

    def is_expanded(self):
        return len(self.P)>0
    
    def expand(self, pacman, ghost):
        action_probs_pacman, value_pacman = pacman.predict(self.state)
        action_probs_ghost, value_ghost = ghost.predict(self.state)

        actions_pacman=list(action_probs_pacman.keys())
        actions_ghost=list(action_probs_ghost.keys())

        probs_pacman = np.array([action_probs_pacman[a] for a in actions_pacman])
        probs_ghost = np.array([action_probs_ghost[a] for a in actions_ghost])

        self.P_pacman = {a: p for a, p in zip(actions_pacman, probs_pacman)}
        self.P_ghost = {a: p for a, p in zip(action_ghost, probs_ghost)}

        for action_pacman in actions_pacman:
            for action_ghost in actions_ghost:
                if (action_pacman, action_ghost) not in self.children:
                    _, _, _, done, _  =  self.env.step(action_pacman, action_ghost)
                    self.children[(action_pacman, action_ghost)] = MCTSNode(self.env, done, parent=self)

        return value_pacman, value_ghost

    def select(self, c_puct):
        bese_score=-float('inf')
        best_action_pacman=None
        best_action_ghost=None
        best_child=None

        total_visits=self.N if self.N>0 else 1
        for (action_pacman, action_ghost), child in self.children.items():
            score = (child.Q + c_puct*(self.P_pacman[action_pacman])*np.sqrt(total_visits)/(1+child.N)
                            + c_puct*(self.P_ghost[action_ghost])*np.sqrt(total_visits)/(1+child.N))
            if score>bese_score:
                bese_score=score
                best_action_pacman=action_pacman
                best_action_ghost=action_ghost
                best_child=child
        
        return best_action_pacman, best_action_ghost, best_child
    
    def update(self, value):
        value_pacman, value_ghost = value

        self.N += 1
        self.W_pacman += value_pacman
        self.W_pacman += value_ghost

        self.Q_pacman = self.W_pacman / self.N
        self.Q_ghost = self.W_ghost / self.N

class MCTS:
    def __init__(self, env, pacman, ghost, c_puct, temperature=1, num_simulations=1600):
        self.env=copy.deepcopy(env)

        self.pacman=pacman
        self.ghost=ghost
        
        self.c_puct=c_puct
        self.temp_inverse=1/temperature
        self.num_simulations=num_simulations

    def search(self, node):
        if node.is_terminal():
            return node.state_dict["score"]
        
        if not node.is_expanded():
            value=node.expand(self.pacman, self.ghost)
            node.update(value)
            return value
        
        _, _, child=node.select(self.c_puct)

        value=self.search(child)
        node.update(value)
        return value
    
    def run(self):
        self.root = MCTSNode(self.env, self.state)
        for _ in range(self.num_simulations):
            self.search(self.root)

        action_prob = {}
        sum_visits = 0

        for action, child in self.root.children.items():
            visit = child.N ** self.temp_inverse
            action_prob[action] = visit
            sum_visits += visit

        sum_visits = sum_visits or 1e-8
        for action in action_prob:
            action_prob[action] /= sum_visits

        actions = list(action_prob.keys())
        probabilities = [action_prob[action] for action in actions]
        selected_action = np.random.choice(actions, p=probabilities)

        return selected_action, action_prob, self.root.Q_pacman, self.root.Q_ghost

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
        num_filters=256,
        num_res_blocks=7,
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
        p = F.log_softmax(p, dim=1)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

class PacmanAgent:
    def __init__(self):
        self.ValueNet=ValueNet(if_Pacman=True)
        self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.init_weight(self.ValueNet)
    
    def init_weight(self, model, name="pacman_zero.pth"):
        if os.path.exists(name):
            model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
        else:
            print("No checkpoint found. Training from scratch.")
            model.init_weights()

    def save_model(self, model, name="pacman_zero.pth"):
        torch.save(model.state_dict(), name)
    
    def predict(self, state):
        pos = state.gamestate_to_statedict()["pacman_coord"]
        legal_action = get_valid_moves_pacman(pos, state)

        act_probs, value = self.ValueNet(state)
        act_probs = zip(legal_action, act_probs[legal_action])

        value_float = value.float()

        return act_probs, value_float

    def train(self, traj):
        # traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, reward_pacman, reward_ghost))
        # loss = mse + cross_rntopy
        for point in traj:
            state, prob_pacman, value_pacman, _, _, _, _, _ = point
            prob_pacman_predict, value_pacman_predict = self.predict(state)
            loss_value = (value_pacman, value_pacman_predict)**2
            loss_prob = -sum(prob_pacman[action] * np.log(prob_pacman_predict[action]) for action in prob_pacman)
            loss = loss_value + loss_prob
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
class GhostAgent:
    def __init__(self):
        self.ValueNet=ValueNet(if_Pacman=True)
        self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.init_weight(self.ValueNet)
    
    def init_weight(self, model, name="ghost_zero.pth"):
        if os.path.exists(name):
            model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
        else:
            print("No checkpoint found. Training from scratch.")
            model.init_weights()

    def save_model(self, model, name="ghost_zero.pth"):
        torch.save(model.state_dict(), name)

    def predict(self, state):
        pos = state.gamestate_to_statedict()["ghosts_coord"]
        legal_actions = get_valid_moves_ghost(pos, state)

        act_probs, value = self.ValueNet(state)

        act_probs_new = {}
        for action in enum(legal_actions):
            index = action[0] + action[1]*5 + action[2]*25
            act_probs_new[action] = act_probs[index]

        value_float = value.float()

        return act_probs_new, value_float

    def train(self, traj):
        for point in traj:
            state, _, _, prob_ghost, value_ghost, _, _, _ = point
            prob_ghost_predict, value_ghost_predict = self.predict(state)
            loss_value = (value_ghost - value_ghost_predict)**2
            loss_prob = -sum(prob_ghost[action] * np.log(prob_ghost_predict[action]) for action in prob_ghost)
            loss = loss_value + loss_prob
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class AlphaZeroTrainer:
    def __init__(self, env, pacman, ghost, c_puct, iterations=100, episodes=100, check_time=10, search_time=100):
        self.env=env

        self.pacman=pacman
        self.ghost=ghost
        self.c_puct=c_puct
        self.MCTS=MCTS(self.env, self.pacman, self.ghost, self.c_puct)

        self.iterations=iterations
        self.episodes=episodes
        self.check_time=check_time
        # self.search_time=search_time

        self.best_score=0.0

    def decide(self):
        (action_prob_pacman, action_prob_ghost, value_pacman, value_ghost) = self.MCTS.run()
        decision_pacman = (action_prob_pacman, value_pacman)
        decision_ghost = (action_prob_ghost, value_ghost)
        return (decision_pacman, decision_ghost)

    def play(self):
        traj=[]
        reward_pacman=0.0
        reward_ghost=0.0
        while True:
            decision_pacman, decision_ghost = self.decide()
            selected_action_pacman, action_prob_pacman, value_pacman = decision_pacman
            selected_action_ghost, action_prob_ghost, value_ghost = decision_ghost
            _, reward_pacman, reward_ghost, done, eatAll = self.env.step(selected_action_pacman, selected_action_ghost)
            state=self.env.game_state()
            traj.append((state, action_prob_pacman, value_pacman, action_prob_ghost, value_ghost, reward_pacman, reward_ghost))
            if done:
                break
        return traj, reward_pacman, reward_ghost

    def learn(self, trajs):
        for traj in trajs:
            loss_pacman=self.pacman.train(traj)
            loss_ghost=self.ghost.train(traj)
        self.env.reset()
        return (loss_pacman, loss_ghost)

    def train(self):
        for ite in range(self.iterations):
            trajs=[]
            for epi in range(self.episodes):
                traj, _, _ = self.play()
                trajs.append(traj)
            loss_pacman, loss_ghost = self.learn(trajs)

            score_pacman=0.0
            score_ghost=0.0
            for check in range(self.check_time):
                _, reward_pacman, reward_ghost = self.play()
                score_pacman+=reward_pacman
                score_ghost+=reward_ghost
            score_pacman/=self.check_time
            score_ghost/=self.check_time
            
            print(f"Iteration: {ite}/{self.iterations}, loss_pacman = {loss_pacman}, loss_ghost = {loss_ghost}, 
                  score_pacman = {score_pacman}, score_ghost = {score_ghost}")

            if(score_pacman+score_ghost>self.best_score):
                print(f"NEW  BEST with score {score_pacman+score_ghost}")
                self.pacman.save_model()
                self.ghost.save_model()