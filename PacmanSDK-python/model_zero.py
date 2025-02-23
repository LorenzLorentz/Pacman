import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.amp import autocast, GradScaler

from core.gamedata import *
from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.data_process import *
from utils.ghostact_int2list import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PacmanEnvDecorator:
    def __init__(self, env=None):
        if env:
            self.env=env
        else:
            self.env = PacmanEnv("local")

    def reset(self):
        self.env.reset()

    def restore(self, state):
        self.env.ai_reset(state.gamestate_to_statedict())

    def step(self, pacmanAction, ghostAction, state=None):
        if state:
            self.restore(state)
        return self.env.step(pacmanAction, ghostAction)
    
    def game_state(self):
        return self.env.game_state()

class MCTSNode:
    def __init__(self, env, done=False, parent=None):
        self.env=env
        self.state=env.game_state()
        self.state_dict=self.state.gamestate_to_statedict()
        self.done=done

        self.parent=parent
        self.children_matrix = [[None] * 125 for _ in range(5)]

        self.N=0
        self.P_pacman=np.zeros(5, dtype=np.float32)
        self.P_ghost=np.zeros(125, dtype=np.float32)    # dict: action -> prob
        self.W_pacman=0.0  # accum prob
        self.W_ghost=0.0
        self.Q_pacman=0.0
        self.Q_ghost=0.0   # ave prob

    def is_terminal(self):
        return self.done

    def is_expanded(self):
        return np.any(self.P_pacman) or np.any(self.P_ghost)
    
    def expand(self, pacman, ghost):
        with torch.no_grad():
            action_probs_pacman, value_pacman = pacman.predict(self.state)
            action_probs_ghost, value_ghost = ghost.predict(self.state)

        for action_pacman, prob_pacman in action_probs_pacman:
            self.P_pacman[action_pacman] = prob_pacman
            for action_ghost, prob_ghost in action_probs_ghost:
                self.P_ghost[action_ghost] = prob_ghost

                self.env.restore(self.state)
                _, _, _, done, _  =  self.env.step(action_pacman, ghostact_int2list(action_ghost), self.state)
                self.children_matrix[action_pacman][action_ghost] = MCTSNode(self.env, done, parent=self)

        return value_pacman, value_ghost

    def select(self, c_puct):
        bese_score=-float('inf')
        best_action_pacman=None
        best_action_ghost=None
        best_child=None

        total_visits=self.N if self.N>0 else 1
        
        legal_pacman = np.nonzero(self.P_pacman)[0]
        legal_ghost = np.nonzero(self.P_ghost)[0]
        for action_pacman in legal_pacman:
            for action_ghost in legal_ghost:
                child = self.children_matrix[action_pacman][action_ghost]
                if child is None:
                    continue
                score = (child.Q_pacman + c_puct*(self.P_pacman[action_pacman])*np.sqrt(total_visits)/(1+child.N)
                        + child.Q_ghost + c_puct*(self.P_ghost[action_ghost])*np.sqrt(total_visits)/(1+child.N))
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
        self.W_ghost += value_ghost

        self.Q_pacman = self.W_pacman / self.N
        self.Q_ghost = self.W_ghost / self.N

class MCTS:
    def __init__(self, env, pacman, ghost, c_puct, temperature=1, num_simulations=1600):
        self.env=env

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
        self.root = MCTSNode(self.env)
        for _ in range(self.num_simulations):
            if (_+1)%40 == 0:
                print(f"search {_+1} times")
            self.search(self.root)

        visits_pacman = np.zeros(5, dtype=np.float32)
        visits_ghost = np.zeros(125, dtype=np.float32)
        sum_visits = 0.0

        for action_pacman in range(5):
            for action_ghost in range(125):
                node = self.root.children_matrix[action_pacman][action_ghost]
                if node is not None:
                    visits_pacman[action_pacman] += node.N
                    visits_ghost[action_ghost] += node.N
                    sum_visits += node.N ** self.temp_inverse

        sum_visits = sum_visits if sum_visits != 0 else 1e-8

        prob_pacman = (visits_pacman ** self.temp_inverse) / sum_visits
        prob_ghost = (visits_ghost ** self.temp_inverse) / sum_visits

        selected_action_pacman = np.random.choice(np.arange(5), p=prob_pacman)
        selected_action_ghost = np.random.choice(np.arange(125), p=prob_ghost)

        selected_action_pacman = 0 if self.root.P_pacman[selected_action_pacman] != 0 else 0
        selected_action_ghost = 0 if self.root.P_ghost[selected_action_ghost] != 0 else 0

        decision_pacman = (selected_action_pacman, prob_pacman, self.root.Q_pacman)
        decision_ghost = (selected_action_ghost, prob_ghost, self.root.Q_ghost)

        return decision_pacman, decision_ghost

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
        num_res_blocks=6,
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
    def __init__(self):
        self.ValueNet=ValueNet(if_Pacman=True)
        self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.scaler = GradScaler('cuda')
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
    
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

        act_probs, value = self.ValueNet(state2tensor(state))
        act_probs = act_probs.squeeze()
        value = value.squeeze()
        act_probs = zip(legal_action, act_probs[legal_action])

        return act_probs, value

    def train(self, traj):
        # traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, reward_pacman, reward_ghost))
        # loss = mse + cross_entopy
        for point in traj:
            state, prob_pacman, value_pacman, _, _, _, _, _ = point
            prob_pacman = torch.from_numpy(prob_pacman, device = device)
            with autocast('cuda'):
                prob_pacman_predict, value_pacman_predict = self.predict(state)
                loss_value = (value_pacman, value_pacman_predict)**2
                loss_prob = -sum(prob_pacman[action] * torch.log(prob_pacman_predict[action]) for action in prob_pacman)
                loss = loss_value + loss_prob
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def train_batch(self, traj):
        for i in range(0, len(traj), self.batch_size):
            batch = traj[i:i + self.batch_size]
            states = []
            probs_pacman = []
            values_pacman = []

            with torch.no_grad():
                for point in batch:
                    state, prob_pacman, value_pacman, _, _, _, _, _ = point
                    states.append(state2tensor(state))
                    probs_pacman.append(torch.from_numpy(prob_pacman))
                    values_pacman.append(value_pacman)
                
                state_batch = torch.cat(states)
                prob_pacman = torch.tensor(probs_pacman, dtype=torch.float32, device=device).view(-1, 5)
                value_pacman = torch.tensor(values_pacman, dtype=torch.float32, device=device).view(-1, 1)
            
            with autocast('cuda'):
                prob_pacman_predict, value_pacman_predict = self.ValueNet(state_batch) # 在valuenet中加入squeeze
                loss_value = F.mse_loss(value_pacman_predict, value_pacman)
                loss_prob = F.kl_div(prob_pacman_predict, prob_pacman, reduction='batchmean')
                loss = loss_value + loss_prob

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
class GhostAgent:
    def __init__(self):
        self.ValueNet=ValueNet(if_Pacman=False)
        self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.scaler = GradScaler('cuda')
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
    
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

        act_probs, value = self.ValueNet(state2tensor(state))

        act_probs = act_probs.squeeze()
        value = value.squeeze()

        act_probs = zip(legal_actions, act_probs[legal_actions])

        value_float = value.float()

        return act_probs, value_float

    def train(self, traj):
        for point in traj:
            state, _, _, prob_ghost, value_ghost, _, _, _ = point
            prob_ghost = torch.from_numpy(prob_ghost, device = device)
            with autocast('cuda'):
                prob_ghost_predict, value_ghost_predict = self.predict(state)
                loss_value = (value_ghost - value_ghost_predict)**2
                loss_prob = -sum(prob_ghost[action] * np.log(prob_ghost_predict[action]) for action in prob_ghost)
                loss = loss_value + loss_prob   
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def train_batch(self, traj):
        for i in range(0, len(traj), self.batch_size):
            batch = traj[i:i + self.batch_size]
            states = []
            probs_ghost = []
            values_ghost = []

            with torch.no_grad():
                for point in batch:
                    state, prob_ghost, value_ghost, _, _, _, _, _ = point
                    states.append(state2tensor(state))
                    probs_ghost.append(torch.from_numpy(prob_ghost))
                    values_ghost.append(value_ghost)
                
                state_batch = torch.cat(states)
                probs_ghost = torch.tensor(probs_ghost, dtype=torch.float32, device=device).view(-1, 15)
                values_ghost = torch.tensor(values_ghost, dtype=torch.float32, device=device).view(-1, 1)
            
            with autocast('cuda'):
                prob_ghost_predict, value_ghost_predict = self.ValueNet(state_batch) # 在valuenet中加入squeeze
                loss_value = F.mse_loss(value_ghost_predict, values_ghost)
                loss_prob = F.kl_div(prob_ghost_predict, probs_ghost, reduction='batchmean')
                loss = loss_value + loss_prob

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

class AlphaZeroTrainer:
    def __init__(self, env, pacman, ghost, c_puct, iterations=100, episodes=10, check_time=10, search_time=1600):
        self.env=env

        self.pacman=pacman
        self.ghost=ghost
        self.c_puct=c_puct
        self.MCTS=MCTS(self.env, self.pacman, self.ghost, self.c_puct, num_simulations=search_time)

        self.iterations=iterations
        self.episodes=episodes
        self.check_time=check_time
        # self.search_time=search_time

        self.best_score=0.0

    def decide(self):
        return self.MCTS.run()

    def play(self):
        traj=[]
        reward_pacman=0.0
        reward_ghost=0.0
        while True:
            decision_pacman, decision_ghost = self.decide()
            selected_action_pacman, action_prob_pacman, value_pacman = decision_pacman
            selected_action_ghost, action_prob_ghost, value_ghost = decision_ghost
            _, reward_pacman, reward_ghost, done, eatAll = self.env.step(selected_action_pacman, ghostact_int2list(selected_action_ghost))
            state=self.env.game_state()
            traj.append((state, action_prob_pacman, value_pacman, action_prob_ghost, value_ghost, reward_pacman, reward_ghost))
            if done:
                print("game end")
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
            
            print(f"Iteration: {ite}/{self.iterations}, loss_pacman = {loss_pacman}, loss_ghost = {loss_ghost}, score_pacman = {score_pacman}, score_ghost = {score_ghost}")

            if(score_pacman+score_ghost>self.best_score):
                print(f"NEW  BEST with score {score_pacman+score_ghost}")
                self.pacman.save_model()
                self.ghost.save_model()

if __name__ == "__main__":
    from core.GymEnvironment import *
    env=PacmanEnvDecorator()
    env.reset()

    pacman = PacmanAgent()
    ghost = GhostAgent()

    action1, value1 = pacman.predict(env.game_state())
    action2, value2 = ghost.predict(env.game_state())

    print(action1, value1, action2, value2)

    env.reset()
    mcts = MCTS(env=env, pacman=pacman, ghost=ghost, c_puct=1.25)
    print(mcts.run())