import os
import sys
import time
import datetime
import copy
import math
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from core.gamedata import *
from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.data_process import *
from utils.ghostact_int2list import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_short = datetime.datetime.now().strftime('%H%M')
time_long = datetime.datetime.now().strftime('%m%d%H%M')
# sys.stdout = open(f'log_zero/output_{time_short}.log', 'w')

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

""""
class MCTSNode:
    def __init__(self, env, done=False, parent=None):
        self.env=copy.deepcopy(env)
        self.state=env.game_state()
        self.state_dict=self.state.gamestate_to_statedict()
        self.done=done

        self.parent=parent
        self.children_matrix = [[None] * 125 for _ in range(5)]

        self.N=0
        self.P_pacman=np.zeros(5, dtype=np.float32)
        self.P_ghost=np.zeros(125, dtype=np.float32)
        self.W_pacman=0.0
        self.W_ghost=0.0
        self.Q_pacman=0.0
        self.Q_ghost=0.0
        
        self.expanded=False

    def is_terminal(self):
        return self.done

    def is_expanded(self):
        return self.expanded
    
    def expand(self, pacman, ghost):
        self.expanded=True

        with torch.no_grad():
            action_probs_pacman, value_pacman = pacman.predict(self.state)
            action_probs_ghost, value_ghost = ghost.predict(self.state)

        self.P_pacman = action_probs_pacman.cpu().numpy()
        self.P_ghost = action_probs_ghost.cpu().numpy()

        pos_pacman = self.state.gamestate_to_statedict()["pacman_coord"]
        legal_action_pacman = get_valid_moves_pacman(pos_pacman, self.state)
        pos_ghost = self.state.gamestate_to_statedict()["ghosts_coord"]
        legal_action_ghost = get_valid_moves_ghost(pos_ghost, self.state)

        for action_pacman in legal_action_pacman:
            for action_ghost in legal_action_ghost:
                self.env.restore(self.state)
                _, _, _, done, _ = self.env.step(action_pacman, ghostact_int2list(action_ghost), self.state)
                self.children_matrix[action_pacman][action_ghost] = MCTSNode(self.env, done, parent=self)

        return value_pacman.item(), value_ghost.item()

    def select(self, c_puct):
        indices=[]
        q_pacman_list=[]
        q_ghost_list=[]
        n_list=[]
        p_pacman_list=[]
        p_ghost_list=[]

        pos_pacman = self.state.gamestate_to_statedict()["pacman_coord"]
        legal_action_pacman = get_valid_moves_pacman(pos_pacman, self.state)
        pos_ghost = self.state.gamestate_to_statedict()["ghosts_coord"]
        legal_action_ghost = get_valid_moves_ghost(pos_ghost, self.state)

        for action_pacman in legal_action_pacman:
            for action_ghost in legal_action_ghost:
                child = self.children_matrix[action_pacman][action_ghost]
                if child is not None:
                    indices.append((action_pacman, action_ghost))
                    q_pacman_list.append(child.Q_pacman)
                    q_ghost_list.append(child.Q_ghost)
                    n_list.append(child.N)
                    p_pacman_list.append(self.P_pacman[action_pacman])
                    p_ghost_list.append(self.P_ghost[action_ghost])

        indices = np.array(indices)
        q_pacman_arr = np.array(q_pacman_list)
        q_ghost_arr = np.array(q_ghost_list)
        n_arr = np.array(n_list)
        p_pacman_arr = np.array(p_pacman_list)
        p_ghost_arr = np.array(p_ghost_list)

        total_visits = self.N if self.N > 0 else 1
        bonus = c_puct * np.sqrt(total_visits) / (1 + n_arr)
        scores = q_pacman_arr + bonus * p_pacman_arr + q_ghost_arr + bonus * p_ghost_arr
        
        best_idx = np.argmax(scores)
        best_action_pacman, best_action_ghost = indices[best_idx]
        best_child = self.children_matrix[best_action_pacman][best_action_ghost]

        return best_action_pacman, best_action_ghost, best_child
    
    def update(self, value):
        value_pacman, value_ghost = value

        self.N += 1
        self.W_pacman += value_pacman
        self.W_ghost += value_ghost

        self.Q_pacman = self.W_pacman / self.N
        self.Q_ghost = self.W_ghost / self.N

class MCTS:
    def __init__(self, env, pacman, ghost, c_puct, temperature=1, num_simulations=120):
        self.env=env
        self.state=env.game_state()

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
    
    def run_batch(self, batch_size=8):
        self.root = MCTSNode(self.env)
        for _ in range(self.num_simulations // batch_size):
            leaf_nodes = []
            paths = []
            
            for _ in range(batch_size):
                node = self.root
                path = []
                while not node.is_terminal() and node.is_expanded():
                    action_pacman, action_ghost, child = node.select(self.c_puct)
                    path.append((node, action_pacman, action_ghost))
                    node = child
                leaf_nodes.append(node)
                paths.append(path)
            
            states = [node.state for node in leaf_nodes]
            state_tensors = [state2tensor(state) for state in states]
            state_batch = torch.cat(state_tensors)
            
            with torch.no_grad():
                action_probs_pacman, values_pacman = self.pacman.ValueNet(state_batch)
                action_probs_ghost, values_ghost = self.ghost.ValueNet(state_batch)
            
            for i, node in enumerate(leaf_nodes):
                if node.is_terminal():
                    value = node.state_dict["score"]
                else:
                    self.P_pacman = action_probs_pacman[i].cpu().numpy()
                    self.P_ghost = action_probs_ghost[i].cpu().numpy()
                    value =  (values_pacman[i].item(), values_ghost[i].item())

                    pos_pacman = node.state.gamestate_to_statedict()["pacman_coord"]
                    legal_action_pacman = get_valid_moves_pacman(pos_pacman, node.state)
                    pos_ghost = node.state.gamestate_to_statedict()["ghosts_coord"]
                    legal_action_ghost = get_valid_moves_ghost(pos_ghost, node.state)

                    for action_pacman in legal_action_pacman:
                        for action_ghost in legal_action_ghost:
                            node.env.restore(self.state)
                            _, _, _, done, _ = node.env.step(action_pacman, ghostact_int2list(action_ghost), node.state)
                            node.children_matrix[action_pacman][action_ghost] = MCTSNode(node.env, done, parent=self)
                
                for parent_node, _, _ in paths[i][::-1]:
                    parent_node.update(value)
                if not node.is_terminal():
                    node.update(value)
        return self.decide()

    def run(self):
        self.root = MCTSNode(self.env)
        for _ in range(self.num_simulations):
            self.search(self.root)
        return self.decide()

    def play_game_pacman(self):
        self.root = MCTSNode(self.env)
        for _ in range(self.num_simulations):
            self.search(self.root)
        visits_pacman = torch.zeros(5, dtype=torch.float32, device=device)
        visits_ghost = torch.zeros(125, dtype=torch.float32, device=device)
        sum_visits = 0.0
        
        pos_pacman = self.state.gamestate_to_statedict()["pacman_coord"]
        legal_action_pacman = get_valid_moves_pacman(pos_pacman, self.state)
        pos_ghost = self.state.gamestate_to_statedict()["ghosts_coord"]
        legal_action_ghost = get_valid_moves_ghost(pos_ghost, self.state)

        for action_pacman in legal_action_pacman:
            for action_ghost in legal_action_ghost:
                node = self.root.children_matrix[action_pacman][action_ghost]
                if node is not None:
                    visits_pacman[action_pacman] += node.N
                    visits_ghost[action_ghost] += node.N
                    sum_visits += node.N ** self.temp_inverse

        sum_visits = sum_visits if sum_visits != 0.0 else 1e-8

        prob_pacman = (visits_pacman ** self.temp_inverse) / sum_visits
        selected_action_pacman = torch.multinomial(prob_pacman, 1) #torch.random.choice(torch.arange(5), p=prob_pacman)
        return selected_action_pacman.cpu().numpy().tolist() if self.root.P_pacman[selected_action_pacman.item()] != 0 else [0]

    def decide(self):
        visits_pacman = torch.zeros(5, dtype=torch.float32, device=device)
        visits_ghost = torch.zeros(125, dtype=torch.float32, device=device)
        sum_visits = 0.0
        
        pos_pacman = self.state.gamestate_to_statedict()["pacman_coord"]
        legal_action_pacman = get_valid_moves_pacman(pos_pacman, self.state)
        pos_ghost = self.state.gamestate_to_statedict()["ghosts_coord"]
        legal_action_ghost = get_valid_moves_ghost(pos_ghost, self.state)

        for action_pacman in legal_action_pacman:
            for action_ghost in legal_action_ghost:
                node = self.root.children_matrix[action_pacman][action_ghost]
                if node is not None:
                    visits_pacman[action_pacman] += node.N
                    visits_ghost[action_ghost] += node.N
                    sum_visits += node.N ** self.temp_inverse

        sum_visits = sum_visits if sum_visits != 0.0 else 1e-8

        prob_pacman = (visits_pacman ** self.temp_inverse) / sum_visits
        prob_ghost = (visits_ghost ** self.temp_inverse) / sum_visits

        selected_action_pacman = torch.multinomial(prob_pacman, 1) #torch.random.choice(torch.arange(5), p=prob_pacman)
        selected_action_ghost = torch.multinomial(prob_ghost, 1) # torch.random.choice(torch.arange(125), p=prob_ghost)

        # print("!!!", "0", sum_visits)
        # print("!!!!", "1", prob_pacman.shape, prob_pacman, selected_action_pacman)
        # print("!!!!", "2", prob_ghost.shape, prob_ghost, selected_action_ghost)

        selected_action_pacman = selected_action_pacman.item() if self.root.P_pacman[selected_action_pacman.item()] != 0 else 0
        selected_action_ghost = selected_action_ghost.item() if self.root.P_ghost[selected_action_ghost.item()] != 0 else 0

        decision_pacman = (selected_action_pacman, prob_pacman, torch.tensor(self.root.Q_pacman, dtype=torch.float32, device=device))
        decision_ghost = (selected_action_ghost, prob_ghost, torch.tensor(self.root.Q_ghost, dtype=torch.float32, device=device))

        return decision_pacman, decision_ghost
"""

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
    def __init__(self, batch_size=32, series=None):
        self.series=series
        self.ValueNet=ValueNet(if_Pacman=True)
        self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.scaler = GradScaler('cuda')
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
        self.batch_size=batch_size
    
    def init_weight(self, model):
        if(self.series):
            name = f"model/pacman_zero_{self.series}.pth"
            if os.path.exists(name):
                model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
        else:
            print("No checkpoint found. Training from scratch.")
            model.init_weights()

    def save_model(self, name=f"model/pacman_zero_{time_long}.pth"):
        torch.save(self.ValueNet.state_dict(), name)
    
    def predict(self, state):
        pos = state.gamestate_to_statedict()["pacman_coord"]
        legal_action = get_valid_moves_pacman(pos, state)

        act_probs, value = self.ValueNet(state2tensor(state))
        act_probs = act_probs.squeeze()
        value = value.squeeze()

        result=torch.zeros_like(act_probs)
        result[legal_action]=act_probs[legal_action]

        return result, value

    def train(self, traj):
        # traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, reward_pacman, reward_ghost))
        # loss = mse + cross_entopy
        for point in traj:
            state, prob_pacman, value_pacman, _, _, _, _ = point
            # prob_pacman = torch.from_numpy(prob_pacman, device = device)
            with autocast('cuda'):
                prob_pacman_predict, value_pacman_predict = self.predict(state)
                loss_value = (value_pacman - value_pacman_predict)**2
                loss_prob = -sum(prob_pacman[action] * torch.log(prob_pacman_predict[action]) for action in range(5))
                loss = loss_value + loss_prob
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def train_batch(self, traj):
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
                # prob_pacman = torch.tensor(probs_pacman, dtype=torch.float32, device=device).view(-1, 5)
                # value_pacman = torch.tensor(values_pacman, dtype=torch.float32, device=device).view(-1, 1)
            
            with autocast('cuda'):
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
    def __init__(self, batch_size=32, series=None):
        self.series=series
        self.ValueNet=ValueNet(if_Pacman=False)
        self.optimizer=optim.Adam(self.ValueNet.parameters())
        self.scaler = GradScaler('cuda')
        self.init_weight(self.ValueNet)
        self.ValueNet.to(device)
        self.batch_size=batch_size
    
    def init_weight(self, model):
        if self.series:
            name = f"model/ghost_zero_{self.series}.pth"
            if os.path.exists(name):
                model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
        else:
            print("No checkpoint found. Training from scratch.")
            model.init_weights()

    def save_model(self, name=f"model/ghost_zero_{time_long}.pth"):
        torch.save(self.ValueNet.state_dict(), name)

    def predict(self, state):
        pos = state.gamestate_to_statedict()["ghosts_coord"]
        legal_actions = get_valid_moves_ghost(pos, state)

        act_probs, value = self.ValueNet(state2tensor(state))

        act_probs = act_probs.squeeze()
        value = value.squeeze()

        result=torch.zeros_like(act_probs)
        result[legal_actions]=act_probs[legal_actions]

        return act_probs, value

    def train(self, traj):
        for point in traj:
            state, _, _, prob_ghost, value_ghost, _, _ = point
            # prob_ghost = torch.from_numpy(prob_ghost, device = device)
            with autocast('cuda'):
                prob_ghost_predict, value_ghost_predict = self.predict(state)
                loss_value = (value_ghost - value_ghost_predict)**2
                loss_prob = -sum(prob_ghost[action] * torch.log(prob_ghost_predict[action]) for action in range(15))
                loss = loss_value + loss_prob   
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def train_batch(self, traj, batch_size=32):
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

                # prob_ghost = torch.tensor(probs_ghost, dtype=torch.float32, device=device).view(-1, 125)
                # value_ghost = torch.tensor(values_ghost, dtype=torch.float32, device=device).view(-1, 1)
            
            with autocast('cuda'):
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

class AlphaZeroTrainer:
    def __init__(self, env, pacman, ghost, c_puct, iterations=10, episodes=32, check_time=5, search_time=32):
        self.env=env

        self.pacman=pacman
        self.ghost=ghost
        self.c_puct=c_puct
        # self.MCTS=MCTS(self.env, self.pacman, self.ghost, self.c_puct, num_simulations=search_time)

        self.iterations=iterations
        self.episodes=episodes
        self.check_time=check_time
        self.search_time=search_time

        self.best_score=0.0

    def decide(self):
        mcts=mcts_module.MCTS(self.env, self.pacman, self.ghost, self.c_puct, num_simulations=self.search_time)
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
            # step+=1
            # if(step%10==0):
            #    print("step: {}, round: {}".format(step, dict["round"]))
            
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

if __name__ == "__main__":
    env=PacmanEnvDecorator()
    env.reset(mode="local")
    pacman = PacmanAgent()
    ghost = GhostAgent()

    print("TEST of net predict")
    t=time.time()
    action1, value1 = pacman.predict(env.game_state())
    action2, value2 = ghost.predict(env.game_state())
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of mcts")
    print("Running search:")
    env.reset()
    t=time.time()
    mcts = mcts_module.MCTS(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, num_simulations=16)
    mcts.run()
    t=time.time()-t
    print(f"time:{t}")
    print("Running batch:")
    env.reset()
    t=time.time()
    mcts = mcts_module.MCTS(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, num_simulations=16)
    # mcts.run_batch()
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of self play")
    SEARCH_TIME=16
    env.reset()
    t=time.time()
    trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, search_time=SEARCH_TIME)
    # trainer.play()
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of agent train")
    traj = []
    for _ in range(78):
        state=env.game_state()
        env.reset(mode="local")

        prob_pacman = torch.rand(5).to(device)
        # prob_pacman = np.random.rand(5)
        prob_pacman = prob_pacman / prob_pacman.sum()
        value_pacman = torch.rand(1).to(device)
        #value_pacman = np.random.rand(1)

        prob_ghost = torch.rand(125).to(device)
        # prob_ghost = np.random.rand(125)
        prob_ghost = prob_ghost / prob_ghost.sum()
        value_ghost = torch.rand(1).to(device)
        # value_ghost = np.random.rand(1)

        dummy1 = 0.0
        dummy2 = 0.0
        
        traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, dummy1, dummy2))
    
    print("Running train:")
    t=time.time()
    pacman.train(traj)
    ghost.train(traj)
    t=time.time()-t
    print(f"time:{t}")
    print("Running batch:")
    t=time.time()
    pacman.train_batch(traj)
    ghost.train_batch(traj)
    t=time.time()-t
    print(f"time:{t}")

    print("TEST of alphazero train")
    SEARCH_TIME=5
    env.reset()
    t=time.time()
    trainer = AlphaZeroTrainer(env=env, pacman=pacman, ghost=ghost, c_puct=1.25, iterations=1, episodes=3, check_time=2, search_time=SEARCH_TIME)
    trainer.train()
    t=time.time()-t
    print(f"time:{t}")