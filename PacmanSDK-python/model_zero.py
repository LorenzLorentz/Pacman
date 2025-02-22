import os
import random
import copy
from collections import deque
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
    def __init__(self, env, gamestate, state, extra, done, player, parent=None, prior_prob=0):
        self.gamestate = gamestate
        self.env = env
        self.state = state
        self.extra = extra
        self.done = done
        self.player = player
        self.parent = parent
        self.children = {} # a map from action to treenode
        self.N = 0      # cnt of visiting
        self.W = 0.0    # accum value
        self.Q = 0.0    # avg value
        self.P = prior_prob # prior prob
    
    def expand(self, action_prob):
        for action, prob in action_prob:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

    def select(self, gamma):
        return max(self.children.items(), key = 0) # key = act_node: act_node[1].get_value(gamma))
    
    def update_step(self, leaf_value):
        self.visits += 1
        self.value += (leaf_value - self.value)/self.visits

    def update(self, leaf_value):
        if self.parent:
            self.parent.update(-leaf_value)
        self.update_step(leaf_value)

    def ucb_score(self, c_puct):
        return self.Q + (c_puct * self.P *  np.sqrt(self.parent.N)/(1 + self.N))
    
    def if_leaf(self):
        return len(self.is_leaf) == 0
    
    def if_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, pacman_net, ghost_net, env, root_state, root_extra, c_puct=1.0, num_simulations=10000):
        self.env = env
        self.state = root_state
        self.extra = root_extra

        self.root = MCTSNode(state=root_state, player="pacman")
        self.pacman_net = pacman_net
        self.ghost_net = ghost_net

        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def palyout(self, env, state, extra):
        node = self.root
        while(1):
            if node.if_leaf():
                break
            action, node = node.select(self.gamma)
            next_state, reward1, reward2, done, _ = env.step(action)
            next_state, next_extra = state_dict_to_tensor(next_state)
            state = next_state
            extra = next_extra
        action_probs, leaf_value = self.policy(state, extra)
        if not done:
            node.expand(action_probs)
        node.update(-reward1)

    def get_move_probs(self, env, state, extra, temp=1e-3):
        for n in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            self.palyout(env_copy, state, extra)
        
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = np.softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self._root = MCTSNode(None, 1.0)

    def search(self):
        self.expand_node(self.root)

        for _ in range(self.num_simulations):
            node = self.root
            search_path = [node]

            while not node.if_leaf():
                best_score = -float('inf')
                best_jointaction = None
                best_child = None
                for jointaction, child in node.children.items():
                    score = child.ucb_score(self.c_puct)
                    if score > best_score:
                        best_score = score
                        best_jointaction = jointaction
                        best_child = child
                node = best_child
                search_path.append(node)

            if node.done:
                value = node.gamestate.reward1
            else:
                self.expand_node(node)
                _, value = self.pacman_net.predict(node.state)

            self.backpropagate(search_path, value, node.player)
        
        joint_counts = {joint: child.N for joint, child in self.root.children.items()}

        pacman_counts = {}
        ghost_counts = {}
        for (lp, lg), count in joint_counts.items():
            pacman_counts[lp] += count
            ghost_counts[lg] += count

        total_p = np.sum(pacman_counts.values())
        total_g = np.sum(ghost_counts.values())
        pacman_policy = {a: count / total_p for a, count in pacman_counts.items()}
        ghost_policy = {a: count / total_g for a, count in ghost_counts.items()}

        return pacman_policy, ghost_policy

    def expand_node(self, node):
        if node.done:
            return

        legal_actions_pacman = get_valid_moves_pacman(node.gamestate.pacman_pos, node.gamestate)
        legal_actions_ghost = get_valid_moves_ghost(node.gamestate.ghost_pos, node.gamestate)

        pacman_policy, _ = self.pacman_net(node.state, node.extra)
        ghost_policy, _ = self.ghost_net(node.state, node.extra)

        joint_prior = {}
        for lp in legal_actions_pacman:
            for lg in legal_actions_ghost:
                p_p = pacman_policy.get(lp, 0)
                p_g = ghost_policy.get(lg, 0)
                joint_prior[(lp, lg)] = p_p * p_g

        total = sum(joint_prior.values())
        if total > 0:
            for joint_action in joint_prior:
                joint_prior[joint_action] /= total
        else:
            num = len(joint_prior)
            for joint_action in joint_prior:
                joint_prior[joint_action] = 1/num

        for joint_action, prior in joint_prior.items():
            lp, lg = joint_action
            next_state, next_extra = state_dict_to_tensor(node.env.step(lp, lg))
            node.children[joint_action] = MCTSNode(node.env, node.gamestate, next_state, next_extra, 'pacman', node, prior) # 修改节点api
    
    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N

    def search(self):
        # 递归调用
        policy_prob = None
        value = None
        return policy_prob, value

# def search:
#   # 递归函数
#   if done: return v
#   if policy is empty:
#       action_prob, value = net(state)
#       self.policy = softmmax(action_prob.correct())
#       return Vs
#       # 此后策略不会变化，但是价值会更新
#   else:
#       # PUCT
#       # find max(\lambda \pi + self.q) # 初始q置零
#       v = search(max)
#       self.count + 1
#       self.q += (v-q)/n # 增量更新
#       return v

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
            policy_out_dim=15

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

        log_act_probs, value = self.ValueNet(state)
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_action, act_probs[legal_action])

        value_float = value.float()

        # act_probs = [] (length = 5)
        # zip ( * , * )

        return act_probs, value_float

    def train(self, traj):
        # traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, reward_pacman, reward_ghost))
        # loss = mse + cross_rntopy
        for point in traj:
            state, prob_pacman, value_pacman, _, _, _, _, _ = traj
            prob_pacman_predict, value_pacman_predict = self.predict(state)
            loss_value = nn.MSELoss(value_pacman, value_pacman_predict)
            loss_prob = nn.CrossEntropyLoss(prob_pacman, prob_pacman_predict)
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
        legal_actions = get_valid_moves_pacman(pos, state)

        log_act_probs, value = self.ValueNet(state)

        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs_new = {}
        for action, index in enum(legal_actions):
            act_probs_new[action]=[]
            act_probs_new[action].append(act_probs[index*3])
            act_probs_new[action].append(act_probs[index*3+1])
            act_probs_new[action].append(act_probs[index*3+2])
        act_probs_new = zip(legal_actions, act_probs_new[legal_actions])

        # act_probs_new = {action=[ , , ]: [ , , ]}
        # zip ([ , , ], [ , , ])

        value_float = value.float()

        return act_probs_new, value_float

    def train(self, traj):
        for point in traj:
            state, _, _, prob_ghost, value_ghost, _, _, _ = traj
            prob_ghost_predict, value_ghost_predict = self.predict(state)
            loss_value = nn.MSELoss(value_ghost, value_ghost_predict)
            loss_prob = nn.CrossEntropyLoss(prob_ghost, prob_ghost_predict)
            loss = loss_value + loss_prob
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

"""
    def predict(self, state):
        pos = state.gamestate_to_statedict()["pacman_coord"]
        legal_action = get_valid_moves_pacman(pos, state)

        log_act_probs, value = self.ValueNet(state)
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_action, act_probs[legal_action])

        value_float = value.float()

        # act_probs = [] (length = 5)
        # zip ( * , * )

        return act_probs, value_float

    def predict(self, state):
        pos = state.gamestate_to_statedict()["ghosts_coord"]
        legal_actions = get_valid_moves_pacman(pos, state)

        log_act_probs, value = self.ValueNet(state)

        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs_new = {}
        for action, index in enum(legal_actions):
            act_probs_new[action]=[]
            act_probs_new[action].append(act_probs[index*3])
            act_probs_new[action].append(act_probs[index*3+1])
            act_probs_new[action].append(act_probs[index*3+2])
        act_probs_new = zip(legal_actions, act_probs_new[legal_actions])

        # act_probs_new = {action=[ , , ]: [ , , ]}
        # zip ([ , , ], [ , , ])

        value_float = value.float()

        return act_probs_new, value_float

    def _train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer_pacman.zero_grad()

        log_act_probs, value = self.policy_value_net_pacman(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer_pacman.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        return loss.item(), entropy.item()
"""

# model: MCTS && AlphaZero
# Class: MCTSNode, MCTS, ValueNet, Trainer(self_play)

# 回合制MCTS, 搜索到回合结束, 开始倾向于广度搜索, 后面倾向于深度搜索
# 通过深度残差网络给出q和pi
# 探索次数说明有价值

# 网络结构, 公共头, 策略头, 价值头
# 网络就是参数 \theta

# agent = Agent()
# for iteration:
#   trajs=[]
#   for episode: 获取经验
#       traj.append(play())
#   learn() 更新参数theta
#   play()
#
# def play:
#   traj = []
#   while True:
#       action, prob_distri = agent.decide
#       step(action)
#       if done: break
#       (p, state, bonus, who)
#
# def decide:
#   agent.search() # 多次MCTS
#   p = count[action]/sum(count[action])
#   a = random_choice(p)
#   return (a, p)
#
# def search:
#   # 递归函数
#   if done: return v
#   if policy is empty:
#       action_prob, value = net(state)
#       self.policy = softmmax(action_prob.correct())
#       return Vs
#       # 此后策略不会变化，但是价值会更新
#   else:
#       # PUCT
#       # find max(\lambda \pi + self.q) # 初始q置零
#       v = search(max)
#       self.count + 1
#       self.q += (v-q)/n # 增量更新
#       return v
#
# def learn:
#   for batches:
#       net_fit(traj) # 训练theta
#   reset() # 只保留theta

class AlphaZeroTrainer:
    def __init__(self, env, pacman, ghost, MCTS, iterations=100, episodes=100, check_time=100, search_time=100):
        self.env=env

        self.pacman=pacman
        self.ghost=ghost
        self.MCTS=MCTS

        self.iterations=iterations
        self.episodes=episodes
        self.check_time=check_time
        self.search_time=search_time

        self.best_score=0.0

    def pacman_decide(self):
        state = self.env.game_state()
        count = self.MCTS.search()
        action = get_valid_moves_pacman(state.gamestate_to_statedict()["pacman_coord"])
        p = count[action]/sum(count[action])
        a = np.multiply(p)
        return (a, p)

    def ghost_decide(self):
        state = self.env.game_state()
        count = self.MCTS.search()
        action = get_valid_moves_pacman(state.gamestate_to_statedict()["pacman_coord"])
        p = count[action]/sum(count[action])
        a = np.multiply(p)
        return (a, p)

    def play(self):
        traj=[]
        while True:
            action_pacman, prob_pacman, value_pacman = self.pacman_decide()
            action_ghost, prob_ghost, value_ghost = self.ghost_decide()
            _, reward_pacman, reward_ghost, done, eatenALl = self.env.step(action_pacman, action_ghost)
            state=self.env.game_state()
            traj.append((state, prob_pacman, value_pacman, prob_ghost, value_ghost, reward_pacman, reward_ghost))
            if done:
                break
        return traj

    def learn(self, trajs):
        for traj in trajs:
            self.pacman.train(traj)
            self.ghost.train(traj) # 可以向量化，批量训练
        self.env.reset()

    def train(self):
        for _ in range(self.iterations):
            trajs=[]
            for __ in range(self.episodes):
                trajs.append(self.play)
            self.learn(trajs)

            score=0.0
            for __ in range(self.check_time):
                score+=self.play()
            score/=self.check_time
            if(score>self.best_score):
                print(f"NEW  BEST with acc {score}")
                self.pacman.save_model()
                self.ghost.save_model()

# def search:
#   # 递归函数
#   if done: return v
#   if policy is empty:
#       action_prob, value = net(state)
#       self.policy = softmmax(action_prob.correct())
#       return Vs
#       # 此后策略不会变化，但是价值会更新
#   else:
#       # PUCT
#       # find max(\lambda \pi + self.q) # 初始q置零
#       v = search(max)
#       self.count + 1
#       self.q += (v-q)/n # 增量更新
#       return v

class MCTSNode:
    def __init__(self, env, done, parent):
        self.state=env.game_state()
        self.env=copy.deepcopy(env)
        self.state_dict=self.state.gamestate_to_statedict()
        self.done=done

        self.parent=parent
        self.children={}   # dict: action -> child

        self.N=0
        self.P={}          # dict: action -> prob
        self.W=0.0         # accum prob
        self.Q=0.0         # ave prob

    def is_terminal(self):
        return self.done

    def is_expanded(self):
        return len(self.P)>0
    
    def expand(self, pacman, ghost):
        """
        使用神经网络扩展当前节点：
          net(state) 返回 (action_logits, value)
          - action_logits: dict，键为动作，值为对应的原始分数
          - value: 对当前状态的评估值（由网络直接给出）
        """
        action_logits_pacman, value_pacman = pacman(self.state)
        action_logits_ghost, value_ghost = ghost(self.state)

        actions_pacman=list(action_logits_pacman.keys())
        actions_ghost=list(action_logits_ghost.keys())

        logits_pacman = np.array([action_logits_pacman[a] for a in actions_pacman])
        logits_ghost = np.array([action_logits_ghost[a] for a in actions_ghost])

        probs_pacman = torch.softmax(probs_pacman)
        probs_ghost = torch.softmax(probs_ghost)

        # self.P = {a: p for a, p in zip(actions, probs)}

        for action_pacman in actions_pacman:
            for action_ghost in actions_ghost:
                if (action_pacman, action_ghost) not in self.children:
                    _, done, _  =  self.env.step(action_pacman, action_ghost)
                    self.children[(action_pacman, action_ghost)] = MCTSNode(self.env, done, parent=self)

        return value_pacman, value_ghost

    def select(self, c_puct):
        # score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1+N(s,a))
        bese_score=-float('inf')
        best_action=None
        best_child=None

        total_visits=self.N if self.N>0 else 1
        for action, child in self.children.items():
            score = child.Q + c_puct*self.P[action]*np.sqrt(total_visits)/(1+child.N)
            if score>bese_score:
                bese_score=score
                best_action=action
                best_child=child
        
        return best_action, best_child
    
    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

class MCTS:
    def __init__(self, env, state, pacman, ghost, c_puct, temperature=1, num_simulations=1600):
        self.env=copy.deepcopy(env)
        self.state=state

        self.pacman=pacman
        self.ghost=ghost
        self.c_puct=c_puct
        self.temp_inverse=1/temperature
        self.num_simulations=num_simulations

    def search(self, node):
        if node.is_terminal:
            return node.state_dict["score"]
        
        if not node.is_expanded:
            value=node.expand(self.pacman, self.ghost)
            node.update(value)
            return value
        
        action, child=node.select(self.c_puct)

        value=self.search(child)
        node.update(value)
        return value
    
    def run(self):
        self.root = MCTSNode(self.env, self.state)
        for _ in range(self.num_simulations):
            self.search(self.root)

        action_prob={}
        sum_visits=0
        for action, child in self.root.children:
            action_prob[action]=child.N**self.temp_inverse
            sum_visits+=child.N**self.temp_inverse
        action_prob/=sum_visits

        return max(action_prob) # 根据***抽取