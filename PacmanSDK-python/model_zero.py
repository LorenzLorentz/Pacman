import os
import random
import copy
from collections import deque
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

class AIState(Enum):
    COLLECT = "COLLECT"
    ESCAPE = "ESCAPE"
    BONUS = "BONUS"
    GETOUT = "GETOUT"

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
    
class ValueNet:
    def __init__(self, input_channel_num, extra_size, num_actions, ifGhost):
        super().__init__()
        self.channels = input_channel_num
        self.embedding_dim = 32
        self.shared_embedding = nn.Embedding(input_channel_num * 10, self.embedding_dim)

        self.conv1 = nn.Conv2d(self.channels * self.embedding_dim, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.attention = nn.Sequential(
            nn.Conv2d(512, 512//8, 1),
            nn.ReLU(),
            nn.Conv2d(512//8, 512, 1),
            nn.Sigmoid()
        )

        self.encoder= nn.Sequential(
            nn.Linear(extra_size, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.act = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU,
            nn.Linear(256, 128),
            nn.ReLU,
            nn.Linear(128, num_actions + 2 * num_actions * ifGhost)
        )

        self.val = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU,
            nn.Linear(256, 128),
            nn.ReLU,
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        offsets = torch.arange(C, device=device).view(1, C, 1, 1) * 10
        x_offset = x.long() + offsets
        x_offset = x_offset.view(B, -1)
        embedded = self.shared_embedding(x_offset)
        embedded = embedded.view(B, C, H, W, self.embedding_dim)
        x_embedded = embedded.permute(0, 4, 1, 2, 3).reshape(B, C * self.embedding_dim, H, W)

        x_out = F.relu(self.conv1(x_embedded))
        x_out = F.relu(self.conv2(x_out))
        x_out = self.bn1(x_out)
        x_out = F.relu(self.conv3(x_out))
        x_out = self.conv4(x_out)
        
        att = self.attention(x_out)
        x_out = x_out * att

        x_out = self.adaptive_pool(x_out).view(B, -1)
        x_out = F.normalize(self.fc(x_out), p=2, dim=1)

        y_encoded = torch.sigmoid(self.encoder(y))

        features = x_out + y_encoded
        
        act = self.act(features)
        val = self.val(features)

        return act, val

class PolicyValueNet:
    def __init__(self, input_channel_num, extra_size, num_actions):
        self.policy_value_net_pacman = ValueNet(input_channel_num, extra_size, num_actions, False)
        self.policy_value_net_ghost = ValueNet(input_channel_num, extra_size, num_actions, True)
        self.optimizer_pacman = optim.Adam(self.policy_value_net_pacman.parameters())

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.uniform_(m.weight, -0.05, 0.05)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def init_network(self, network, name):
        if os.path.exists(name):
            network.load_state_dict(torch.load(name, map_location=device, weights_only=True))
        else:
            print("No checkpoint found. Training from scratch.")
            network.apply(self.init_weights)
        
    def save_model(self, network, name):
        torch.save(network.state_dict(), name)

    def predict(self, env, state, extra):
        pos = env.get_return_dict["pacman_coord"]
        legal_positions = self.get_valid_moves(pos, state)
        log_act_probs, value = self.policy_value_net_pacman(state, extra)
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value
    
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
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
    
class AlphaZeroTrainer: # Train Pipeline
    def __init__(self, env, n_games, c_puct, pacman_policy, ghost_policy):
        self.env = env
        self.n_games = n_games
        self.c_puct = c_puct

        self.pacman_policy = pacman_policy
        self.ghost_policy = ghost_policy

        self.buffer_size = 10000
        self.data_selfplay = deque(maxlen=self.buffer_size)

        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        """
        if init_True:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTS(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        """

    def self_play(self, n_games=1000):
        for i in range(n_games):
            gamestates, probs, rewards = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = data_process(gamestates, probs, rewards)
            self.data_buffer.extend(play_data)
    
    def self_play_step(self):
        self.env.reset()
        states, actions, probs, current_players = [], [], [], []
        if_pacman = True
        gamestate = self.env.game_state()
        while True:
            player = MCTS(self.pacman_policy, self.ghost_policy, self.env, gamestate, if_pacman)
            action, action_prob = player.search()
            states.append(gamestate)
            actions.append(action)
            probs.append(action_prob)
            current_players.append("pacman" if if_pacman else "ghost")
            if not if_pacman:
                self.env.step(actions[-1], action)
            if_pacman = False if if_pacman else True
            gamestate = self.env.game_state()
            done, reward_pacman, reward_ghost = gamestate
            if done:
                reward = np.zeros(len(current_players))
                reward[np.array(current_players) == "pacman"] = reward_pacman
                reward[np.array(current_players) == "ghpst"] = reward_ghost
                return zip(states, probs, reward)

    def evaluate(self):
        # def __init__(self, pacman_net, ghost_net, env, root_state, root_extra, c_puct=1.0, num_simulations=10000):
        current_mcts_player = MCTS(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = {}
        for i in range(self.n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / self.n_games
        return win_ratio

    def update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        
        return loss, entropy

    def train(self):
        best_win_ratio = 0.0
        for i in range(self.n_games):
            self.self_play(self.n_games)
            if (i+1) % self.check_freq == 0:
                win_ratio = self.policy_evaluate()
                self.pacman_policy.save_model('pacman_zero.pth')
                self.ghost_policy.save_model('ghost_zero.pth')
                print("Batch: %d / %d, acc: %f", i, self.n_games, win_ratio)
                if win_ratio > best_win_ratio:
                    print("NEW BEST")
                    self.pacman_policy.save_model('pacman_zero_best.pth')
                    self.ghost_policy.save_model('ghost_zero_best.pth')
                    if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                        self.pure_mcts_playout_num += 1000
                        self.best_win_ratio = 0.0