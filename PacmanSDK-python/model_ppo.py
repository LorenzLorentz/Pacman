import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from train import state_dict_to_tensor
from core.GymEnvironment import PacmanEnv
import sys
import copy
import random
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureNetwork(nn.Module):
    def __init__(self, input_channel_num, extra_size):
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

        self.fc = nn.Linear(512, 512)

        self.encoder= nn.Sequential(
            nn.Linear(extra_size, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
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
        
        return features

"""

class FeatureNetwork(nn.Module):
    def __init__(self, input_channel_num, num_actions, extra_size):
        super().__init__()
        self.channels = input_channel_num
        self.embeddings = nn.ModuleList([nn.Embedding(10, 16) for _ in range(input_channel_num)])

        self.conv1 = nn.Conv2d(64, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.encoder = nn.Linear(extra_size, 64)

    def forward(self, x, y):
        B, C, H, W = x.shape
        embedded_channels = []
        for i in range(self.channels):
            flattened_channel = x[:, i, :, :].view(B, -1).long() # ??? 数据类型?
            embedded_channel = self.embeddings[i](flattened_channel)
            embedded_channel = embedded_channel.view(B, 16, H, W)
            embedded_channels.append(embedded_channel)
        x = torch.cat(embedded_channels, dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = F.relu(self.conv3(x))
        y = torch.sigmoid(self.encoder(y))
        features = x.view(x.size(0), -1) + y

        return features

"""

class PacmanNetwork_Policy(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.policy_head=nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, features):
        action_logits = self.policy_head(features)
        return action_logits
    
class PacmanNetwork_Value(nn.Module):
    def __init__(self):
        super().__init__()

        self.value_head=nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        value = self.value_head(features)
        return value

class GhostNetwork_Policy(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.policy_head=nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions*3)
        )

    def forward(self, features):
        action_logits = self.policy_head(features).view(-1,3,5)
        return action_logits
    
class GhostNetwork_Value(nn.Module):
    def __init__(self):
        super().__init__()

        self.value_head=nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        value = self.value_head(features)
        return value

class PPO:
    def __init__(self, pacman_policy, pacman_value, ghost_policy, ghost_value, feature_pacman, feature_ghost):
        super().__init__()
        self.params={"GAMMA":0.99,"LR_P":5e-2, "LR_V":5e-2,"EPSILON":0.6, "UPDATE_EPOCHS":40,"LAMBDA": 0.95,}
        
        self.feature_pacman = feature_pacman
        self.feature_ghost = feature_ghost
        self.pacman_policy = pacman_policy
        self.pacman_value = pacman_value
        self.ghost_policy = ghost_policy
        self.ghost_value = ghost_value

        self.pacman_policy_optimizer = optim.Adam(self.pacman_policy.parameters(), self.params["LR_P"])
        self.pacman_value_optimizer = optim.Adam(self.pacman_value.parameters(), self.params["LR_V"])
        self.ghost_policy_optimizer = optim.Adam(self.ghost_policy.parameters(), self.params["LR_P"])
        self.ghost_value_optimizer = optim.Adam(self.ghost_value.parameters(), self.params["LR_V"])

        self.scaler = GradScaler('cuda')

    def get_pacman_action(self, features):
        logits = self.pacman_policy(features)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(0)
        return action, probs.detach()
    
    def get_ghost_action(self, features):
        logits = self.ghost_policy(features)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        actions = torch.multinomial(probs, 1).view(1,3).squeeze(0)
        return actions, probs.detach()
    
    def get_pacman_action_probs_batch(self, features):
        logits = self.pacman_policy(features)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(0)
        action_probs = probs.gather(dim=1, index=action).squeeze(-1)
        return action_probs
    
    def get_ghost_action_probs_batch(self, features):
        logits = self.ghost_policy(features)
        probs = F.softmax(logits, dim=-1)
        B, N, C = probs.shape
        flat_probs = probs.view(B * N, C)
        flat_actions = torch.multinomial(flat_probs, 1)
        actions = flat_actions.view(B, N).unsqueeze(-1)
        actions_probs = probs.gather(dim=2, index=actions).squeeze(-1)
        return actions_probs

    def compute_advantages(self, rewards, values, next_values, dones):
        T, B = rewards.shape
        deltas = rewards + self.params["GAMMA"] * next_values * (1 - dones.float()) - values
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            last_advantage = deltas[t] + self.params["GAMMA"] * self.params["LAMBDA"] * (1 - dones[t].float()) * last_advantage
            advantages[t] = last_advantage
        return advantages

        """
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            # 根据公式 delta_t = r_t + gamma * v[s_t+1] - v[s_t]
            # A_t = sum (gamma * lambda)^l delta_t+l
            # 因此存在递推公式 A_t = delta_t + gamma * delta * A_t+1
            delta = rewards[t] + self.params["GAMMA"] * next_values[t] - values[t]
            last_advantage = delta + self.params["GAMMA"] * self.params["LAMDA"] * last_advantage if not dones[t] else delta
            advantages.append(last_advantage)
        advantages.reverse()
        return torch.tensor(advantages, dtype=torch.float32).to(device)
        """
    
    def train(self, states, extras, next_states, next_extras, rewards_pacman, rewards_ghost, pacman_old, ghost_old, dones):
        rewards_pacman = rewards_pacman.unsqueeze(-1)
        rewards_ghost = rewards_ghost.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        with torch.no_grad():
            features_pacman = self.feature_pacman(states, extras)
            features_ghost = self.feature_ghost(states, extras)
            next_features_pacman = self.feature_pacman(next_states, next_extras)
            next_features_ghost = self.feature_ghost(next_states, next_extras)
            values_pacman = self.pacman_value(features_pacman)
            next_values_pacman = self.pacman_value(next_features_pacman)
            values_ghost = self.ghost_value(features_ghost)
            next_values_ghost = self.ghost_value(next_features_ghost)

            advantages_pacman = self.compute_advantages(rewards_pacman, values_pacman, next_values_pacman, dones)
            advantages_pacman = (advantages_pacman - advantages_pacman.mean()) / (advantages_pacman.std() + 1e-8)
            # advantages_pacman = advantages_pacman.unsqueeze(1)
            returns_pacman = advantages_pacman + values_pacman

            advantages_ghost = self.compute_advantages(rewards_ghost, values_ghost, next_values_ghost, dones)
            advantages_ghost = (advantages_ghost - advantages_ghost.mean()) / (advantages_ghost.std() + 1e-8)
            # advantages_ghost = advantages_ghost.unsqueeze(1)
            returns_ghost = advantages_ghost + values_ghost

            loss1 = []
            loss2 = []

        for epoch in range(self.params["UPDATE_EPOCHS"]):
            with autocast('cuda'):
                # features = self.feature_nn(states, extras)
                features_pacman = self.feature_pacman(states, extras)

                pacman_action_probs = self.get_pacman_action_probs_batch(features_pacman)
                ratio = pacman_action_probs / (pacman_old + 1e-10)
                clipped_ratio_pacman = torch.clamp(ratio, 1 - self.params["EPSILON"], 1 + self.params["EPSILON"])
                clipped_loss_pacman = torch.min(ratio * advantages_pacman, clipped_ratio_pacman * advantages_pacman)
                Pacman_Policy_loss = -torch.mean(clipped_loss_pacman)

                values_pred_pacman = self.pacman_value(features_pacman)
                value_loss_pacman = F.mse_loss(values_pred_pacman, returns_pacman)
                entropy_pacman = -(pacman_action_probs * torch.log(pacman_action_probs + 1e-10)).sum(dim=0).mean()

                Pacman_Total_loss = Pacman_Policy_loss + 0.5 * value_loss_pacman  - 0.002 * entropy_pacman

            self.pacman_policy_optimizer.zero_grad()
            self.pacman_value_optimizer.zero_grad()

            self.scaler.scale(Pacman_Total_loss).backward()

            self.scaler.step(self.pacman_policy_optimizer)
            self.scaler.step(self.pacman_value_optimizer)

            self.scaler.update()

            with autocast('cuda'):
                features_ghost = self.feature_ghost(states, extras)

                ghost_action_probs = self.get_ghost_action_probs_batch(features_ghost)
                ratio = ghost_action_probs / (ghost_old + 1e-10)
                clipped_ratio_ghost = torch.clamp(ratio, 1 - self.params["EPSILON"], 1 + self.params["EPSILON"])
                advantages_ghost_reshaped = advantages_ghost.unsqueeze(-1)
                clipped_loss_ghost = torch.min(ratio * advantages_ghost_reshaped, clipped_ratio_ghost * advantages_ghost_reshaped)
                Ghost_Policy_loss = -torch.mean(clipped_loss_ghost)

                values_pred_ghost = self.ghost_value(features_ghost)
                value_loss_ghost = F.mse_loss(values_pred_ghost, returns_ghost)
                entropy_ghost = -(ghost_action_probs * torch.log(ghost_action_probs + 1e-10)).sum(dim=0).mean()
                
                Ghost_Total_loss = Ghost_Policy_loss + 0.5 * value_loss_ghost  - 0.02 * entropy_ghost
            
            self.ghost_policy_optimizer.zero_grad()
            self.ghost_value_optimizer.zero_grad()

            self.scaler.scale(Ghost_Total_loss).backward()
            
            self.scaler.step(self.ghost_policy_optimizer)
            self.scaler.step(self.ghost_value_optimizer)

            self.scaler.update()

            loss1.append(Pacman_Total_loss)
            loss2.append(Ghost_Total_loss)

        return loss1.pop(), loss2.pop(), (rewards_pacman.mean().item(), rewards_ghost.mean().item())

class MCTSNode:
    def __init__(self, env, pacnman_action=None, ghost_action=None, parent=None):
        self.env = env
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

        self.done = False
        self.pacman_action = pacnman_action
        self.ghost_action = ghost_action

        self.next_env = self.env.clone()
        self.next_state = self.next_env.step(self.pacman_action, self.ghost_action)
        self.reward, self.done = self._get_info(self.next_state)
        self.next_state, self.next_extra = state_dict_to_tensor(self.next_state)

    def _get_info(self, state):
        _1,rp,rg,done,__ = state
        return (rp-rg, done)

    def update(self, value):
        self.value += value
        self.visits += 1

    def fully_expanded(self):
        return len(self.children) == 5**4
    
    def add_child(self, pacman_action, ghost_action):
        new_child = MCTSNode(self.next_env, pacman_action, ghost_action, self)
        self.children.append(new_child)

    def best_child(self, exploration_weight=1.4):
        if not self.children:
            return None
        ucb_values = [ (child.total_value / (child.visits + 1e-6)) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)) for child in self.children ]
        return self.children[np.argmax(ucb_values)]
        # return max(node.children, key=lambda child : (child.value / (child.visits + 1e-6)) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)))

    def expand(self):
        states = [(child.pacman_action, child.ghost_action) for child in self.children]
        for pacman_action in range(5):
            for ghost_action1 in range(5):
                for ghost_action2 in range(5):
                    for ghost_action3 in range(5):
                        pacman_action = [pacman_action]
                        ghost_action = [ghost_action1, ghost_action2, ghost_action3]
                        new_state = (pacman_action, ghost_action)
                        if new_state not in states:
                            self.add_child(pacman_action, ghost_action)
                            return self.children[-1]
        return None
            

class MCTS:
    def __init__(self, env, pacman_policy, ghost_policy, simulations=50):
        self.root_env = copy.deepcopy(env)
        self.pacman_policy = pacman_policy
        self.ghost_policy = ghost_policy
        self.simulations = simulations

    def search(self, env):
        self.root = MCTSNode(self.root_env)
        for iter in range(self.simulations):
            front = self.tree_search(self.root)
            value = self.value(front)
            self.backpropagate(front, value)
        return self.root.best_child(0).pacman_action, self.root.best_child(0).ghost_action

    def value(self, node):
        value = self.pacman_policy(node.next_state, node.next_extra)
        return value

    def tree_search(self, node):
        PROB = 0.5
        Exploration_Weight = 1.4
        while True:
            if len(node.children) == 0:
                return node.expand()
            elif random.uniform(0, 1) < PROB:
                node = node.best_child(node, Exploration_Weight)
            else:
                if not node.fully_expanded():
                    return node.expand()
                else:
                    node = node.best_child(node, Exploration_Weight)
            if node.done:
                break
        return node
    
    def backpropagate(node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent