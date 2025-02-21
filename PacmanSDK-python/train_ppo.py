import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import sys
import os

from model_ppo import *
from collections import namedtuple
from core.GymEnvironment import PacmanEnv
from utils.state_dict_to_tensor import state_dict_to_tensor

env = PacmanEnv("local")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def init_weights(m):
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

def init_network(network, name):
    if os.path.exists(name):
        network.load_state_dict(torch.load(name, map_location=device, weights_only=True))
    else:
        print("No checkpoint found. Training from scratch.")
        network.apply(init_weights)

# feature_nn = FeatureNetwork(4, 40)
feature_pacman = FeatureNetwork(4, 40)
feature_ghost = FeatureNetwork(4, 40)
pacman_pn = PacmanNetwork_Policy(5)
pacman_vn = PacmanNetwork_Value()
ghost_pn = GhostNetwork_Policy(5)
ghost_vn = GhostNetwork_Value()

# init_network(feature_nn, "feature_nn_ppo.pth")
init_network(feature_pacman, "feature_pacman_ppo.pth")
init_network(feature_ghost, "feature_ghost_ppo.pth")
init_network(pacman_pn, "pacman_pn_ppo.pth")
init_network(pacman_vn, "pacman_vn_ppo.pth")
init_network(ghost_pn, "ghost_pn_ppo.pth")
init_network(ghost_vn, "ghost_vn_ppo.pth")

# feature_nn.to(device)
feature_pacman.to(device)
feature_ghost.to(device)
pacman_pn.to(device)
pacman_vn.to(device)
ghost_pn.to(device)
ghost_vn.to(device)

ppo = PPO(pacman_pn, pacman_vn, ghost_pn, ghost_vn, feature_pacman, feature_ghost)
mcts = MCTS(env, pacman_vn, ghost_vn)

if __name__ == "__main__":
    ITE = 1000
    BATCH = 400
    TARGET_UPDATE = 10

    for episode in range(ITE):
        state = env.reset(mode="local")
        state, extra = state_dict_to_tensor(state)
        state = state.to(device)
        extra = extra.to(device)
        states, extras, rewards, next_states, next_extras, dones, pacman_old, ghost_old, rewards1, rewards2 = [], [], [], [], [], [], [], [], [], []

        for t in range(BATCH):
            """
            if random.random() < 0.3:
                action_pacman, action_ghost = mcts.search(state, extra)
            else:
                action_pacman = ppo.get_pacman_action(state, extra)
                action_ghost = ppo.get_ghost_action(state, extra)
            """
            features_pacman = feature_pacman(state, extra)
            features_ghost = feature_ghost(state, extra)
            action_pacman, probs_pacman = ppo.get_pacman_action(features_pacman)
            prob_pacman = probs_pacman[0][action_pacman]
            action_ghost, probs_ghost = ppo.get_ghost_action(features_ghost)
            probs_ghost = torch.tensor([probs_ghost[0][action_ghost[0]], probs_ghost[1][action_ghost[1]] , probs_ghost[2][action_ghost[2]]], device=device)

            next_state, reward1, reward2, done, _ = env.step(action_pacman.item(), action_ghost.tolist())
            next_state, next_extra = state_dict_to_tensor(next_state)
            next_state = next_state.to(device)
            next_extra = next_extra.to(device)

            states.append(state)
            extras.append(extra)
            next_states.append(next_state)
            next_extras.append(next_extra)

            # rewards.append(reward1 - np.sum(reward2)) # rewards形状[BATCH * reward]
            rewards1.append(reward1)
            rewards2.append(np.sum(reward2))
            pacman_old.append(prob_pacman) # actions_pacman形状[BATCH * [ ]]
            ghost_old.append(probs_ghost.unsqueeze(0)) # actions_ghost形状[BATCH * [ , , ]]
            dones.append(done)

            state = next_state
            extra = next_extra

        # rewards = torch.FloatTensor(rewards).to(device)
        rewards1 = torch.FloatTensor(rewards1).to(device)
        rewards2 = torch.FloatTensor(rewards2).to(device)
        dones = torch.BoolTensor(dones).to(device)
        states = torch.cat(states, dim=0).to(device)
        extras = torch.cat(extras, dim=0).to(device)
        next_states = torch.cat(next_states, dim=0).to(device)
        next_extras = torch.cat(next_extras, dim=0).to(device)
        pacman_old = torch.cat(pacman_old, dim=0).to(device)
        ghost_old = torch.cat(ghost_old, dim=0).to(device)

        loss1, loss2, reward = ppo.train(states, extras, next_states, next_extras, rewards1, rewards2, pacman_old, ghost_old, dones)

        if episode % TARGET_UPDATE == 0:
        # if True:
            print("(Iteration %d / %d) loss: %f,%f; reward: %s"% (episode, ITE, loss1, loss2, (reward, )))
            torch.save(pacman_pn.state_dict(), "pacman_pn_ppo.pth")
            torch.save(pacman_vn.state_dict(), "pacman_vn_ppo.pth")
            torch.save(ghost_pn.state_dict(), "ghost_pn_ppo.pth")
            torch.save(ghost_vn.state_dict(), "ghost_vn_ppo.pth")
            torch.save(feature_pacman.state_dict(), "feature_pacman_ppo.pth")
            torch.save(feature_ghost.state_dict(), "feature_ghost_ppo.pth")