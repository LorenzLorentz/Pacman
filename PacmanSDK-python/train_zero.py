import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import sys
import os

from collections import namedtuple
from core.GymEnvironment import PacmanEnv
from utils.state_dict_to_tensor import state_dict_to_tensor

from model_zero import *

env = PacmanEnv("local")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pacman_model = PolicyValueNet(4, 40, 5)
ghost_model = PolicyValueNet(4, 40, 5)

trainer = AlphaZeroTrainer(pacman_model, ghost_model)

EPOCHS = 10000
UPDATES = 10000

for epoch in range(10000):
    trainer.self_play()
    trainer.train_step()
    if epoch % UPDATES == 0:
        trainer.evaluate()