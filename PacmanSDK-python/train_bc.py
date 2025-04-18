import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.seed import *
from utils.logger import *
from utils.ghostact_int2list import *
from core.gamedata import *
from model import *
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BCTrainer:
    def __init__(self, agent:Agent, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, test_loader:torch.utils.data.DataLoader, num_epochs:int=10, num_test:int=5, logger:logging=None, writer:SummaryWriter=None):
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.num_test = num_test

        self.best_acc =0.0

        self.logger = logger
        self.writter = writer

    def train(self):
        self.agent.ValueNet.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            for batch in self.train_loader:
                inputs, target_policies, target_values = batch
                inputs = inputs.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)

                with autocast("cuda"):
                    p, v = self.agent.ValueNet(inputs)
                    v = v.squeeze(1)
                    v = v/200
                    target_values = target_values/200
                    # loss_policy = F.l1_loss(p, target_policies)
                    loss_policy = F.kl_div(F.log_softmax(p, dim=1), target_policies, reduction="batchmean")
                    loss_value = F.mse_loss(v, target_values)
                    loss = loss_value + loss_policy
                
                self.agent.optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                self.agent.scaler.scale(loss).backward()
                self.agent.scaler.step(self.agent.optimizer)
                self.agent.scaler.update()

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()
                num_batches += 1

            # self.writer.add_scalar("TrainLoss/Total", total_loss/num_batches)
            # self.writer.add_scalar("TrainLoss/Policy", total_policy_loss/num_batches)
            # self.writer.add_scalar("TrainLoss/Value", total_value_loss/num_batches)
            
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/num_batches:.4f}, "
                    f"Policy Loss: {total_policy_loss/num_batches:.4f}, Value Loss: {total_value_loss/num_batches:.4f}")
            
            self.validate(epoch)

            if (epoch+1)%self.num_test==0:
                self.test()

    def validate(self, epoch):
        self.agent.ValueNet.eval()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, target_policies, target_values = batch
                inputs = inputs.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)

                with autocast("cuda"):
                    p, v = self.agent.ValueNet(inputs)
                    v = v.squeeze(1)
                    v = v/200
                    target_values = target_values/200
                    # loss_policy = F.l1_loss(p, target_policies)
                    loss_policy = F.kl_div(torch.log_softmax(p, dim=1), target_policies, reduction="batchmean")
                    loss_value = F.mse_loss(v, target_values)
                    loss = loss_policy + loss_value

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()
                num_batches += 1

        # self.writer.add_scalar("ValLoss/Total", total_loss/num_batches)
        # self.writer.add_scalar("ValLoss/Policy", total_policy_loss/num_batches)
        # self.writer.add_scalar("ValLoss/Value", total_value_loss/num_batches)
        
        if self.logger:
            self.logger.info(f"Validation Epoch {epoch+1}: Loss: {total_loss/num_batches:.4f}, "
                    f"Policy Loss: {total_policy_loss/num_batches:.4f}, Value Loss: {total_value_loss/num_batches:.4f}")
    
    def test(self):
        self.agent.ValueNet.eval()

        is_pacman = self.agent.ValueNet.if_Pacman
        
        total_v_error = 0.0
        total_p_error = 0.0
        num_points = 0
        eps = 1e-8
        
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, target_policies, target_values = batch
                inputs = inputs.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)

                with autocast("cuda"):
                    p, v = self.agent.ValueNet(inputs)
                    v = v.squeeze(1)

                for i in range(inputs.size(0)):
                    num_points += 1
                    v_pred = v[i].item()
                    v_target = target_values[i].item()
                    rel_error = abs(v_pred-v_target)/max(abs(v_target), eps)
                    error_v = rel_error if rel_error<=0.05 else 1.0
                    total_v_error += error_v

                    if is_pacman:
                        pred_action = torch.argmax(p[i]).item()
                        target_action = torch.argmax(target_policies[i]).item()
                        error_p = 0.0 if pred_action == target_action else 1.0
                    else:
                        pred_action_int = torch.argmax(p[i]).item()
                        target_action_int = torch.argmax(target_policies[i]).item()
                        pred_action_list = ghostact_int2list(pred_action_int)
                        target_action_list = ghostact_int2list(target_action_int)
                        mismatches = sum([1 for a, b in zip(pred_action_list, target_action_list) if a != b])
                        if mismatches == 0:
                            error_p = 0.0
                        elif mismatches == 1:
                            error_p = 0.5
                        else:
                            error_p = 1.0
                    total_p_error += error_p

        p_acc = 1-total_p_error/num_points
        v_acc = 1-total_v_error/num_points

        # self.writer.add_scalar("TestAcc/p_acc", p_acc)
        # self.writer.add_scalar("TestAcc/v_acc", v_acc)

        if 0.7*(p_acc-0.2) + 0.3*v_acc > self.best_acc:
            self.agent.save_model()
            self.best_acc = 0.7*(p_acc-0.2) + 0.3*v_acc
                
        if self.logger:
            logger.info(f"Test: Prob acc: {1-total_p_error/num_points}, Value acc: {1-total_v_error/num_points}")

if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)
    logger = get_logger(name="PacmanLog", seed=SEED, log_file="log/train_bc_{}.log".format(datetime.datetime.now().strftime("%m%d%H%M")))
    # writer = SummaryWriter(log_dir="/root/tf-logs")

    batch_size = 1024
    num_epochs = 400

    train_dataset_pacman = PacmanDataset("data/train_dataset_pacman.pt")
    val_dataset_pacman = PacmanDataset("data/val_dataset_pacman.pt")
    test_dataset_pacman = PacmanDataset("data/test_dataset_pacman.pt")
    train_loader_pacman = DataLoader(train_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_pacman = DataLoader(val_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader_pacman = DataLoader(test_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # train_dataset_ghost = PacmanDataset("data/train_dataset_ghost.pt")
    # val_dataset_ghost = PacmanDataset("data/val_dataset_ghost.pt")
    # test_dataset_ghost = PacmanDataset("data/test_data_ghost.pt")
    # train_loader_ghost = DataLoader(train_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader_ghost = DataLoader(val_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader_ghost = DataLoader(test_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
    
    pacman = PacmanAgent()
    trainer = BCTrainer(pacman, train_loader_pacman, val_loader_pacman, test_loader_pacman, num_epochs=num_epochs, logger=logger)
    trainer.train()
    pacman.save_model()

    # ghost = GhostAgent()
    # trainer = Trainer(ghost, train_dataset_ghost, val_dataset_ghost, test_dataset_ghost, num_epochs=num_epochs)
    # trainer.train()
    # ghost.save_model()