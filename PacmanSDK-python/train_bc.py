import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.ghostact_int2list import *
from core.gamedata import *
from model import *
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=True)

        # data = torch.load(data_path)
        # if isinstance(data, list) and isinstance(data[0], list):
        #     self.data = [item for batch in data for item in batch]
        # else:
        #     self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Trainer:
    def __init__(self, agent, train_loader, val_loader, test_loader, num_epochs=10, num_test=5):
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.num_test = num_test

    def train(self):
        self.agent.ValueNet.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            for batch in self.train_loader:
                inputs, targets = zip(*batch)
                inputs = torch.stack(inputs).to(device)
                target_policies = torch.stack([t[0] for t in targets]).to(device)
                target_values = torch.stack([t[1] for t in targets]).to(device)

                with autocast("cuda"):
                    p, v = self.agent.ValueNet(inputs)
                    loss_policy = -torch.sum(target_policies*torch.log(p+1e-8)) / inputs.size(0)
                    loss_value = torch.mean((target_values.view(-1) - v.view(-1))**2)
                    loss = loss_policy + loss_value
                
                self.agent.optimizer.zero_grad()
                self.agent.scaler.scale(loss).backward()
                self.agent.scaler.step(self.agent.optimizer)
                self.agent.scaler.update()

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()
                num_batches += 1

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/num_batches:.4f}, "
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
                inputs, targets = zip(*batch)
                inputs = torch.stack(inputs).to(self.device)
                target_policies = torch.stack([t[0] for t in targets]).to(device)
                target_values = torch.stack([t[1] for t in targets]).to(device)
                
                p, v = self.agent.ValueNet(inputs)
                loss_policy = -torch.sum(target_policies*torch.log(p+1e-8)) / inputs.size(0)
                loss_value = torch.mean((target_values.view(-1) - v.view(-1))**2)
                loss = loss_policy+loss_value

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()
                num_batches += 1

        print(f"Validation Epoch {epoch+1}: Loss: {total_loss/num_batches:.4f}, "
                f"Policy Loss: {total_policy_loss/num_batches:.4f}, Value Loss: {total_value_loss/num_batches:.4f}")
    
    def test(self):
        self.agent.ValueNet.eval()

        is_pacman = (self.agent.ValueNet.policy_fc.out_features==5)
        total_v_error = 0.0
        total_p_error = 0.0
        
        num_points = 0
        eps = 1e-8
        
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = zip(*batch)
                inputs = torch.stack(inputs).to(device)
                target_policies = torch.stack([t[:-1] for t in targets]).to(device)
                target_values = torch.stack([t[-1] for t in targets]).to(device)

                with autocast("cuda"):
                    p, v = self.agent.ValueNet(inputs)

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
                    total_policy_error += error_p
                
        print(f"Test: Prob acc: {1-total_p_error/num_points}, Value acc: {1-total_v_error/num_points}")

if __name__ == '__main__':
    batch_size = 512
    num_epochs = 20

    train_dataset_pacman = Dataset("data/train_dataset_pacman.pt")
    val_dataset_pacman = Dataset("data/val_dataset_pacman.pt")
    test_dataset_pacman = DataLoader("data/test_data_pacman.pt")
    train_loader_pacman = DataLoader(train_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_pacman = DataLoader(val_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_pacman = DataLoader(test_dataset_pacman, batch_size=batch_size, shuffle=True, num_workers=4)

    train_dataset_ghost = Dataset("data/train_dataset_ghost.pt")
    val_dataset_ghost = Dataset("data/val_dataset_ghost.pt")
    test_dataset_ghost = DataLoader("data/test_data_ghost.pt")
    train_loader_ghost = DataLoader(train_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_ghost = DataLoader(val_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_ghost = DataLoader(test_dataset_ghost, batch_size=batch_size, shuffle=True, num_workers=4)
    
    pacman = PacmanAgent()
    trainer = Trainer(pacman, train_loader_pacman, val_loader_pacman, test_dataset_pacman, num_epochs=num_epochs)
    trainer.train()
    pacman.save_model()

    # ghost = GhostAgent()
    # trainer = Trainer(ghost, train_dataset_ghost, val_dataset_ghost, test_dataset_ghost, num_epochs=num_epochs)
    # trainer.train()
    # ghost.save_model()