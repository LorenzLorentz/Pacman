import numpy as np
import torch

from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, env:PacmanEnvDecorator, pacman:PacmanAgent, ghost:GhostAgent, update_epoch:int=20, episodes:int=10):
        self.GAMMA=0.99
        self.LAMBDA=0.95
        self.EPS=0.8

        self.UPDATE_EPOCH=update_epoch
        self.EPISODES=episodes

        self.env=env
        
        self.pacman=pacman
        self.ghost=ghost

    def play(self):
        states, actions_pacman, actions_ghost, values_pacman, next_values_pacman, values_ghost, next_values_ghost, rewards_pacman, rewards_ghost, dones = [], [], [], [],[], [],[], [],[], [],
        self.env.reset()
        
        while True:
            with torch.no_grad():
                state=self.env.game_state()
                selected_action_pacman, _, value_pacman = self.pacman.predict(state)
                selected_action_ghost, _, value_ghost = self.ghost.predict(state)
                dict, reward_pacman, reward_ghost, done, eatAll=self.env.step(selected_action_pacman, ghostact_int2list(selected_action_ghost))
                next_state=self.env.game_state()
                _, _, next_value_pacman=self.pacman.predict(next_state)
                _, _, next_value_ghost=self.ghost.predict(next_state)
                
                states.append(state)
                
                actions_pacman.append(selected_action_pacman)
                actions_ghost.append(selected_action_ghost)
                values_pacman.append(value_pacman)
                values_ghost.append(value_ghost)
                next_values_pacman.append(next_value_pacman)
                next_values_ghost.append(next_value_ghost)
                rewards_pacman.append(reward_pacman)
                rewards_ghost.append(np.sum(reward_ghost))
                dones.append(done)
            
            if done:
                break
        
        actions_pacman=torch.tensor(actions_pacman, device=device, dtype=torch.int64)
        actions_ghost=torch.tensor(actions_ghost, device=device, dtype=torch.int64)
        rewards_pacman=torch.tensor(rewards_pacman, device=device, dtype=torch.float32)
        rewards_ghost=torch.tensor(rewards_ghost, device=device, dtype=torch.float32)
        dones=torch.tensor(dones, device=device, dtype=torch.int8)
        
        values_pacman=torch.stack(values_pacman).to(device)
        values_ghost=torch.stack(values_ghost).to(device)
        next_values_pacman=torch.stack(next_values_pacman).to(device)
        next_values_ghost=torch.stack(next_values_ghost).to(device)

        return states, actions_pacman, actions_ghost, values_pacman, next_values_pacman, values_ghost, next_values_ghost, rewards_pacman, rewards_ghost, dones

    def compute_advantage(self, td_delta:list):
        advantages=[]
        advantage=0.0
        for delta in reversed(td_delta):
            advantage=self.GAMMA*self.LAMBDA*advantage+delta
            advantages.append(advantage)
        advantages.reverse()
        return torch.tensor(advantages, device=device, dtype=torch.float32).squeeze()
    
    def train_step(self):
        states, actions_pacman, actions_ghost, values_pacman, next_values_pacman, values_ghost, next_values_ghost, rewards_pacman, rewards_ghost, dones = self.play()
        
        states_tensor=[]
        legal_actions_mask_pacman=[]
        legal_actions_mask_ghost=[]
        for state in states:
            state_tensor=state2tensor(state)
            states_tensor.append(state_tensor)
            pos_pacman = state.gamestate_to_statedict()["pacman_coord"]
            pos_ghost = state.gamestate_to_statedict()["ghosts_coord"]
            legal_action_mask_pacman = get_valid_moves_pacman_mask(pos_pacman, state)
            legal_action_mask_ghost = get_valid_moves_ghosts_mask(pos_ghost, state)
            legal_actions_mask_pacman.append(legal_action_mask_pacman)
            legal_actions_mask_ghost.append(legal_action_mask_ghost)
        states_tensor=torch.cat(states_tensor)
        legal_actions_mask_pacman=torch.stack(legal_actions_mask_pacman).to(device)
        legal_actions_mask_ghost=torch.stack(legal_actions_mask_ghost).to(device)

        with torch.no_grad():
            _, old_prob_pacman, _ =self.pacman.predict_batch(states_tensor, legal_actions_mask_pacman)
            old_prob_pacman=old_prob_pacman.gather(1, actions_pacman.unsqueeze(1)).squeeze(1).detach()
            
            _, old_prob_ghost, _ =self.ghost.predict_batch(states_tensor, legal_actions_mask_ghost)
            old_prob_ghost=old_prob_ghost.gather(1, actions_ghost.unsqueeze(1)).squeeze(1).detach()
            
            td_target_pacman=rewards_pacman+self.GAMMA*next_values_pacman*(1-dones)
            td_delta_pacman=td_target_pacman-values_pacman
            advantages_pacman=self.compute_advantage(td_delta_pacman).detach()

            td_target_ghost=rewards_ghost+self.GAMMA*next_values_ghost*(1-dones)
            td_delta_ghost=td_target_ghost-values_ghost
            advantages_ghost=self.compute_advantage(td_delta_ghost).detach()
        
        for _ in range(self.UPDATE_EPOCH):
            print(f"  update epoch:{_}")
            self.pacman.ppo_train(states_tensor, legal_actions_mask_pacman, old_prob_pacman, actions_pacman, td_target_pacman, advantages_pacman, self.EPS)
            self.ghost.ppo_train(states_tensor, legal_actions_mask_ghost, old_prob_ghost, actions_ghost, td_target_pacman, advantages_ghost, self.EPS)
        # self.pacman.save_model()
        # self.ghost.save_model()

    def train(self):
        for _ in range(self.EPISODES):
            print(f"Episodes:{_}")
            self.train_step()
            if _%2==0:
                self.pacman.save_model()
                self.ghost.save_model()

if __name__=="__main__":
    env=PacmanEnvDecorator(PacmanEnv())
    pacman=PacmanAgent(load_series="03092227")
    ghost=GhostAgent(load_series="03092227")
    ppo=PPO(env, pacman, ghost, episodes=20)
    ppo.train()