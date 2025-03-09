import copy
import numpy as np
import torch

from core.gamedata import *
from core.GymEnvironment import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.ghostact_int2list import *
from utils.PacmanEnvDecorator import *

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