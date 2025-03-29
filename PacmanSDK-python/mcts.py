import copy
import numpy as np
import torch

from core.gamedata import *
from core.GymEnvironment import *
from model import *
from utils.state_dict_to_tensor import *
from utils.valid_action import *
from utils.ghostact_int2list import *
from utils.PacmanEnvDecorator import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCTSNode:
    def __init__(self, env:PacmanEnvDecorator, agent:Agent, otheragent:Agent, done:bool, parent=None):
        self.env=copy.deepcopy(env)
        self.state=self.env.game_state()
        self.state_dict=self.state.gamestate_to_statedict()

        self.agent = agent
        self.otheragent = otheragent
        self.isPacman = agent.is_pacman()
        self.action_dim = 5 if self.isPacman else 125

        self.parent=parent
        self.children = [None] * self.action_dim
        self.done = done
        self.expanded = False

        self.N=0
        self.P=np.zeros(self.action_dim, dtype=np.float32)
        self.W=0.0
        self.Q=0.0

        # self.children_W = np.zeros()
        # self.chidren_Q = np.zeros()

    def is_terminal(self) -> bool:
        return self.done

    def is_expanded(self) -> bool:
        return self.expanded
    
    def is_pacman(self) -> bool:
        return self.isPacman

    def expand(self) -> float:
        if self.is_terminal():
            return
        
        with torch.no_grad():
            state_tensor = state2tensor(self.state)
            selected_action, action_probs, value = self.agent.predict(self.state)

        self.P = action_probs.cpu().numpy()
        self.expanded = True

        if self.is_pacman():
            pos = self.state.gamestate_to_statedict()["pacman_coord"]
            legal_actions = get_valid_moves_pacman(pos, self.state)
        else:
            pos = self.state.gamestate_to_statedict()["ghosts_coord"]
            legal_actions = get_valid_moves_ghost(pos, self.state)

        for action in legal_actions:
            env_copy=copy.deepcopy(self.env)
            if self.isPacman:
                ghost_action = self._select_opponent_action()
                _, _, _, done, _ = env_copy.step(action, ghost_action, self.state)
            else:
                pacman_action = self._select_opponent_action()
                _, _, _, done, _ = env_copy.step(pacman_action, action, self.state)
            self.children[action] = MCTSNode(env_copy, self.agent, self.otheragent, done, parent=self)

        return value.item()

    def select(self, c_puct:float=2.5):
        if not self.is_expanded():
            raise ValueError("Not expanded")
        
        if self.is_pacman():
            pos = self.state.gamestate_to_statedict()["pacman_coord"]
            legal_actions = get_valid_moves_pacman(pos, self.state)
        else:
            pos = self.state.gamestate_to_statedict()["ghosts_coord"]
            legal_actions = get_valid_moves_ghost(pos, self.state)
        
        uct_scores = [
            self.children[action].Q + c_puct * self.P[action] * np.sqrt(self.N) / (1 + self.children[action].N)
            for action in legal_actions
        ] 
       
        # TODO 能否修改使之并行化？应该需要把Q等东西储存在parent中

        best_action = legal_actions[np.argmax(uct_scores)]
        
        return best_action, self.children[best_action]

    def update(self, value:float) -> None:
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def _select_opponent_action(self) -> int:
        # idea 2: 使用对手的mcts或者纯神经网络
        selected_action, _, _ = self.otheragent.predict(self.state)
        return selected_action

        # idea 1: 随机选择
        if self.is_pacman():
            pos = self.state.gamestate_to_statedict()["ghosts_coord"]
            legal_actions = get_valid_moves_ghost(pos, self.state)
        else:
            pos = self.state.gamestate_to_statedict()["pacman_coord"]
            legal_actions = get_valid_moves_pacman(pos, self.state)
        return np.random.choice(legal_actions)

class MCTS:
    def __init__(self, env:PacmanEnvDecorator, agent:Agent, otheragent:Agent, c_puct:float, n_simulations:int, n_search:int, temp:float, det:bool=True):
        self.env=copy.deepcopy(env)
        self.agent=agent
        self.otheragent=otheragent
        
        self.det=det

        self.c_puct=c_puct
        self.n_simulations=n_simulations
        self.n_search=n_search
        self.inv_temp=temp

    def search(self, node:MCTSNode) -> float:
        if node.is_terminal():
            return self.get_terminal_value(node)

        if not node.is_expanded():
            value = node.expand()
            node.update(value)
            return value

        action, child = node.select(self.c_puct)
        value = self.search(child)
        node.update(value)
        return value

    def run(self) -> tuple[int, torch.tensor, float]:
        self.root=MCTSNode(self.env, self.agent, self.otheragent, done=False)
        for _ in range(self.n_search):
            value = self.search(self.root)
        selected_action, prob = self.decide()
        return selected_action, prob, value

    def decide(self) -> tuple[int, torch.tensor]:
        visits=torch.zeros(self.root.action_dim, device=device, dtype=torch.float32)
        sum_visits=0.0
        
        state = self.env.game_state()

        if self.root.is_pacman():
            pos = state.gamestate_to_statedict()["pacman_coord"]
            legal_actions = get_valid_moves_pacman(pos, state)
        else:
            pos = state.gamestate_to_statedict()["ghosts_coord"]
            legal_actions = get_valid_moves_ghost(pos, state)
        
        for action in legal_actions:
            node = self.root.children[action]
            if node:
                visits[action] += node.N**self.inv_temp
                sum_visits += node.N**self.inv_temp
        
        prob = visits/sum_visits

        if self.det:
            selected_action = torch.argmax(prob)
        else:
            selected_action = torch.multinomial(prob, 1)

        return (selected_action, prob)

    def get_terminal_value(self) -> float:
        raise NotImplementedError

class MCTS_pacman(MCTS):
    def __init__(self, env:PacmanEnvDecorator, pacman:Agent, ghost:Agent, c_puct:float, n_simulations:int, n_search:int, temp:float=1, det:bool=True):
        super().__init__(env=env, agent=pacman, otheragent=ghost, c_puct=c_puct, n_simulations=n_simulations, n_search=n_search, temp=temp, det=det)

    def get_terminal_value(self) -> float:
        return float(self.env.game_state().pacman_score)

class MCTS_ghost(MCTS):
    def __init__(self, env:PacmanEnvDecorator, ghost:Agent, pacman:Agent, c_puct:float, n_simulations:int, n_search:int, temp:float=1, det:bool=True):
        super().__init__(env=env, agent=ghost, otheragent=pacman, c_puct=c_puct, n_simulations=n_simulations, n_search=n_search, temp=temp, det=det)
    
    def get_terminal_value(self) -> float:
        return float(self.env.is_eaten()) - 2*int(self.env.is_gone())
    
if __name__ == "__main__":
    import time

    env = PacmanEnvDecorator()
    env.reset()
    pacman = PacmanAgent(load_series="03290333")
    ghost = GhostAgent()
    
    t= time.time()
    mcts = MCTS_pacman(env=env, pacman=pacman, ghost=ghost, c_puct=2.5, n_simulations=10, n_search=10)
    action, prob, value = mcts.run()
    t = time.time()-t
    
    print(t)
    print(action, prob, value)