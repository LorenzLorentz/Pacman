import os
import json
import numpy as np
import torch
from gym import spaces

from core.gamedata import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_action(last_coord, current_coord):
    dx = current_coord[0] - last_coord[0]
    dy = current_coord[1] - last_coord[1]
    if dx > 0:
        return Direction.DOWN.value
    elif dx < 0:
        return Direction.UP.value
    elif dy > 0:
        return Direction.RIGHT.value
    elif dy < 0:
        return Direction.LEFT.value
    else:
        return Direction.STAY.value

def data_synthesize(data_json, eps=5e-2):
    trajs = []
    traj = []

    board = None
    board_size = None
    beannumber = None
    portal_coord = None
    last_pacman_coord = None
    last_ghosts_coord = None
    last_score = None

    for event in data_json:
        if "board" in event:
            board = np.array(event["board"]).copy()
            board_size = event["board_size"]
            beannumber = event["beannumber"]
            portal_coord = event["portal_coord"]
            last_pacman_coord = event["pacman_coord"]
            last_ghosts_coord = event["ghosts_coord"]
            last_score = event["score"]
            if len(traj) > 0:
                trajs.append(traj)
            traj=[]
            continue
        
        current_pacman_coord = event["pacman_coord"]
        current_ghosts_coord = event["ghosts_coord"]
        current_score = event["score"]

        pacman_action = get_action(last_pacman_coord, current_pacman_coord)
        ghost_action = [get_action(last_ghosts_coord[i], current_ghosts_coord[i]) for i in range(3)]

        for point in event["pacman_step_block"]:
            point = [int(p) for p in point]
            if (0 <= point[0] < len(board) and 0 <= point[1] < len(board[0])):
                if board[point[0]][point[1]] in [Space.REGULAR_BEAN, Space.BONUS_BEAN]:
                    board[point[0]][point[1]] = Space.EMPTY
                    beannumber -= 1

        gamestate = GameState(
            space_info={
                "observation_space": spaces.MultiDiscrete(np.ones((board_size, board_size)) * SPACE_CATEGORY),
                "pacman_action_space": spaces.Discrete(OPERATION_NUM),
                "ghost_action_space": spaces.MultiDiscrete(np.ones(3) * OPERATION_NUM),
            },
            level=event["level"],
            round=event["round"],
            board_size=board_size,
            board=board,
            pacman_skill_status=event["pacman_skills"],
            pacman_pos=current_pacman_coord,
            ghosts_pos=current_ghosts_coord,
            pacman_score=current_score[0],
            ghosts_score=current_score[1],
            beannumber=beannumber,
            portal_available=event["portal_available"],
            portal_coord=portal_coord,
        )

        pacman_reward = current_score[0]
        ghost_reward = current_score[1]

        pacman_prob=torch.zeros([5], device=device, dtype=torch.float32)
        ghost_prob=torch.zeros([125], device=device, dtype=torch.float32)

        pacman_prob[:]=eps/5
        pacman_prob[pacman_action]+=1-eps

        ghost_prob[:]=eps/125
        for _ in range(5):
            ghost_prob[_ + ghost_action[1]*5 + ghost_action[2]*25] += (1-eps)/2/15
            ghost_prob[ghost_action[0] + _*5 + ghost_action[2]*25] += (1-eps)/2/15
            ghost_prob[ghost_action[0] + ghost_action[1]*5 + _*25] += (1-eps)/2/15
        ghost_prob[ghost_action[0] + ghost_action[1]*5 + ghost_action[2]*25] += (1-eps)/2

        point = (gamestate, pacman_prob, torch.tensor(pacman_reward, device=device, dtype=torch.float32), 
                 ghost_prob, torch.tensor(ghost_reward, device=device, dtype=torch.float32), None, None)
        traj.append(point)

        last_pacman_coord = current_pacman_coord
        last_ghosts_coord = current_ghosts_coord
        last_score = current_score

    if len(traj) > 0:
        trajs.append(traj)

    return trajs

def train_zero_mimic(num_episodes=10):
    pacman=PacmanAgent()
    ghost=GhostAgent()
    best_loss=float('inf')
    for epi in range(num_episodes):
        for filename in os.listdir('matchdata_json'):
            if filename.endswith('.jsonl'):
                file_path = os.path.join('matchdata_json', filename)
                print("!!!", file_path)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data_json = [json.loads(line) for line in file if line.strip()]
                trajs=data_synthesize(data_json)
                for traj in trajs:
                    loss_pacman=pacman.train_batch(traj)
                    loss_ghost=ghost.train_batch(traj)
                    print(f"loss_pacman:{loss_pacman}, loss_ghost:{loss_ghost}")
                    
                    if loss_ghost+loss_pacman < best_loss:
                        pacman.save_model()
                        ghost.save_model()
                        best_loss=loss_pacman+loss_ghost

if __name__=="__main__":
    train_zero_mimic()