import numpy as np
import torch
import copy

from core.gamedata import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_dict_to_tensor(state_dict):
    board = state_dict["board"]
    if isinstance(board, list):
        board = np.array(board)
    size = board.shape[0]
    # print(board)
    # pad board to 38x38
    padding_num = 41 - size
    board = np.pad(board, pad_width=(0, padding_num),
                   mode="constant", constant_values=0)
    # pacman position matrix
    pacman_pos = np.zeros((41, 41))
    if "pacman_pos" in state_dict:
        pacman_pos[state_dict["pacman_pos"][0] + padding_num][
            state_dict["pacman_pos"][1] + padding_num
        ] = 1

    # ghost position matrix
    ghost_pos = np.zeros((41, 41))
    if "ghost_pos" in state_dict:
        for ghost in state_dict["ghost_pos"]:
            ghost_pos[ghost[0] + padding_num][ghost[1] + padding_num] = 1

    portal_pos = np.zeros((41, 41))
    if "portal" in state_dict:
        portal = state_dict["portal"]
        if portal[0] != -1 and portal[1] != -1:
            portal_pos[portal[0] + padding_num][portal[1] + padding_num] = 1

    level = state_dict["level"]
    if "round" in state_dict:
        round = state_dict["round"]
    else:
        round = 0
    # board_size = state_dict['board_size']
    portal_available = False
    if "portal_available" in state_dict:
        portal_available = int(state_dict["portal_available"])

    # print(board.shape, pacman_pos.shape, ghost_pos.shape,
    #       board_area.shape, portal_pos.shape)
    return torch.tensor(
        np.stack([board, pacman_pos, ghost_pos, portal_pos]),
        dtype=torch.float32,
    ).unsqueeze(0), torch.tensor(
        np.array([level, round, size, portal_available] * 10), dtype=torch.float32
    ).unsqueeze(
        0
    )

def one_hot(board:np.ndarray, num_classes:int) -> np.ndarray:
    one_hot = np.eye(num_classes)[board]
    one_hot = one_hot.transpose(2, 0, 1)
    return one_hot

def extract_nearby(board, center, patch_size=7):
    padding_num = patch_size//2
    board_padded = np.pad(board, pad_width=(padding_num, padding_num), mode='constant', constant_values=0)
    nearby = board_padded[center[0]:center[0]+2*padding_num+1, center[1]:center[1]+2*padding_num+1]
    return nearby

def state2tensor(state:GameState) -> torch.tensor:
    state_dict=state.gamestate_to_statedict()
    
    board=np.array(state_dict["board"])
    padding_num = 42 - state_dict["board_size"]
    board = np.pad(board, pad_width=(padding_num//2, padding_num-padding_num//2), mode="constant", constant_values=0)

    extra = np.zeros(board.shape)

    pacman = copy.deepcopy(state_dict["pacman_coord"])
    pacman[0] += padding_num
    pacman[1] += padding_num
    pacman_nearby = extract_nearby(board, pacman, patch_size=7)

    ghosts = copy.deepcopy(state_dict["ghosts_coord"])
    ghosts_nearby = []
    for ghost in ghosts:
        ghost[0] += padding_num
        ghost[1] += padding_num
        ghosts_nearby.append(extract_nearby(board, ghost, patch_size=7))

    portal = copy.deepcopy(state_dict["portal_coord"])
    if portal[0] != -1 and portal[1] != -1 and state_dict["portal_available"]:
        portal[0] += padding_num
        portal[1] += padding_num
        portal_nearby = extract_nearby(board, portal, patch_size=7)
    else:
        portal_nearby = np.zeros([7, 7])

    extra_info=state_dict["pacman_skill_status"]
    extra_info=np.insert(extra_info, 5, state_dict["round"]//100)
    extra_info=np.insert(extra_info, 6, state_dict["beannumber"]//100)

    for i, (x,y) in enumerate([(0, 0), (0, 7), (0, 14)]):
        extra[x:x+7, y:y+7] = ghosts_nearby[i]
        
    extra[0:7, 21:21+7] = pacman_nearby

    extra[0:7, 28:28+7] = portal_nearby

    extra[0:1, 35:35+7] = extra_info

    # board_avail = copy.deepcopy(board)
    # board_avail[board_avail>=1] = 1

    # board_coin = copy.deepcopy(board)
    # board_coin[board_coin>=4]=0

    # board_special = copy.deepcopy(board)
    # board_special[board_special<=3]=0
    # board_special[board_special==9]=0

    state_arrays = [board, extra]
    state_stacked = np.stack(state_arrays, axis=0)
    state_tensor = torch.tensor(state_stacked, dtype=torch.float16, device=device).unsqueeze(0)

    return state_tensor

def state2pacmanout(prob:torch.tensor, reward:float) -> torch.tensor:
    return torch.cat((prob, torch.tensor([reward], device=device, dtype=torch.float16)), dim=0).to(device)

def state2ghostout(prob:torch.tensor, EATEN:bool, GONE:bool) -> torch.tensor:
    return torch.cat((prob, torch.tensor([int(EATEN)-2*int(GONE)], device=device, dtype=torch.float16)), dim=0).to(device)

def state2pacman() -> torch.tensor:
    pass

def state2ghost() -> torch.tensor:
    pass