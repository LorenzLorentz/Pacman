import numpy as np
import torch
import copy

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
        half = patch_size // 2
        board_padded = np.pad(board, (half, half, half, half), mode='constant', value=0)
        board_padded = board_padded.squeeze(0).squeeze(0)
        c_row, c_col = center
        c_row = int(c_row.item()) if isinstance(c_row, torch.Tensor) else int(c_row)
        c_col = int(c_col.item()) if isinstance(c_col, torch.Tensor) else int(c_col)
        c_row += half
        c_col += half
        patch = board_padded[c_row-half:c_row+half+1, c_col-half:c_col+half+1]
        return patch

def state2tensor(state) -> torch.tensor:
    state_dict=state.gamestate_to_statedict()
    
    board=np.array(state_dict["board"])
    padding_num = 42 - state_dict["board_size"]
    board = np.pad(board, pad_width=(padding_num//2, padding_num-padding_num//2), mode="constant", constant_values=0)

    board_pos = np.zeros_like(board)

    pacman = state_dict["pacman_coord"]
    pacman_nearby = extract_nearby(board, pacman, patch_size=7)

    for ghost in state_dict["ghosts_coord"]:
        board_pos[ghost[0] + padding_num][ghost[1] + padding_num] += 2

    portal = state_dict["portal_coord"]
    if portal[0] != -1 and portal[1] != -1 and state_dict["portal_available"]:
        board_pos[portal[0] + padding_num][portal[1] + padding_num] = 3

    # board_avail = copy.deepcopy(board)
    # board_avail[board_avail>=1] = 1

    # board_coin = copy.deepcopy(board)
    # board_coin[board_coin>=4]=0

    # board_special = copy.deepcopy(board)
    # board_special[board_special<=3]=0
    # board_special[board_special==9]=0

    extra=state_dict["pacman_skill_status"]
    extra=np.insert(extra, 5, state_dict["round"])
    extra=np.insert(extra, 6, state_dict["beannumber"])
    extra_expanded=np.resize(extra, (42, 42))

    state_arrays = [board, board_pos, extra_expanded]
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