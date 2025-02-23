import numpy as np
import torch

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

def state2tensor(state):
    state_dict=state.gamestate_to_statedict()
    
    board=np.array(state_dict["board"])
    padding_num = 42 - state_dict["board_size"]
    board = np.pad(board, pad_width=(padding_num//2, padding_num-padding_num//2), mode="constant", constant_values=0)

    board_coin = board
    board_coin[board_coin>=4]=0

    board_special = board
    board_special[board_special<=3]=0
    board_special[board_special==9]=0

    pacman_pos = np.zeros((42, 42))
    pacman = state_dict["pacman_coord"]
    pacman_pos[pacman[0] + padding_num//2][pacman[1] + padding_num//2] = 1

    ghost_pos = np.zeros((42, 42))
    for ghost in state_dict["ghosts_coord"]:
        ghost_pos[ghost[0] + padding_num][ghost[1] + padding_num] = 1

    portal_pos = np.zeros((42, 42))
    portal = state_dict["portal_coord"]
    if portal[0] != -1 and portal[1] != -1 and state_dict["portal_available"]:
        portal_pos[portal[0] + padding_num][portal[1] + padding_num] = 1

    extra=state_dict["pacman_skill_status"]
    extra=np.insert(extra, 5, state_dict["round"])
    extra=np.insert(extra, 6, state_dict["beannumber"])
    extra_expanded=np.resize(extra, (42, 42))

    state_arrays = [board, board_coin, board_special, pacman_pos, ghost_pos, portal_pos, extra_expanded]
    state_stacked = np.stack(state_arrays, axis=0)
    state_tensor = torch.tensor(state_stacked, dtype=torch.float32, device=device).unsqueeze(0)

    return state_tensor