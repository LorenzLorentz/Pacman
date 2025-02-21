import numpy as np
import torch

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