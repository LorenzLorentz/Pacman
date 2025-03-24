import os
import json
import random
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
        portal_available = event["portal_available"]
        pacman_skill_status = event["pacman_skills"]

        pacman_action = get_action(last_pacman_coord, current_pacman_coord)
        ghost_action = [get_action(last_ghosts_coord[i], current_ghosts_coord[i]) for i in range(3)]

        # BEAMNUMBER
        for point in event["pacman_step_block"]:
            point = [int(p) for p in point]
            if (0 <= point[0] < len(board) and 0 <= point[1] < len(board[0])):
                if board[point[0]][point[1]] in [Space.REGULAR_BEAN, Space.BONUS_BEAN]:
                    board[point[0]][point[1]] = Space.EMPTY
                    beannumber -= 1

        # EATEN
        EATEN = False
        for _ in range(3):
            if current_pacman_coord == current_ghosts_coord[_]:
                EATEN=True
                break
        if EATEN:
            if pacman_skill_status[3]>0:
                EATEN=False

        # GONE
        GONE = False
        if portal_coord == current_pacman_coord and portal_available:
            GONE = False

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
            pacman_skill_status=pacman_skill_status,
            pacman_pos=current_pacman_coord,
            ghosts_pos=current_ghosts_coord,
            pacman_score=current_score[0],
            ghosts_score=current_score[1],
            beannumber=beannumber,
            portal_available=portal_available,
            portal_coord=portal_coord,
        )

        pacman_reward = current_score[0]
        ghost_reward = current_score[1]

        pacman_prob=torch.zeros([5], device=device, dtype=torch.float16) # 
        ghost_prob=torch.zeros([125], device=device, dtype=torch.float16) # 

        pacman_prob[:]=eps/5
        pacman_prob[pacman_action]+=1-eps

        ghost_prob[:]=eps/125
        for _ in range(5):
            ghost_prob[_ + ghost_action[1]*5 + ghost_action[2]*25] += (1-eps)/2/15
            ghost_prob[ghost_action[0] + _*5 + ghost_action[2]*25] += (1-eps)/2/15
            ghost_prob[ghost_action[0] + ghost_action[1]*5 + _*25] += (1-eps)/2/15
        ghost_prob[ghost_action[0] + ghost_action[1]*5 + ghost_action[2]*25] += (1-eps)/2

        point = (gamestate, pacman_prob, torch.tensor(pacman_reward, device=device, dtype=torch.float16), 
                 ghost_prob, torch.tensor(ghost_reward, device=device, dtype=torch.float16), EATEN, GONE)
        traj.append(point)

        last_pacman_coord = current_pacman_coord
        last_ghosts_coord = current_ghosts_coord
        last_score = current_score

    if len(traj) > 0:
        trajs.append(traj)

    return trajs

def create_batches(dataset, batch_size=512):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        
        inputs = [item[0] for item in batch]
        output_policies = [item[1][:-1] for item in batch]
        output_values = [item[1][-1] for item in batch]
        
        batch_inputs = torch.cat(inputs)
        batch_policies = torch.stack(output_policies)
        batch_values = torch.stack(output_values)
        
        print("???", batch[0][0].shape, batch[0][1][:-1].shape, batch[0][1][-1].shape)
        print("!!!", batch_inputs.shape, batch_policies.shape, batch_values.shape)

        batches.append((batch_inputs, batch_policies, batch_values))
    
    return batches

def dataset_synthesize_from_traj(trajs:list, batch_size=128, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dataset_pacman=[]
    dataset_ghost=[]

    for traj in trajs:
        for datapoint in traj:
            gamestate, pacman_prob, pacman_reward, ghost_prob, ghost_reward, EATEN, GONE = datapoint
            input_tensor = state2tensor(gamestate)
            output_tensor_pacman = state2pacmanout(pacman_prob, pacman_reward)
            output_tensor_ghost = state2ghostout(ghost_prob, EATEN, GONE)
            dataset_pacman.append((input_tensor, output_tensor_pacman))
            dataset_ghost.append((input_tensor, output_tensor_ghost))

    random.shuffle(dataset_pacman)
    random.shuffle(dataset_ghost)

    total = len(dataset_pacman)
    train_end = int(total*train_ratio)
    val_end = train_end + int(total*val_ratio)
    
    train_data_pacman = dataset_pacman[:train_end]
    val_data_pacman = dataset_pacman[train_end:val_end]
    test_data_pacman = dataset_pacman[val_end:]

    train_data_ghost = dataset_ghost[:train_end]
    val_data_ghost = dataset_ghost[train_end:val_end]
    test_data_ghost = dataset_ghost[val_end:]

    print(f"Total datapoints: {total}")
    print(f"Train: {len(train_data_pacman)}, Validation: {len(val_data_pacman)}, Test: {len(test_data_pacman)}")

    train_batches_pacman = create_batches(train_data_pacman, batch_size)
    val_batches_pacman = create_batches(val_data_pacman, batch_size)
    test_batches_pacman = create_batches(test_data_pacman, batch_size)

    torch.save(train_batches_pacman, 'selfplay/train_dataset_pacman.pt')
    torch.save(val_batches_pacman, 'selfplay/val_dataset_pacman.pt')
    torch.save(test_batches_pacman, 'selfplay/test_dataset_pacman.pt')

    train_batches_ghost = create_batches(train_data_ghost, batch_size)
    val_batches_ghost = create_batches(val_data_ghost, batch_size)
    test_batches_ghost = create_batches(test_data_ghost, batch_size)

    torch.save(train_batches_ghost, 'selfplay/train_dataset_ghost.pt')
    torch.save(val_batches_ghost, 'selfplay/val_dataset_ghost.pt')
    torch.save(test_batches_ghost, 'selfplay/test_dataset_ghost.pt')

def dataset_synthesize(json_dir:str="matchdata_json", batch_size:int=512, train_ratio:float=0.8, val_ratio:float=0.1, test_ratio:float=0.1):
    dataset_pacman=[]
    dataset_ghost=[]

    for filename in os.listdir(json_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data_json = [json.loads(line) for line in file if line.strip()]
            trajs=data_synthesize(data_json)
            for traj in trajs:
                for datapoint in traj:
                    gamestate, pacman_prob, pacman_reward, ghost_prob, ghost_reward, EATEN, GONE = datapoint
                    input_tensor = state2tensor(gamestate)
                    output_tensor_pacman = state2pacmanout(pacman_prob, pacman_reward)
                    output_tensor_ghost = state2ghostout(ghost_prob, EATEN, GONE)
                    dataset_pacman.append((input_tensor.squeeze(), output_tensor_pacman[:-1], output_tensor_pacman[-1]))
                    dataset_ghost.append((input_tensor.squeeze(), output_tensor_ghost[:-1], output_tensor_ghost[-1]))

    random.shuffle(dataset_pacman)
    random.shuffle(dataset_ghost)

    total = len(dataset_pacman)
    train_end = int(total*train_ratio)//batch_size*batch_size
    val_end = train_end + int(total*val_ratio)//batch_size*batch_size
    
    train_data_pacman = dataset_pacman[:train_end]
    val_data_pacman = dataset_pacman[train_end:val_end]
    test_data_pacman = dataset_pacman[val_end:]

    train_data_ghost = dataset_ghost[:train_end]
    val_data_ghost = dataset_ghost[train_end:val_end]
    test_data_ghost = dataset_ghost[val_end:]

    print(f"Total datapoints: {total}")
    print(f"Train: {len(train_data_pacman)}, Validation: {len(val_data_pacman)}, Test: {len(test_data_pacman)}")

    torch.save(train_data_pacman, 'data/train_dataset_pacman.pt')
    torch.save(val_data_pacman, 'data/val_dataset_pacman.pt')
    torch.save(test_data_pacman, 'data/test_dataset_pacman.pt')

    torch.save(train_data_ghost, 'data/train_dataset_ghost.pt')
    torch.save(val_data_ghost, 'data/val_dataset_ghost.pt')
    torch.save(test_data_ghost, 'data/test_dataset_ghost.pt')

if __name__=="__main__":
    dataset_synthesize()