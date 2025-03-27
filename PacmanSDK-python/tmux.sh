#!/bin/bash
# tmux kill-session -t pacman

tmux new-session -d -s pacman "python train_bc.py"
# tmux new-session -d -s crawler "python crawler.py"
tmux ls