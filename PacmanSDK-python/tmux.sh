#!/bin/bash
# tmux kill-session -t pacman

tmux new-session -d -s pacman "python train_bc.py"
tmux ls