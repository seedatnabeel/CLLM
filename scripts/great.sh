#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
SCRIPT="$SCRIPT_DIR/../src/cllm/run_great.py"

# Declare the arrays
datasets=("covid" "seer" "compas" "cutract" "maggic")
n_values=(10 50 100)
seeds=(0 1 2)

# Iterate through the arrays
for dataset in "${datasets[@]}"
do
  for ns in "${n_values[@]}"
  do
    for runseed in "${seeds[@]}"
    do
      # Run the command
      # fuser -v /dev/nvidia0 -k # Use with caution
      python "$SCRIPT" --dataset $dataset --seed $runseed --ns $ns

    done
  done
done
