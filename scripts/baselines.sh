#!/bin/bash

# Declare the arrays
datasets=("covid" "seer" "compas" "cutract" "maggic")
n_values=(10 50 100)
seeds=(0 1 2 3 4 5 6 7 8 9 10)

# Iterate through the arrays
for seed in "${seeds[@]}"
do
  for ns in "${n_values[@]}"
  do
    for dataset in "${datasets[@]}"
    do
      # Run the command
      fuser -v /dev/nvidia0 -k
      python ../src/run_baselines.py --dataset $dataset --ns $ns --seed $seed

    done
  done
done


