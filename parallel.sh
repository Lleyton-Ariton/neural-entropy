#!/bin/bash

n=60

mkdir -p logs/

echo "Spawning a total of $n of tasks..."

for i in $(seq 1 $n); do
  echo "Spawning task [$i/$n]"

  python3 -u hyperopt.py --sweep_file sweep.json --study_name entropy-optimization --steps 10000 --simulation_trials 10 --n_trials 100 --device cpu --dtype f32 --dt 0.1 > "logs/hyperopt-task-${i}.log" 2>&1 &
done

echo "Finished spawning all tasks."
