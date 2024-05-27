#!/bin/bash

elements=("stump" "bicycle" "bonsai" "counter" "garden" "kitchen" "room")

for element in "${elements[@]}"; do
    command="python benchmark.py --model_path /mnt/d/Stanford/CS348K/models/$element --source_path /mnt/d/Stanford/CS348K/360_v2/$element --data_device cuda --benchmark 3"
    output=$(eval $command)
    times_data=$(echo "$output" | grep 'Times')
    echo "$times_data"
done
