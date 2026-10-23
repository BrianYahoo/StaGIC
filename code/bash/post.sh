#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p ../logs

gpu_id=${GPU_ID:--1}
seed_start=${SEED_START:-1}
seed_end=${SEED_END:-100}
species='human'
atlas='Glasser'
metric='fiber_count'
approach='edr'
for ((seed=seed_start; seed<=seed_end; seed++))
do
    for step in 4 5
    do
        python ../script/running.py "$gpu_id" "$species" "$atlas" "$metric" "$approach" "$seed" "$step" 2>> "../logs/err${gpu_id}hm.log"
    done
done
