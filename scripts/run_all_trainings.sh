#!/usr/bin/env bash
# Run the three policy trainings SEQUENTIALLY (avoids the OOM/oversubscription
# that killed the parallel run on the 15 GB sandbox). Works on macOS or Linux.
#
# On an M4 Pro: leave threads high (fast per-core) and just run sequentially.
# On a small/shared box: keep threads modest to avoid memory blowups.
#
#   ./scripts/run_all_trainings.sh
#
# Each run checkpoints every ~20k steps to agents/checkpoints/ckpt/, so a crash
# resumes from the latest checkpoint rather than losing everything.
set -e
cd "$(dirname "$0")/.."

# Limit BLAS/torch threads a bit so memory + scheduler stay sane (tune for your box).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

echo "== [1/3] vec (vector movement) =="
python3 scripts/train_per_drone.py --difficulty medium --timesteps 2000000 \
  --continuous --tag vec --eval-episodes 16

echo "== [2/3] vol (vector + water-cost) =="
python3 scripts/train_per_drone.py --difficulty medium --timesteps 2000000 \
  --continuous --water-cost 0.5 --tag vol --eval-episodes 16

echo "== [3/3] dispatch (truck logistics) =="
python3 scripts/train_truck_dispatch.py --timesteps 3000000 --grid 140 \
  --trucks 3 --dpt 100 --max-steps 200 --base-spread 0.12 --water-cost 0.2 \
  --eval-episodes 12 --tag v1

echo "== all trainings done =="
