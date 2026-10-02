#!/bin/sh
# Trains the pro field from scratch:
#   1. pace: solo time trials on randomised tracks (evolution strategies)
#   2. racecraft: races against a self-play league from the front and back of the grid
#   3. field: the 20 best drivers, ranked by racing each other on unseen tracks
# Two architectures (64-64 and 64-64-64), two seeds each. About 30-40 minutes on a 16+ core machine.
set -e
cd "$(dirname "$0")/.."
mkdir -p models
for spec in c:64,64 d:64,64,64; do
  id=${spec%%:*}
  arch=${spec#*:}
  for s in 1 2; do
    [ -f models/pace-$id-s$s.json ] || node train/es.js --name pace-$id-s$s --arch $arch --stage tt --iters 200 --evalEvery 20 --seed $s > models/pace-$id-s$s.log 2>&1
    node train/es.js --name pro-$id-s$s --init models/pace-$id-s$s.json --stage race --field 8 --iters 100 --lr 0.005 --evalEvery 10 --seed $((s + 4)) > models/pro-$id-s$s.log 2>&1
  done
done
node train/field.js models/pro-*.json --out models/field.json > models/field.log 2>&1
echo "field built: models/field.json"
