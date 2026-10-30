#!/usr/bin/env bash
# Tier 3 references + Tier 2 ablations (3 seeds each). Rerunning skips finished seeds.
# Every ablation changes ONE factor of the Tier 2 default; "baseline" is the default retrained on
# the same machine, so all deltas are measured against identical hardware / software.
set -euo pipefail
cd "$(dirname "$0")/../src"
PY=${PY:-python}
A=../results/ablation

$PY -u patchcore.py                                  # Tier 3: PatchCore, one-class (same paradigm as Tier 2)
$PY -u CNN.py --method resnet                        # Tier 3: ResNet-18, supervised (upper bound)
$PY -u CNN.py --method sup                           # Tier 3 control: Tier 2 CNN, supervised

$PY -u CNN.py --out $A/baseline                      # Tier 2 default, local rerun
$PY -u CNN.py --out $A/binary --binary               # A3 head: 2-way instead of 9-way
for c in 0.01 0.05 0.10; do                          # A1 training contamination
  $PY -u CNN.py --out $A/contam_$c --contam $c
done
for t in Center Donut Edge-Loc Edge-Ring Loc Near-full Random Scratch; do   # A2 leave-one-type-out
  $PY -u CNN.py --out $A/drop_$t --drop-types $t
done
