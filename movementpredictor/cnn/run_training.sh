#!/usr/bin/env bash
set -euo pipefail

# ————————————————————————————————————————————————————————————————
#  Static env parameters
# ————————————————————————————————————————————————————————————————
export MODEL_ARCHITECTURE="MobileNet_v3"          
export PATH_SAE_DATA_TRAIN="movementpredictor/data/source/RABPenn106th_2025-03-07.saedump"
export PATH_SAE_DATA_TEST="movementpredictor/data/source/RABPenn106th_2025-04-09.saedump"
export NAME_DATA="0.5sec"
export TIME_DIFF_PREDICTION="0.5"
export CAMERA="RABPenn106th"
export PERCENTAGE_OF_ANOMALIES="99.7"
export PIXEL_PER_AXIS="120"

# ————————————————————————————————————————————————————————————————
#  Loop over symmetric & asymmetric, 5 runs each
# ————————————————————————————————————————————————————————————————
for OUTPUT_DISTR in symmetric asymmetric; do
  export OUTPUT_DISTR
  for i in {0..4}; do
    echo
    echo "Trial #$i — output_distr=$OUTPUT_DISTR"

    python movementpredictor/cnn/main_training.py

    # rename the generated weights
    MODEL_DIR="models/${CAMERA}/${NAME_DATA}/${MODEL_ARCHITECTURE}_${OUTPUT_DISTR}_prob"
    SRC="${MODEL_DIR}/model_weights.pth"
    DST="${MODEL_DIR}/model_weights${i}.pth"

    if [[ -f "$SRC" ]]; then
      mv "$SRC" "$DST"
      echo "✓ Renamed $SRC → $DST"
    else
      echo "Warning: $SRC not found, skipping rename"
    fi
  done
done

echo
echo "All runs completed."