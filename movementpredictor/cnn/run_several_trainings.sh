#!/usr/bin/env bash
set -euo pipefail

# ————————————————————————————————————————————————————————————————
#  Static env parameters
# ————————————————————————————————————————————————————————————————
export MODEL_ARCHITECTURE="MobileNet_v3"          
export PATH_SAE_DATA_TRAIN="movementpredictor/data/source/MononElmStreetNB_2025-06-18_24h.saedump"    
export PATH_SAE_DATA_TEST="movementpredictor/data/source/MononElmStreetNB_2025-07-30_24h.saedump" 
export NAME_DATA="0.5sec"
export TIME_DIFF_PREDICTION="0.5"
export CAMERA="MononElmStreetNB"
export PERCENTAGE_OF_ANOMALIES="99.95"
export PIXEL_PER_AXIS="120"
export YOLO_OBJECT_TYPE_OF_INTEREST="2"
export FRAME_RATE="10"
export COMPUTE_STEPS="train"

# ————————————————————————————————————————————————————————————————
#  Loop over symmetric & asymmetric, 5 runs each
# ————————————————————————————————————————————————————————————————
for OUTPUT_DISTR in symmetric asymmetric; do
  export OUTPUT_DISTR
  for i in {0..4}; do
    echo
    echo "Trial #$i — output_distr=$OUTPUT_DISTR"

    python movementpredictor/main.py

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
    # rename the parameters
    SRC="${MODEL_DIR}/parameters.json"
    DST="${MODEL_DIR}/parameters${i}.json"

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