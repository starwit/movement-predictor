#!/bin/bash
export PATH_LABEL_BOX="movementpredictor/evaluation/label_box"
export PATH_STORE_PREDICTED_ANOMALIES="movementpredictor/evaluation/predictions"
export PATH_SAE_DUMPS="movementpredictor/data/source/RABPenn106th_2025-04-09.saedump"
export NUM_ANOMALIES=50
export CAMERA="RABPenn106th"
export MODEL_ARCHITECTURE="MobileNet_v3"


seconds_list=(
    "0.5sec"
    "1sec"
    "2sec"
)
model_weights_list=(
    "model_weights0.pth"
    "model_weights1.pth"
    "model_weights2.pth"
    "model_weights3.pth"
    "model_weights4.pth"
)

output_distr_list=(
    "symmetric"
    "asymmetric"
)

path_test_data_start="movementpredictor/data/datasets/"


for seconds in "${seconds_list[@]}"; do
    for distr in "${output_distr_list[@]}"; do
        for weights in "${model_weights_list[@]}"; do
            export MODEL_WEIGHTS="models/${CAMERA}/${seconds}/${MODEL_ARCHITECTURE}_${distr}_prob/${weights}"
            export OUTPUT_DISTR="$distr"
            export PATH_TEST_DATA="${path_test_data_start}/${CAMERA}/${seconds}/test/RABPenn106th_2025-04-09.pkl"

            if [ ! -f "$MODEL_WEIGHTS" ]; then
                echo "MODEL_WEIGHTS not found: $MODEL_WEIGHTS"
                continue
            fi

            echo "Running with:"
            echo "  MODEL_WEIGHTS=$MODEL_WEIGHTS"
            echo "  PATH_TEST_DATA=$PATH_TEST_DATA"
            echo "  OUTPUT_DISTR=$OUTPUT_DISTR"

            python3 movementpredictor/evaluation/eval_data.py

            echo "--- Finished ---"
        done
    done
done