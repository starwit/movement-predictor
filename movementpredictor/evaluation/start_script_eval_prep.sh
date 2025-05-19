#!/bin/bash
export PATH_LABEL_BOX="movementpredictor/evaluation/label_box"
export PATH_STORE_PREDICTED_ANOMALIES="movementpredictor/evaluation/predictions"
export PATH_SAE_DUMPS="movementpredictor/data/source/2024-10-23T12-03-05-0400_RangelineSMedicalDr_36h.saedump"
export NUM_ANOMALIES=200
export CAMERA="RangelineSMedicalDr"
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

path_test_data_start="movementpredictor/data/datasets/RangelineSMedicalDr/"


for seconds in "${seconds_list[@]}"; do
    for distr in "${output_distr_list[@]}"; do
        for weights in "${model_weights_list[@]}"; do
            export MODEL_WEIGHTS="models/RangelineSMedicalDr/${seconds}/${MODEL_ARCHITECTURE}_${distr}_prob/${weights}"
            export OUTPUT_DISTR="$distr"
            export PATH_TEST_DATA="${path_test_data_start}${seconds}/test/2024-10-23T12-03-05-0400_RangelineSMedicalDr_36h.pkl"

            if [ ! -f "$MODEL_WEIGHTS" ]; then
                echo "MODEL_WEIGHTS not found: $MODEL_WEIGHTS"
                continue
            fi

            echo "Running with:"
            echo "  MODEL_WEIGHTS=$MODEL_WEIGHTS"
            echo "  PATH_TEST_DATA=$PATH_TEST_DATA"
            echo "  OUTPUT_DISTR=$OUTPUT_DISTR"

            python3 movementpredictor/evaluation/eval_prep.py

            echo "--- Finished ---"
        done
    done
done