# Anomaly Detection Through Probabilistic Movement Forecasting

## Description

The new anomaly detection approach not only detects anomalies based on the path a vehicle takes but also based on a vehicles velocity and on the traffic. The method uses information of all the vehicles in a scene and is able to detect vehicles that ignore the right of way or are too careful at crossings (wait too long).

Through a dynamic clustering algorithm it should further be possible to give human feedback to the anomalies at runtime and to classify the anomalies, e.g.: not interesting/not harmful/critical/dangerous behavior. The now labeled anomalies will be clustered.
When a new anomaly is detected it will be checked whether is can be put in an existing cluster of similar anomalies or when it is too different from other stored anomalies it will be asked for human feedback/classification.

A visualisation of the preliminary approach can be seen in docs/ApproachVisualization.pdf.
![Approach Visualization](docs/ApproachVisualization.pdf)

## Prerequisites

- python 3.11
- Virtual env (e.g. sudo apt install python3.11-venv)
- Install Poetry

## Setup

- Create virtual environment with 'python3 -m venv .venv && source .venv/bin/activate'
- Run 'poetry install', this should install all necessary dependencies
- Run one of the three main methods of the project (e.g. python3 movementpredictor/data/main_data.py) see [Usage](#usage)

## Usage

The repository can be used for 3 separated steps: 
- the preparation of datasets based on the data provided by the sae (movementpredictor/data)
  - you need to store the main sae data as a file of type '.saedump' and provide the file's path in the '.env' variable 'PATH_SAE_DATA_TEST' and then run 'main_data.py' (the generated datasts will be stored in movementpredictor/data/datasets/CAMERA/NAME_DATA)
  - provide your test dataset similarly and then run 'main_test_data.py'
  - necessary program input: 'PATH_SAE_DATA_TRAIN' and 'PATH_SAE_DATA_TEST' including the saedump paths for training and validation; 'TIME_DIFF_PREDICTION' to specify how many seconds ahead the vehicle's position will be predicted; 'CAMERA' and 'NAME_DATA' to store the data in fitting folders; and 'PIXEL_PER_AXIS' to provide the frame resolution
- the training of the movement-prediction convolutional neural network (movementpredictor/cnn)
  - train the model by running 'main_training.py'
  - necessary program input: all environment variables to store everything for successful later inferencing, most importantly:
    -  'MODEL_ARCHITECTURE' (the model architecture that should be used)
    -  'OUTPUT_DISTR' (the type of gaussian output distribution: 'symmetric' or 'aymmetric')
    -  'PIXEL_PER_AXIS' to spezify the input image size
    -  'CAMERA', 'NAME_DATA' (Camera from which the data originates -> data should be stored at movementpredictor/data/datasets/CAMERA/NAME_DATA)
  -  the model will be stored at models/CAMERA/NAME_MODEL (model weights and all parameters necessary to use the model for inferencing)
- calculation of parameters for the anomaly detection (movementpredictor/anomalydetection) 
  - you can use the trained CNN to make predictions on huge datasets
  - based on all outputs a probability threshold is calculated so that 'PERCENTAGE_OF_ANOMALIES' percent of all samples are considered as normal
  - the treshold will be stored in the same location as the model weights in parameters.json
  - future work: these anomalies should be further clustered
  - necessary program input:
    - 'CAMERA', 'OUTPUT_DISTR', 'MODEL_ARCHITECTURE', 'NAME_DATA' - to get the trained weights, data and to store parameters like the threshold
    - 'PERCENTAGE_OF_ANOMALIES'
    - if you want to generate videos of found anomalies you need 'PATH_SAE_DATA'
  
## Library Usage

After performing the 3 setup steps the movement predictor anomaly detection is ready to be used. All necessary functions and Classes can be imported like 'from movementpredictor import ...'. To perform inferencing and extraction of anomalies the recommended workflow is the following: 
- extract sae data with 'TrackingDataManager'
- smooth and filter the tracks with 'DataFilterer.apply_filtering'
- generate the dataset with 'makeTorchDataLoader' 
- load the model: 
  - 'import torch'
  - 'from movementpredictor.cnn import model_architectures'
  - 'model = model_architectures.get_model(architecture="MobileNet_v3", output_prob="asymmetric", path_model="path-to-your-model-weights.pth")'
- inferencing with 'inference_with_stats'
- extract anomalies with 'get_meaningful_unlikely_samples'

## Github Workflows and Versioning

The following Github Actions are available:

* [PR build](.github/workflows/pr-build.yml): Builds docker image for each pull request to main branch. Inside the docker image, `poetry install` and `poetry run python test.py` are executed, to compile and test entire python code.
* [Create release](.github/workflows/create-release.yml): Manually executed action. Creates a github release with tag. Poetry is updating to next version by using "patch, minor and major" keywords. If you want to change to non-incremental version, set version in directly in pyproject.toml and execute create release afterwards.

## Dependabot Version Update

With [dependabot.yml](.github/dependabot.yml) a scheduled version update via Dependabot is configured. Dependabot creates a pull request if newer versions are available and the compilation is checked via PR build.
