# Description

The new anomaly detection approach not only detects anomalies based on the path a vehicle takes but also based on a vehicles velocity and on the traffic. The method uses information of all the vehicles in a scene and is able to detect vehicles that ignore the right of way or are too careful at crossings (wait too long).

Through a dynamic clustering algorithm it should further be possible to give human feedback to the anomalies at runtime and to classify the anomalies, e.g.: not interesting/not harmful/critical/dangerous behavior. The now labeled anomalies will be clustered.
When a new anomaly is detected it will be checked whether is can be put in an existing cluster of similar anomalies or when it is too different from other stored anomalies it will be asked for human feedback/classification.

A visualisation of the preliminary approach can be seen in docs/ApproachVisualization.pdf.
![Approach Visualization](docs/ApproachVisualization.pdf)

# Usage

The repository can be used for 3 separated steps: 
- the preparation of datasets usabe for the CNN based on the data provided by the sae (movementpredictor/data)
  - you need to store the sae data as a file of type .saedump and provide the file's path in the .env variable PATH_SAE_DATA and then run main_data.py (the generated datasts will be stored in PATH_STORE_DATA) and the frame resolution will be DIM_X, DIM_Y or ...
  - you extract the sae data in another way and only use the functions in movementpredicor/data/ to generate a dataset (see repository sae-anomaly-detection)
- the training of the movement-prediction convolutional neural network (movementpredictor/cnn)
  - train the model by running main_training.py
  - necessary program input: all environment variables to store everything for successful later inferencing, most importantly:
    -  PATH_STORE_DATA (path to the previously generated datasets) and 
    -  PATH_INFERENCE_BUNDLE (path where to store the model weights and all parameters necessary to use the model for inferencing)
- calculation of parameters for the anomaly detection and performing clustering (movementpredictor/clustering) 
  - you can use the trained CNN to make predictions on huge datasets, based on all outputs a probability threshold is calculated so that the unlikliest PERCENTAGE_OF_ANOMALIES percent of all samples are considered as anomalies; the treshold will be stored in the same location as the model weights
  - these anomalies are further clustered
  - necessary program input: PATH_STORE_DATA, PATH_INFERENCE_BUNDLE (to get the trained weights and to store parameters like the threshold), PERCENTAGE_OF_ANOMALIES and if you want to generate videos of found anomalies you need PATH_SAE_DATA

## Github Workflows and Versioning

The following Github Actions are available:

* [PR build](.github/workflows/pr-build.yml): Builds docker image for each pull request to main branch. Inside the docker image, `poetry install` and `poetry run python test.py` are executed, to compile and test entire python code.
* [Create release](.github/workflows/create-release.yml): Manually executed action. Creates a github release with tag. Poetry is updating to next version by using "patch, minor and major" keywords. If you want to change to non-incremental version, set version in directly in pyproject.toml and execute create release afterwards.

## Dependabot Version Update

With [dependabot.yml](.github/dependabot.yml) a scheduled version update via Dependabot is configured. Dependabot creates a pull request if newer versions are available and the compilation is checked via PR build.