# Anomaly Detection Through Probabilistic Movement Forecasting

## Description

This project detects anomalies in vehicle behavior not only by path deviations but also by velocity and traffic context. It leverages all vehicles in a scene to spot right-of-way violations or overly cautious waiting. Detected anomalies can be clustered at runtime with human feedback into categories (e.g. “not interesting”, “critical”, “dangerous”). A visualization of the approach is in `docs/ApproachVisualization.pdf`.

## Prerequisites

- Python 3.10  
- Virtual env (e.g. sudo apt install python3.10-venv) 
- Poetry  

## Setup

```bash
git clone <repo-url>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
poetry install
```

## Usage

Example parameter values are provided in `movementpredictor/.env.template`. To run the full pipeline—or any subset of its steps—follow these instructions:

1. **Configure**  
   Create a `.env` in the project root (or copy `movementpredictor/.env.template`) and fill in your paths and parameters. In particular, set:
   - `PATH_SAE_DATA_TRAIN`, `PATHS_SAE_DATA_TEST` (SAE-dump files for training & validation)  
   - `TIME_DIFF_PREDICTION` (seconds ahead to predict)  
   - `CAMERA`, `NAME_DATA` (used to name output folders)  
   - `PIXEL_PER_AXIS` (frame resolution)  
   - `MODEL_ARCHITECTURE`, `OUTPUT_DISTR` (e.g. “asymmetric”)  
   - `PERCENTAGE_OF_ANOMALIES` (e.g. 99.95)  
   - `COMPUTE_STEPS` (comma-separated subset of `prepare,train,threshold`)  
   - etc.

2. **Run the pipeline**  
   ```bash
   python3 movementpredictor/main.py
   ```

By default, all three stages run in order (`prepare`, `train`, `threshold`).
If the .env Parameter `VISUALIZE` is set, plots and videos of found anomalies during the last stage are stored in folder `plots/<CAMERA>/<NAME_DATA>/…`. Frames are needed for this, so only set `VISUALIZE=True` if your sae-dumps contain frames. 

### Select individual stages

Adjust the `COMPUTE_STEPS` env var to run only the steps you need:

| Step         | What it does                                                                                                                |
|--------------|-----------------------------------------------------------------------------------------------------------------------------|
| **prepare**   | Builds the train & test datasets from the SAE-dump files.                                                                  |
| **train**     | Trains the movement-prediction CNN and saves model weights and parameters.                                                                |
| **threshold** | Runs inference on the test set to compute the anomaly threshold so that `PERCENTAGE_OF_ANOMALIES` % are normal; saved alongside the model.|


_Example:_ To only prepare data and train the model, set:

```dotenv
COMPUTE_STEPS="prepare,train"
```

## Outputs

After running, you’ll find:

- **Datasets** under  
  `movementpredictor/data/datasets/<CAMERA>/<NAME_DATA>/…`

- **Model & parameters** under  
  `models/<CAMERA>/<NAME_DATA>/…` (weights + `parameters.json`)

- **Anomaly threshold** in the same folder as the model, added to the `parameters.json`
  
- **plots & Videos** under `plots/<CAMERA>/<NAME_DATA>/…` if the .env Parameter `VISUALIZE` is set

All steps share the same configuration and folder conventions, making it easy to automate or integrate into larger workflows.

  
## Library Usage

After performing the 3 setup steps the movement predictor anomaly detection is ready to be used. All necessary functions and classes can be imported via 'from movementpredictor import ...'. To perform inferencing and extraction of anomalies the recommended workflow is the following: 
- extract sae data with `TrackingDataManager`
- smooth and filter the tracks with `DataFilterer.apply_filtering`
- generate the dataset with `makeTorchDataLoader` 
- load the model: 
  - `import torch`
  - `from movementpredictor.cnn import model_architectures`
  - `model = model_architectures.get_model(architecture="MobileNet_v3", output_prob="asymmetric", path_model="path-to-your-model-weights.pth")`
- inferencing with `inference_with_stats`
- extract anomalies with `get_meaningful_unlikely_samples`

## Model Evaluation and Dataset Generation

This repo includes supplementary scripts for model evaluation (`movementpredictor/evaluation/`)
and for generating the anomaly dataset released on Hugging Face (`movementpredictor/anomalydataset/`).

These scripts were used for the master’s thesis *Anomaly Detection in Traffic Applications:
A Probabilistic Forecasting Approach Based on Object Tracking* (Hanna Lichtenberg).
The exact model weights and configs used in the thesis are available here:[OneDrive](https://1drv.ms/f/c/29bd20621baa5af0/En9bfIFXlP5FqIJN5uQKIdkB31psQElUbdLvwUHE5YdH1A?e=XteuXG)

## Github Workflows and Versioning

The following Github Actions are available:

* [PR build](.github/workflows/pr-build.yml): Builds docker image for each pull request to main branch. Inside the docker image, `poetry install` and `poetry run python test.py` are executed, to compile and test entire python code.
* [Create release](.github/workflows/create-release.yml): Manually executed action. Creates a github release with tag. Poetry is updating to next version by using "patch, minor and major" keywords. If you want to change to non-incremental version, set version in directly in pyproject.toml and execute create release afterwards.

## Dependabot Version Update

With [dependabot.yml](.github/dependabot.yml) a scheduled version update via Dependabot is configured. Dependabot creates a pull request if newer versions are available and the compilation is checked via PR build.
