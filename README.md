# SAE Anomaly Detection

This repository takes the results of the Starwit Awareness Engine, calls Anomaly Detection libs to get anomalies and send the results as AnomalyMessage to Valkey.

## Run in Development

- go to `deployment/compose`
- copy .env.template and rename the copy to env.sh. Check version information - the version should be the same like in pyproject.toml
- execute `docker compose up`

## Prerequisites
- python 3.11, you can switch between python versions with pyenv:
  - see https://github.com/pyenv/pyenv/wiki#suggested-build-environment
  - `curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash`
  - choose environment with `pyenv local 3.11.9`
- Virtual env (e.g. sudo apt install python3.11-venv)
- Install Poetry
- Install Docker with compose plugin
- Running instance of starwit-awareness-engine, anomaly detection and Valkey

## Setup
- Create virtual environment with `python3 -m venv .venv && source .venv/bin/activate`
- Run `poetry install`, this should install all necessary dependencies
- Run `poetry run python main.py`. If you see log messages like `Received anomaly message from pipeline`, everything works as intended.

## Configuration
This code employs pydantic-settings for configuration handling. On startup, the following happens:
1. Load defaults (see `config.py`)
2. Read settings `settings.yaml` if it exists
3. Search through environment variables if any match configuration parameters (converted to upper_snake_case, nested levels delimited by `__`), overwriting the corresponding setting
4. Validate settings hierarchy if all necessary values are filled, otherwise Pydantic will throw a hopefully helpful error

The `settings.template.yaml` should always reflect a correct and fully fledged settings structure to use as a starting point for users.

## Publish Docker file
 - use github action `docker-internal build and push (build.yaml)` in order to create a new docker version

