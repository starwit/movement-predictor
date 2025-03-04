# SAE Anomaly Detection

This repository takes the results of the Starwit Awareness Engine from valkey, calls Anomaly Detection libs to get anomalies and send the results as AnomalyMessage to Valkey.

## Run for Development

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

## Github Workflows and Versioning

The following Github Actions are available:

* [PR build](.github/workflows/pr-build.yml): Builds docker image for each pull request to main branch. Inside the docker image, `poetry install` and `poetry run python test.py` are executed, to compile and test entire python code.
* [Build and publish latest image](.github/workflows/build-publish-latest.yml): Manually executed action. Same like PR build. Additionally puts latest docker image to internal docker registry.
* [Create release](.github/workflows/create-release.yml): Manually executed action. Creates a github release with tag, docker image in internal docker registry, helm chart in chartmuseum by using and incrementing the version in pyproject.toml. Poetry is updating to next version by using "patch, minor and major" keywords. If you want to change to non-incremental version, set version in directly in pyproject.toml and execute create release afterwards.

## Dependabot Version Update

With [dependabot.yml](.github/dependabot.yml) a scheduled version update via Dependabot is configured. Dependabot creates a pull request if newer versions are available and the compilation is checked via PR build.
