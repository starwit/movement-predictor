.PHONY: install build-deb clean

export PACKAGE_NAME=movementpredictor

install:
	poetry install

test: install
	poetry run pytest
