.PHONY: install build-deb clean

export PACKAGE_NAME=movementpredictor

install: check-settings
	poetry install

check-settings:
	./check_settings.sh

test: install
	poetry run pytest
