SHELL:=/bin/bash

PODMANAGER=podman
PYTHON_RUNNER=poetry run
PYTHON_SCRIPTS=scripts

install:
	@poetry install

update:
	@poetry update

clean:
	@poetry cache clean

build:
	@$(PODMANAGER) build --tag bp-federated-base --file containers/base.dockerfile .
