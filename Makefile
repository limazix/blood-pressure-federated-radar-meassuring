SHELL:=/bin/bash

PODMANAGER=podman
IMAGE_TAG_PREFIX=bp-federated

install:
	@poetry install

update:
	@poetry update

clean:
	@poetry cache clean

build:
	@$(PODMANAGER) build --tag $(IMAGE_TAG_PREFIX)-base --file containers/base.dockerfile .
	@$(PODMANAGER) build --tag $(IMAGE_TAG_PREFIX)-global-agent --file containers/global_agent.dockerfile .
