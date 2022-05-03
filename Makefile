SHELL:=/bin/bash

PODMANAGER=podman
IMAGE_TAG_PREFIX=bp-federated

GLOBAL_AGENT_NAME=global_agent

install:
	@poetry install

update:
	@poetry update

clean:
	@poetry cache clean

build:
	@$(PODMANAGER) build --tag $(IMAGE_TAG_PREFIX)-base --file containers/base.dockerfile .
	@$(PODMANAGER) build --tag $(IMAGE_TAG_PREFIX)-$(GLOBAL_AGENT_NAME) --file containers/global_agent.dockerfile .
	@$(PODMANAGER) build --tag $(IMAGE_TAG_PREFIX)-$(GLOBAL_AGENT_NAME) --file containers/local_agent.dockerfile .

run:
	@$(PODMANAGER) run -p 8080:8080 --name $(GLOBAL_AGENT_NAME) -it $(IMAGE_TAG_PREFIX)-$(GLOBAL_AGENT_NAME)

stop:
	@$(PODMANAGER) stop $(GLOBAL_AGENT_NAME)
	@$(PODMANAGER) rm $(GLOBAL_AGENT_NAME)
