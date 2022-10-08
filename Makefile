SHELL:=/bin/bash

VERBOSE=1

PODMANAGER=docker
IMAGE_TAG_PREFIX=bp-federated

GLOBAL_AGENT_NAME=global_agent
LOCAL_AGENT_NAME=local_agent
CURRENT_DIR=$(shell pwd)

MAX_SUBJECTS := 30
SUBJECT_NUMBERS := $(shell seq -w 1 $(MAX_SUBJECTS))
SUBJECT_IDS := $(addprefix GDN00, $(SUBJECT_NUMBERS))


install:
	@poetry install

update:
	@poetry update

clean:
	@poetry cache clean

build:
	$(PODMANAGER) build --force-rm --tag $(IMAGE_TAG_PREFIX)-base --file containers/base.dockerfile .
	$(PODMANAGER) build --force-rm --tag $(IMAGE_TAG_PREFIX)-$(GLOBAL_AGENT_NAME) --file containers/global_agent.dockerfile .
	$(PODMANAGER) build --force-rm --tag $(IMAGE_TAG_PREFIX)-$(LOCAL_AGENT_NAME) --file containers/local_agent.dockerfile .

run-global:
	$(PODMANAGER) run --rm -p 8080:8080 --name $(GLOBAL_AGENT_NAME) --net host -it $(IMAGE_TAG_PREFIX)-$(GLOBAL_AGENT_NAME)

run-local:
	$(foreach ID, $(SUBJECT_IDS), $(PODMANAGER) run --name $(LOCAL_AGENT_NAME)-$(ID) --net host -v $(CURRENT_DIR)/data:/opt/app/data -e SUBJECT_ID=$(ID) $(IMAGE_TAG_PREFIX)-$(LOCAL_AGENT_NAME);)

stop:
	$(PODMANAGER) stop $(GLOBAL_AGENT_NAME)
	$(PODMANAGER) rm $(GLOBAL_AGENT_NAME)
	$(foreach ID, $(SUBJECT_IDS), $(PODMANAGER) stop $(LOCAL_AGENT_NAME)-$(ID))
	$(foreach ID, $(SUBJECT_IDS), $(PODMANAGER) rm $(LOCAL_AGENT_NAME)-$(ID))
