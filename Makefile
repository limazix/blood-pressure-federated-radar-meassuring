SHELL:=/bin/bash

VERBOSE=1

install:
	@poetry install

update:
	@poetry update

clean:
	@poetry cache clean

run-federated:
	@poetry run python scripts/main.py --is-federated

run-centralized:
	@poetry run python scripts/main.py

dashboard:
	@poetry run tensorboard serve --logdir ./lightning_logs