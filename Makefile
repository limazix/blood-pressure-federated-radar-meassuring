SHELL:=/bin/zsh

VERBOSE=1

CURRENT_DIR=${shell pwd}
VENV_DIR=${CURRENT_DIR}/.venv
VENV_BIN_DIR=${VENV_DIR}/bin
SCRIPTS_DIR=${CURRENT_DIR}/scripts


setup:
	@python3 -m venv ${VENV_DIR}
	@${VENV_BIN_DIR}/pip install -r ${CURRENT_DIR}/requirements.txt

clean:
	@rm -rf ${VENV_DIR}
	@find ${SCRIPTS_DIR} -type f -name "*.pyc" -delete
	@find ${SCRIPTS_DIR} -type d -name "__pycache__" -delete

run-federated:
	@${VENV_BIN_DIR}/python scripts/main.py --is-federated

run-centralized:
	@${VENV_BIN_DIR}/python scripts/main.py

dashboard:
	@${VENV_BIN_DIR}/tensorboard serve --logdir ./lightning_logs
