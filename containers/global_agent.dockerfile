FROM bp-federated-base

ARG APP_PATH

WORKDIR ${APP_PATH}

ENTRYPOINT [ "poetry", "run" ]
CMD [ "scripts/main.py", "--is-global", "--is-federated" ]