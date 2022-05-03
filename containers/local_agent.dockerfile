FROM bp-federated-base

ARG APP_PATH
ARG SUBJECT_ID

WORKDIR ${APP_PATH}

ENTRYPOINT [ "poetry", "run" ]
CMD [ "scripts/main.py", "--is-federated", "--subject-id=${SUBJECT_ID}" ]