FROM bp-federated-base

ARG APP_PATH
ARG SUBJECT_ID

ENV SUBJECT_ID ${SUBJECT_ID}

WORKDIR ${APP_PATH}

RUN mkdir ./data

ENTRYPOINT [ "poetry", "run" ]
CMD [ "scripts/main.py", "--is-federated" ]