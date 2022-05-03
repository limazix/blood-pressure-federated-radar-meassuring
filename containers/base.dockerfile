FROM docker.io/python:3.8

ARG APP_DIR=app
ARG APP_PATH=/opt/${APP_DIR}
ARG POETRY_VERSION=1.1.11

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 
ENV \
    POETRY_VERSION=$POETRY_VERSION \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR ${APP_PATH}

COPY ./poetry.lock ./pyproject.toml ./

COPY ./config.ini ./

RUN poetry install
