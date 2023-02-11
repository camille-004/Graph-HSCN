FROM ucsdets/scipy-ml-notebook:2022.3-stable

RUN pip install --upgrade pip
RUN pip install poetry

RUN poetry config virtualenvs.create false

COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --no-cache --only main

COPY ./configs ./configs
COPY ./run ./run
COPY ./Makefile Makefile
COPY ./gnn_180b gnn_180b
RUN poetry install --no-cache --only-root
