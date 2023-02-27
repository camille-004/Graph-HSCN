FROM python:3.10

COPY ./configs ./configs
COPY ./run ./run
COPY ./Makefile Makefile
COPY ca_net gnn_180b
COPY pyproject.toml .
COPY README.md .

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-cache --only-root --no-dev

ENTRYPOINT [ "make" ]
