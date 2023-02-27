FROM python:3.10

COPY ./configs ./configs
COPY ./run ./run
COPY ./ca_net ./ca_net
COPY run.py .
COPY Makefile Makefile
COPY pyproject.toml .
COPY README.md .

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-cache --only-root --no-dev

ENTRYPOINT [ "make" ]
