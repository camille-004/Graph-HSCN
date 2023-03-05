FROM python:3.10

COPY ./configs ./configs
COPY ./run ./run
COPY graph_hscn ./graph_hscn
COPY run.py .
COPY Makefile Makefile
COPY pyproject.toml .
COPY README.md .

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-cache --no-root --no-dev

ENTRYPOINT [ "make" ]
