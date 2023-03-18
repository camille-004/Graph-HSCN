FROM python:3.10

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /graph_hscn

# Install pip requirements
RUN python -m pip install -U pip setuptools wheel
COPY ./Makefile ./requirements-cpu.txt ./requirements-gpu.txt ./requirements-core.txt ./setup.py ./
RUN make env

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "-m", "run.py"]
