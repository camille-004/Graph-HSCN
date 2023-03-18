ENV_NAME := graph_hscn
REQ_FILE := requirements-gpu.txt

# Set the requirements file to use if GPU is available
ifeq ($(shell which nvcc),)
	REQ_FILE := requirements-cpu.txt
else
	REQ_FILE := requirements-gpu.txt
endif

.PHONY: env
env:
	python3 -m pip install -U pip setuptools wheel
	echo "Installing requirements from $(REQ_FILE)"
	python3 -m pip install -r $(REQ_FILE)
