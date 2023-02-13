# gnn_DSC180B

[![Publish Docker image](https://github.com/camille-004/gnn_DSC180B/actions/workflows/docker-image.yml/badge.svg)](https://github.com/camille-004/gnn_DSC180B/actions/workflows/docker-image.yml)

This project is built on [poetry](https://python-poetry.org/) for dependency management and packaging.

## Usage
The config files available for experiments are available in the `configs` directory. To run a repeated baseline GCN on the resampled Cora dataset, run the following:

```bash
python run.py --cfg configs/GCN/cora_GCN.yaml --repeats 3
```

**For the Week 5 checkpoint, simply run the following command in the root directory to set up your environment and run baseline models:**
```bash
make
```
The upload to DockerHub is currently failing due to Python version conflicts with poetry. ETS is currently working on fixing this.

## Acknowledgements
The code in this project is heavily adapted and modified from the following repositories:
1. [Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb)
2. [torch_geometric GraphGym](https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym)
3. [Hierarchical Graph Net](https://github.com/rampasek/HGNet)
