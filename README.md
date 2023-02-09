# gnn_DSC180B

This project is built on [poetry](https://python-poetry.org/) for dependency management and packaging.

## Usage
The config files available for experiments are available in the `configs` directory. To run a repeated baseline GCN on the Cora dataset, run the following:

```bash
python.main.py --cfg configs/gcn/peptides_func_GCN.yaml --repeats 3
```

## Acknowledgements
The code in this project is heavily adapted (and modified) from the following repositories:
1. [Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb)
2. [torch_geometric GraphGym](https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym)
3. [Hierarchical Graph Net](https://github.com/rampasek/HGNet)