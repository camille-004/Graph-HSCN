<h1 align="center">
GraphHSCN: Heterogenized Spectral Cluster Network for Long Range Representation Learning</h1>
<div align="center">

  <a href="https://camille-004.github.io/">Camille Dunning</a>, <a href="https://www.linkedin.com/in/zhishang-luo-a51a8120b/">Zhishang Luo</a>, <a href="https://dylantao.github.io/">Sirui Tao</a>
  <p><a href="https://datascience.ucsd.edu/">Halıcıoğlu Data Science Institute</a>, UC San Diego, La Jolla, CA</p>
</div>

<p align="center">
  <a href="https://drive.google.com/file/d/1kODg7Qw4hAj1e2Ct91R_tvom8MHdeGln/view" alt="Paper">
        <img src="https://img.shields.io/badge/Project-Paper-%238affca?style=plastic" /></a>
        
  <a href="https://graphhscn.github.io//" alt="Website">
        <img src="https://img.shields.io/badge/Project-Website-%238affca?style=plastic" /></a>
        
  <a href="https://github.com/camille-004/Graph-HSCN/actions/workflows/build-and-push.yml" alt="Build">
        <img src="https://github.com/camille-004/Graph-HSCN/actions/workflows/build-and-push.yml/badge.svg" /></a>

</p>
<hr/>


<!-- [![Paper (First Draft)](https://img.shields.io/badge/Project-Paper-9cf)](https://drive.google.com/file/d/1kODg7Qw4hAj1e2Ct91R_tvom8MHdeGln/view) -->

## :rocket: Highlights and Contributions

TODO: Flowchart figure

>**<p align="justify"> Abstract:** *Graph Neural Networks (GNNs) have gained tremendous popularity for their potential to effectively learn from graph-structured data, commonly encountered in real-world applications. However, most of these models, based on the message-passing paradigm (interactions within a neighborhood of a few nodes), can only handle local interactions within a graph. When we enforce the models to use information from far away nodes, we will encounter two major issues — oversmoothing & oversquashing. Architectures such as the transformer and diffusion models are introduced to solve this; although transformers are powerful, they require significant computational resources for both training and inference, thereby limiting their scalability, particularly for graphs with long-term dependencies. Hence, this paper proposes GraphHSCN—a Heterogenized Spectral Cluster Network, a message-passing-based approach specifically designed for capturing long-range interaction. On our first iteration of ablation studies, we observe reduced time complexities compared to SAN, the most popular graph transformer model, yet comparable performance in graph-level prediction tasks.*

### Main Contributions
1. **Graph coarsening via spectral clustering**: We propose a scheme to coarsen graph representation via spectral clustering with the relaxed formulation of the MinCUT problem, as presented in the [paper](https://arxiv.org/abs/1907.00481) from Bianchi et. al. We observe the structural patterns uncovered by SC reveal which long-range virtual connections should be made.
2. **New connections learned by a heterogeneous network**: We create an intra-cluster connection with a virtual node, and learn the new relationship as a graph indepdenent of the original graph. A heterogeneous convolutional network is trained on these separate relations, further coarsening the representations. On our set of ablation studies, and after hyperparameter tuning, Graph-HSCN out-performs the traditional message-passing architectures by up to 10 percent, achieving metrics similar to those of SAN while reducing the time complexity.

## Getting Started

### Prerequisites
1. To set up the environment and install all dependencies, run `make env`.
2. Create a `datasets` directory at the project level. This is where the loader will save downloaded datasets.
  
### `.devcontainers` Support
TODO
### Running with CLI
TODO
### Running in Prefect UI
TODO

<hr/>

## Results
TODO

<hr/>

## Contact
Feel free to open an issue on this repository or e-mail adunning@ucsd.edu.
  
## Acknowledgements
The code in this project is heavily adapted and modified from the following repositories:
1. [Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb)
2. [torch_geometric GraphGym](https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym)
3. [Hierarchical Graph Net](https://github.com/rampasek/HGNet)
