<h1 align="center">
GraphHSCN: Heterogenized Spectral Cluster Network for Long Range Representation Learning</h1>
<div align="center">

  <a href="https://camille-004.github.io/">Camille Dunning</a>, <a href="https://www.linkedin.com/in/zhishang-luo-a51a8120b/">Zhishang Luo</a>, <a href="https://dylantao.github.io/">Sirui Tao</a>
  <p><a href="https://datascience.ucsd.edu/">Halıcıoğlu Data Science Institute</a>, UC San Diego, La Jolla, CA</p>

</div>
<hr />
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://graphhscn.github.io/)

<!-- [![Paper (First Draft)](https://img.shields.io/badge/Project-Paper-9cf)](https://drive.google.com/file/d/1kODg7Qw4hAj1e2Ct91R_tvom8MHdeGln/view) -->

## :rocket: Highlights and Contributions

TODO: Flowchart figure

>**<p align="justify"> Abstract:** *Graph Neural Networks (GNNs) have gained tremendous popularity for their potential to effectively learn from graph-structured data, commonly encountered in real-world applications. However, most of these models, based on the message-passing paradigm (interactions within a neighborhood of a few nodes), can only handle local interactions within a graph. When we enforce the models to use information from far away nodes, we will encounter two major issues — oversmoothing & oversquashing. Architectures such as the transformer and diffusion models are introduced to solve this; although transformers are powerful, they require significant computational resources for both training and inference, thereby limiting their scalability, particularly for graphs with long-term dependencies. Hence, this paper proposes GraphHSCN—a Heterogenized Spectral Cluster Network, a message-passing-based approach specifically designed for capturing long-range interaction. On our first iteration of ablation studies, we observe reduced time complexities compared to SAN, the most popular graph transformer model, yet comparable performance in graph-level prediction tasks.*

## Project Setup 
1. To set up the environment and install all dependencies, run `make env`.
2. Create a `datasets` directory at the project level. This is where the loader will save downloaded datasets.
  
### `.devcontainers` Support
TODO
### Running with CLI
TODO
### Running in Prefect UI
TODO
  
## Contact
TODO
  
## Acknowledgements
TODO
