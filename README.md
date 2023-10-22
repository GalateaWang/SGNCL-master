# SGNCL-master 
![github](https://img.shields.io/badge/github-GalateaWang-brightgreen.svg) ![thanks coauthor](https://img.shields.io/badge/thankscoauthor-JiafeiShao-green.svg)  ![watchers](https://img.shields.io/github/watchers/galateawang/SGNCL-master) ![Github stars](https://img.shields.io/github/stars/GalateaWang/SGNCL-master.svg)

This is a project for the paper "Subgraph Networks Based Contrastive Learning".

![SGNCL](https://github.com/GalateaWang/PaperFigs/blob/bb8da30843a0216f57f858876f977717410ca0ee/Edge-to-Node.png)
Graph contrastive learning (GCL), as a self-supervised learning method, can solve the problem of annotated data scarcity. It mines explicit features in unannotated graphs to generate favorable graph representations for downstream tasks.  Most existing GCL methods focus on the design of graph augmentation strategies and mutual information estimation operations. Graph augmentation produces augmented views by graph perturbations. These views preserve a locally similar structure and exploit explicit features.  However, these methods have not considered the interaction existing in subgraphs. To explore the impact of substructure interactions on graph representations, we propose a novel framework called subgraph network-based contrastive learning (SGNCL). SGNCL applies a subgraph network generation strategy to produce augmented views. This strategy converts the original graph into an Edge-to-Node mapping network with both topological and attribute features. The single-shot augmented view is a first-order subgraph network that mines the interaction between nodes, node-edge, and edges. In addition, we also investigate the impact of the second-order subgraph augmentation on mining graph structure interactions, and further, propose a contrastive objective that fuses the first-order and second-order subgraph information.  We compare SGNCL with classical and state-of-the-art graph contrastive learning methods on multiple benchmark datasets of different domains. Extensive experiments show that SGNCL achieves competitive or better performance (top three) on all datasets in unsupervised learning settings.  Furthermore, SGNCL achieves the best average gain of 6.9\% in transfer learning compared to the best method. Finally, experiments also demonstrate that mining substructure interactions have positive implications for graph contrastive learning.


## Experiments

- Unsupervised representation learning on TU datasets

- Transfer Learning on MoleculeNet datasets

## Environment

- dgl                           0.7.2
- dive-into-graphs              0.1.2
- gensim                        3.6.0
- keras                         2.7.0
- networkx                      2.6.3
- numpy                         1.19.5
- ogb                           1.3.3
- pandas                        1.3.4
- PyGCL                         0.1.1
- torch                         1.9.0
- torch-cluster                 1.5.9
- torch-geometric               1.7.0
- torch-scatter                 2.0.9
- torch-sparse                  0.6.12
- rdkit-pypi                    2022.3.2.1
