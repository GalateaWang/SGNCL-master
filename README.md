# Subgraph Networks Based Contrastive Learning

## Introduction

Graph contrastive learning (GCL), as a self-supervised learning method, can solve the problem of annotated data scarcity. It mines explicit features in unannotated graphs to generate favorable graph representations for downstream tasks.  Most existing GCL methods focus on the design of graph augmentation strategies and mutual information estimation operations. Graph augmentation produces augmented views by graph perturbations. These views preserve a locally similar structure and exploit explicit features.  However, these methods have not considered the interaction existing in subgraphs. To explore the impact of substructure interactions on graph representations, we propose a novel framework called subgraph network-based contrastive learning (SGNCL). SGNCL applies a subgraph network generation strategy to produce augmented views. This strategy converts the original graph into an Edge-to-Node mapping network with both topological and attribute features. The single-shot augmented view is a first-order subgraph network that mines the interaction between nodes, node-edge, and edges. In addition, we also investigate the impact of the second-order subgraph augmentation on mining graph structure interactions, and further, propose a contrastive objective that fuses the first-order and second-order subgraph information.  We compare SGNCL with classical and state-of-the-art graph contrastive learning methods on multiple benchmark datasets of different domains. Extensive experiments show that SGNCL achieves competitive or better performance (top three) on all datasets in unsupervised learning settings.  Furthermore, SGNCL achieves the best average gain of 6.9\% in transfer learning compared to the best method. Finally, experiments also demonstrate that mining substructure interactions have positive implications for graph contrastive learning.

![image-20230416220512529](C:\Users\62406\AppData\Roaming\Typora\typora-user-images\image-20230416220512529.png)

## Experiments

- Unsupervised representation learning on TU datasets

- Transfer Learning on MoleculeNet datasets

# SGNCL-master
