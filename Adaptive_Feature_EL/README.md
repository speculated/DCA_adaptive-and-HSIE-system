Improving entity linking with two adaptive features, Frontiers of Information Technology & Electronic Engineering, 2022

Our code is based on the code from https://github.com/YoungXiyuan/DCA

# Datasaet
Download data from here and unzip to the main folder (i.e. your-path/DCA).

The above data archive mainly contains the following resource files:

Dataset: One in-domain dataset (AIDA-CoNLL) and Five cross-domain datasets (MSNBC / AQUAINT / ACE2004 / CWEB / WIKI). And these datasets share the same data format.

Mention Type: Adopted to compute type similarity between mention-entity pairs. We predict types for each mention in datasets using a typing system called NFETC model trained by the AIDA dataset.

Wikipedia inLinks: Surface names of inlinks for a Wikipedia page (entity) are used to construct dynamic context in our model learning process.

Entity Description: Wikipedia page contents (entity description) are used by one of our base model -- Berkeley-CNN

# Installation
Requirements: Python 3.6.5, Pytorch 1.6.0,Numpy 1.19.1,  CUDA 10.1 or 10.2
