# VaccineSimulator
Project for the 2024-2025 single cell lecture at TUM, where we will predict vaccination response of individual clones using temporal multimodal single cell data.

## Set up
Clone this repository via:

```
git clone git@github.com:SchubertLab/VaccineSimulator.git
```

access the repository directory and run the following commands to create a conda environment with all initial requirements:

```
conda env create -f environment.yaml -y && conda activate VaccineSimulator && conda install nb_conda_kernels -y
```

You can find an introduction to the dataset and project under notebooks/introduction.ipynb
