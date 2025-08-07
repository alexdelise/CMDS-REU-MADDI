# Model-Aware and Data-Driven Inference

Contributors: [Alexander DeLise](https://www.linkedin.com/in/alexanderdelise/), [Kyle Loh](https://www.linkedin.com/in/kyle-loh-a2a3272a9/), [Krish Patel](https://www.linkedin.com/in/krish-patel-1a8804224/), [Meredith Teague](https://www.linkedin.com/in/meredithcteague/)

Advisors: Andrea Arnold, Matthias Chung

This repository was made for the "Model-Aware and Data-Driven Inference" project from the Summer 2025 "Model Meets Data REU" in Emory University's Math Department. More information can be found on the [REU website](https://www.math.emory.edu/site/cmds-reuret/summer2025/).

This project is partially supported by NSF DMS-2349534. 

# Instructions
## Basic Demo
The `demo.py` script provides code to compute the theoretically optimal mappings for the forward and inverse end-to-end problems, as well as for autoencoding and denoising. It generates a random input data matrix $\mathbf{X}$ and constructs the corresponding observation matrix $\mathbf{Y}$ using a rank-deficient forward operator. 

To generate affine linear mappings instead, set `affine = True`.


## Biomedical Imaging with $\texttt{MedMNIST}$
To get started, install the $\texttt{MedMNIST}$ dataset package by running:
```python
pip install medmnist
```
More details about the $\texttt{MedMNIST}$ dataset can be found on the [official website](https://medmnist.com/). 

We provide separate `Python` notebooks for the following problem formulations:
- General forward and inverse end-to-end mappings
- Autoencoding
- Data denoising
- Each of the above, along with their affine linear counterparts

Each notebook will generate:
- A representative error comparison sample
- A rank sweep plot

For example:

<div align="center"> <img src="README-Pics/classic_chestmnist_mapping7181_errorcomparison.png" alt="errorSample" width="450"/> </div> <div align="center"> <img src="README-Pics/classic_ranksweep_200ep.png" alt="rankSweep" width="450"/> </div>

We use `PyTorch` to run all experiments. If you have an NVIDIA GPU, computations will automatically utilize it for acceleration. The results, including pickle files of learned and optimal mappings at various ranks, are stored in organized subdirectories.

In the `SpecialErrorComparisonPlot` folder, we include scripts to generate side-by-side comparison plots for each problem type, with their affine linear variants. For example:

<div align="center"> <img src="README-Pics/aeFullComparison.png" alt="errorSampleLinAffLin" width="600"/> </div>


## Financial

## Shallow Water Equations


# Relevant Links
- Our poster can be found [here](https://drive.google.com/file/d/1kZ1RPy-E8zGCxs_8ntEbNDc42YKNFbQ0/view?usp=drive_link).

- Our ArXiV manuscript can be found here
