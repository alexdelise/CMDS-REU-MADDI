# Model-Aware and Data-Driven Inference

Contributors: [Alexander DeLise](https://www.linkedin.com/in/alexanderdelise/), [Kyle Loh](https://www.linkedin.com/in/kyle-loh-a2a3272a9/), [Krish Patel](https://www.linkedin.com/in/krish-patel-1a8804224/), [Meredith Teague](https://www.linkedin.com/in/meredithcteague/)

Advisors: Andrea Arnold, Matthias Chung

This repository was made for the "Model-Aware and Data-Driven Inference" project from the Summer 2025 "Model Meets Data REU" in Emory University's Math Department. More information can be found on the [REU website](https://www.math.emory.edu/site/cmds-reuret/summer2025/).

This project is partially supported by NSF DMS-2349534. 

# Instructions
## Basic Demo
We provide code that computes the theoretically optimal mappings for the forward and inverse end-to-end problems, as well as for autoencoding and denoising in `demo.py`. This code generates a random input data matrix $\mathbf{X}$ and generates an observation matrix $\mathbf{Y}$ via a rank-deficient forward operator. For generating affine linear mappings, make `affine = True`.

## Biomedical Imaging with $\texttt{MedMNIST}$
Begin by installing the $\texttt{MedMNIST}$ datasets via the command 
```python
pip install medmnist
```
For more information on the $\texttt{MedMNIST}$ dataset, click [here](https://medmnist.com/). 

We provide a `python` notebook for each of general forward and inverse end-to-end problems, autoencoding, and data denoising, as well as their affine linear counterparts. Running each `python` notebook will produce the representative error sample

![errorSample](README-Pics/classic_chestmnist_mapping7181_errorcomparison.png)

as well as the rank sweep plot

![rankSweep](README-Pics/classic_ranksweep_200ep.png)

We use `PyTorch` to run our experiments, thus if you have an NVIDIA GPU, experiments will be run on there. Results are stored in corresponding subfolders, including `pickle` files that contain optimal and learned mappings for each tested rank.


Within the `SpecialErrorComparisonPlot` folder, you can find code that produces representative error sample figures for each problem formualtion and their affine linear counterpart, such as in the picture below. 


## Financial

## Shallow Water Equations


# Relevant Links
- Our poster can be found [here](https://drive.google.com/file/d/1kZ1RPy-E8zGCxs_8ntEbNDc42YKNFbQ0/view?usp=drive_link).

- Our ArXiV manuscript can be found here
