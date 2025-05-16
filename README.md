# HLoG-VBGMM 

## Requirements: 

- python >= 3.11.5
- libraries: matplotlib, nibabel, numpy, pandas, scikit-image, scikit-learn, SimpleITK

## Description: 

HLoG-VBGMM is a state-of-the-art blob detector based on Laplacian of Gaussian (LoG), Hessian analysis, and Variational Bayesian Gaussian Mixture Model (VBGMM). This model was previously used by K. Bennett _et al._ to count glomeruli in contrast-enhanced MRI (CE-MRI).

Here, for the first time, the model was used to estimate the glomerular number and density of human _ex vivo_ kidney samples from ultra high-field MRI (16.4T) acquired at 30-micron isotropic resolution. 

## Dataset:

The dataset consists of six three-dimensional MRI acquired at 30-micron isotropic resolution, obtained from healthy portions of 6 nephroctomy samples from 4 distinct patients, and 6 masks delineating the tissue from the background. 

The MRIs and masks used for this project, along with the glomerular segmentation masks can be downloaded here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15428478.svg)](https://doi.org/10.5281/zenodo.15428478)

## Project Structure 

```
HLoG_VBGMM/
│
│
├── input/ 
│ ├── X_mri.npy 
│ └── X_mask.npy 
│
├── output/
│
├── python/ 
│ ├── __init__.py
│ └── arguments.py
│ └── main.py
│ └── utils.py 
│
├── HLoG_notebook.ipynb # Example notebook
│
├── README.md # Project overview and instructions
└── requirements.txt # Python dependencies
```

- **`input/`**
  Replace `X` with the sample identifier, such as `A`, `B`, `C`, `D`, or `E` (e.g. for sample A: `A_mri.npy` and `A_mask.npy`).

- **`output/`**
   Empty folder to save the segmentation masks. 

## Run the model:

To run the model on a sample (e.g. A, B, C, D, or E), use the command prompt:

```
python main.py X
```
Replace X with the sample identifier (e.g. A, B, C, D, or E).




