# HLoG-VBGMM 

## Requirements: 

- python >= 3.11.5
- libraries: numpy, matplotlib, pandas, skimage, scipy, os, argparse, SimpleITK, nibabel

## Description: 

HLoG-VBGMM is a state-of-the-art blob detector based on Laplacian of Gaussian (LoG), Hessian analysis, and Variational Bayesian Gaussian Mixture Model (VBGMM). This model was previously used by K. Bennett _et al._ to count glomeruli in contrast-enhanced MRI (CEMRI).

Here, for the first time, we use this model to estimate the glomerular number from ultra high-field MRI (16.4T), acquired at 30, 60, and 100-micron. 

## Dataset:

The model was tested with seven three-dimensional MRI of _ex-vivo_ human kidney samples acquired at 30, 60, and 100 $\mu$m isotropic resolution. 

## Run the model:

```
python main.py 'path_to_mri' 'path_to_mask_cortex_medula' resolution --n_iter n --step_sigma step
```

## Reults
