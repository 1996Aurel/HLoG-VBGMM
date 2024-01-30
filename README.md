# HLoG-VBGMM 

## Requirements: 

- python >= 3.11.5
- libraries: numpy, matplotlib, pandas, skimage, scipy, os, argparse, SimpleITK, nibabel

## Description: 

HLoG-VBGMM is a state-of-the-art blob detector based on Laplacian of Gaussian (LoG), Hessian analysis, and Variational Bayesian Gaussian Mixture Model (VBGMM). This model was previously used by K. Bennett _et al._ to count glomeruli in contrast-enhanced MRI (CE-MRI).

Here, for the first time, the model was used to estimate the glomerular number and density of human _ex-vivo_ kidney samples from ultra high-field MRI (16.4T) acquired at 30, 60, and 100-micron. 

## Dataset:

* seven three-dimensional MRI acquired at 30, 60, and 100 micron isotropic resolution
* seven masks delineating the cortex and medulla on each kidney samples

## Run the model:

```
python main.py 'path_to_mri' 'path_to_mask_cortex_medulla' resolution --n_iter n --step_sigma step
```

## Reults

The figure below shows the different steps of the segmentation to obtain the blob candidates. The first image is an orignal MRI of a kidney sample acquired at 60 micron. The second image is the LoG filtered image at the optimum scale sigma. For details on how the parameter sigma is optimised, please refer to our paper. The third image represents the concave elliptical structure (potential glomeruli) detected by Hessian analysis. Finally, the last image shows the plausible blob candidates (in yellow), ie. the ones with the right size, on the original sample. 

![plot](./figures/fig1.png)


The following figure allows to visualise the performance of the clustering with the Variational Bayesian Gaussian Mixture Model. On the left, we have all the blob candidates (in yellow) on the original MRI. In the middle, we have the final segmentation with all the glomeruli, and on the right, all the other blobs. 

![plot](./figures/fig2.png)


