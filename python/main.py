import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import os
from utils import get_index_matrix_3D, get_mean_value, get_R_value, get_S_value
from scipy import ndimage
from skimage.measure import label, regionprops_table
from sklearn.mixture import BayesianGaussianMixture
from arguments import args


def main():
    ## 1 - data preparation
    # read the data (stored in nii.gz format):
    mri = sitk.ReadImage(args.mri_path)
    mask = sitk.ReadImage(args.mask_path)

    # convert into numpy arrays
    mri = sitk.GetArrayFromImage(mri)
    mask = sitk.GetArrayFromImage(mask)

    # use matlab format
    Nz, Ny, Nx = mri.shape

    # normalisation
    mri /= np.max(mri)

    # parameters:
    resolution = args.resolution
    factor = resolution / 30 
    radius = 2.5 / factor
    radius_min = 2.0 / factor
    radius_max = 6.0 / factor
    log_sigma = radius / np.sqrt(3)
    vol_min = (4 / 3) * np.pi * radius_min**3 
    vol_max = (4 / 3) * np.pi * radius_max**3

    ## 2 - optimal scale identification 
    # lists to store the results: 
    list_sigma = []
    list_avg_score = []

    # parameters
    BLoG_best = -100 
    n_iter = args.n_iter
    step = args.step_sigma
    min_sigma = log_sigma
    max_sigma = min_sigma + n_iter * step 

    # iterate through sigma values
    i = min_sigma 
    while (i <= max_sigma):
        # LoG filtering
        LoG_temp =  ndimage.gaussian_laplace(mri, sigma=i)

        # normalisation of LoG_temp in [-1 ; 1] for fair comparaison
        xmin = np.min(LoG_temp)
        xmax = np.max(LoG_temp)
        LoG_temp =  2 * ((LoG_temp - xmin) / (xmax - xmin)) - 1

        # get the binary index matrix (convexity map) from Hessian analysis
        I_temp, lambda1_temp, lambda2_temp, lambda3_temp = get_index_matrix_3D(im=LoG_temp, mask=mask)
        
        # calculate the average pixel value
        BLoG = (np.sum(LoG_temp * I_temp)) / np.sum(I_temp)

        # compare results and update values if the score is improved
        if BLoG > BLoG_best:
            BLoG_best = BLoG
            I = I_temp
            LoG_norm = LoG_temp 
            lambda1, lambda2, lambda3 = lambda1_temp, lambda2_temp, lambda3_temp
        
        # store the results 
        list_sigma.append(i)
        list_avg_score.append(BLoG)

        # Update sigma value
        i += step

    ## 3 - features extraction and size filtering
    # labelisation    
    labels = label(I)
    list_properties = ['area','bbox', 'label', 'centroid', 'EquivDiameter', 'axis_major_length', 'inertia_tensor_eigvals', 'coords']
    
    # data frame of blob candidates
    df = pd.DataFrame(regionprops_table(labels, properties = list_properties))

    # remove the blobs that are too small or too big
    df = df[(df['area'] <= vol_max) & (df['area'] >= vol_min)] 
    df = df.reset_index(drop=True)

    # add S, R, and mean_intensity columns:
    df.insert(1, "S", '')
    df.insert(2, "R1", '') # To measure abs(lamba3) / abs(lambda1)
    df.insert(3, "R2", '') # To measure abs(lamba2) / abs(lambda1)
    df.insert(4, "S_max", 0)
    df.insert(5, "R1_max", 0)
    df.insert(6, "R2_max", 0)
    df.insert(7, "S_mean", 0)
    df.insert(8, "R1_mean", 0)
    df.insert(9, "R2_mean", 0)
    df.insert(10, "mean_intensity", 0)
    for i in range(len(df)):

        # S features:
        df["S"][i] = get_S_value(lambda1, lambda2, lambda3, df["coords"][i])
        df["S_max"][i] = np.max(df["S"][i])
        df["S_mean"][i] = np.mean(df["S"][i])

        # R1 features:
        df["R1"][i] = get_R_value(lambda3, lambda1, df["coords"][i])
        df["R1_max"][i] = np.max(df["R1"][i])
        df["R1_mean"][i] = np.mean(df["R1"][i])
        
        # R2 features: 
        df["R2"][i] = get_R_value(lambda2, lambda1, df["coords"][i])
        df["R2_max"][i] = np.max(df["R2"][i])
        df["R2_mean"][i] = np.mean(df["R2"][i])
        
        # mean_intensity feature:
        df["mean_intensity"][i] = get_mean_value(LoG_norm, df["coords"][i])

    # FA (fractional anisotropy):
    lambda_0 = df['inertia_tensor_eigvals-0']
    lambda_1 = df['inertia_tensor_eigvals-1']
    lambda_2 = df['inertia_tensor_eigvals-2']
    FA = np.sqrt(1 / 2) * np.sqrt((lambda_0 - lambda_1)**2 + (lambda_1 - lambda_2)**2 + (lambda_2 - lambda_0)**2) / np.sqrt(lambda_0**2 + lambda_1**2 + lambda_2**2)
    df.insert(11, "FA", FA)

    # add std_eigvalues column to dataframe
    std_eigvalues = np.std([df['inertia_tensor_eigvals-0'], df['inertia_tensor_eigvals-1'], df['inertia_tensor_eigvals-2']], axis=0)
    df.insert(12, 'std_eigvalues', std_eigvalues)

    # Number of blob candidates
    count_blob_candidates = len(df)

    # generate a mask for blob candidates:
    mask_blob = np.zeros(mri.shape)
    for i in range(count_blob_candidates):
        for j in range(len(df.coords[df.index[i]])):
            x, y, z = df.coords[df.index[i]][j]
            mask_blob[(x, y, z)] = 1

    ## 4 - VBGMM for clustering        
    vbgmm = BayesianGaussianMixture(n_components=2, 
                     covariance_type='full', 
                     random_state=0).fit(df[["area", "S_max", "R1_max"]])
    
    # Prediction TP or FP:
    labels_vbgmm = vbgmm.predict(df[["area", "S_max", "R1_max"]])
    array_vbgmm = np.reshape(labels_vbgmm, (len(labels_vbgmm), 1))

    # Add labels to dataframe:
    df[["labels_vbgmm"]]= array_vbgmm

    # score: 
    print('Total number of blob candidates:')
    print(count_blob_candidates)
    print('Mean values of VBGMM')
    print(vbgmm.means_)

    # we keep the cluster with the highest value of R
    if vbgmm.means_[0, 2] > vbgmm.means_[1, 2]:
        n_glom = count_blob_candidates - np.sum(array_vbgmm)
        n_non_glom = np.sum(array_vbgmm)
        df_glom = df[df["labels_vbgmm"] == 0]
        df_non_glom = df[df["labels_vbgmm"] == 1]
    else:
        n_glom = np.sum(array_vbgmm)
        n_non_glom = count_blob_candidates - np.sum(array_vbgmm)
        df_glom = df[df["labels_vbgmm"] == 1]
        df_non_glom = df[df["labels_vbgmm"] == 0]

    print("Total number of glomeruli:")
    print(n_glom)

    # create masks for each cluster 
    mask_glom = np.zeros(mri.shape)
    mask_non_glom = np.zeros(mri.shape)
    for i in range(n_glom):
        for j in range(len(df_glom.coords[df_glom.index[i]])):
            x, y, z = df_glom.coords[df_glom.index[i]][j]
            mask_glom[(x, y, z)] = 1
    for i in range(n_non_glom):
        for j in range(len(df_non_glom.coords[df_non_glom.index[i]])):
            x, y, z = df_non_glom.coords[df_non_glom.index[i]][j]
            mask_non_glom[(x, y, z)] = 1
    
    # store the masks in nifti format:
    glom_nifti = nib.Nifti1Image(mask_glom.swapaxes(0,2), affine=np.eye(4))
    non_glom_nifti = nib.Nifti1Image(mask_non_glom.swapaxes(0,2), affine=np.eye(4))
    nib.save(glom_nifti, os.path.join("../results/glom.nii.gz"))
    nib.save(non_glom_nifti, os.path.join("../results/non_glom.nii.gz"))
   
    
if __name__ == "__main__":
    main()
