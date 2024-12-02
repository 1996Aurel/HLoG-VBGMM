import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import nibabel as nib
import os
from utils import get_index_matrix_3D, get_mean_value, get_R_value, get_S_value, z_score_normalisation, optimale_scale_identification, hessian, get_Ht
from scipy import ndimage
from skimage.measure import label, regionprops_table
from sklearn.mixture import BayesianGaussianMixture
from arguments import args


def main():
    ## data preparation
    # read the data (stored in nii.gz format):
    sample = args.sample # A, B, C, D, E or F

    mri = sitk.ReadImage("./input/" + sample + "_mri.nii.gz")
    mask = sitk.ReadImage("./input/" + sample + "_mask.nii.gz")

    # convert into numpy arrays
    mri = sitk.GetArrayFromImage(mri)
    mask = sitk.GetArrayFromImage(mask)

    # use matlab format
    Nz, Ny, Nx = mri.shape

    # normalisation
    mri = z_score_normalisation(mri)

    # parameters:
    resolution = 30
    factor = resolution / 30 
    radius = 2.5 / factor
    radius_min = 2.0 / factor
    radius_max = 5.0 / factor
    log_sigma = radius / np.sqrt(3)
    vol_min = (4 / 3) * np.pi * radius_min**3 
    vol_max = (4 / 3) * np.pi * radius_max**3

    ## optimale scale identification
    _, _, LoG_norm, I, lambda1, lambda2, lambda3 = optimale_scale_identification(mri, mask)

    # Hessian: 
    gxx, gxy, gxz, gyy, gyz, gzz = hessian(LoG_norm)

    ## Features extraction 
    labels = label(I)
    list_properties = ['area','bbox', 'label', 'centroid', 'EquivDiameter', 'axis_major_length', 'inertia_tensor_eigvals', 'coords']
    df = pd.DataFrame(regionprops_table(labels, properties = list_properties))

    # remove the blobs that are too small or too big
    df = df[(df['area'] <= vol_max) & (df['area'] >= vol_min)] 
    df = df.reset_index(drop=True)

    # add S, R, and mean_intensity columns: (we have L1 < L2 < L3 < 0 ie |L3| < |L2| < |L1|)
    df.insert(1, "S", '')
    df.insert(2, 'Rb', '') #  To measure L3 / sqrt(L2 * L1)
    df.insert(3, "R1", '') # To measure L3 / L1
    df.insert(4, "Ra", '') # To measure L2 / L1
    df.insert(5, "S_max", 0)
    df.insert(6, "Rb_max", 0)
    df.insert(7, "R1_max", 0)
    df.insert(8, "Ra_max", 0)
    df.insert(9, "S_mean", 0)
    df.insert(10, "Rb_mean", 0)
    df.insert(11, "R1_mean", 0)
    df.insert(12, "Ra_mean", 0)
    df.insert(13, "mean_intensity", 0)
    df.insert(14, "Ht", '')
    df.insert(15, "L1", 0)
    df.insert(16, "L2", 0)
    df.insert(17, "L3", 0)
    df.insert(18, "Rb_t", 0)
    df.insert(19, "R1_t", 0)
    df.insert(20, "Ra_t", 0)
    df.insert(21, "S_t", 0)

    denominator_Rb = np.sqrt(np.abs(lambda2*lambda1))

    for i in range(len(df)):
        # S features:
        df["S"][i] = get_S_value(lambda1, lambda2, lambda3, df["coords"][i])
        df["S_max"][i] = np.max(df["S"][i])
        df["S_mean"][i] = np.mean(df["S"][i])
        # Rb features:
        df["Rb"][i] = get_R_value(lambda3, denominator_Rb, df["coords"][i])
        df["Rb_max"][i] = np.max(df["Rb"][i])
        df["Rb_mean"][i] = np.mean(df["Rb"][i])
        # R1 features:
        df["R1"][i] = get_R_value(lambda3, lambda1, df["coords"][i])
        df["R1_max"][i] = np.max(df["R1"][i])
        df["R1_mean"][i] = np.mean(df["R1"][i])
        # R2 features:
        df["Ra"][i] = get_R_value(lambda2, lambda1, df["coords"][i])
        df["Ra_max"][i] = np.max(df["Ra"][i])
        df["Ra_mean"][i] = np.mean(df["Ra"][i])
        # mean_intensity feature:
        df["mean_intensity"][i] = get_mean_value(LoG_norm, df["coords"][i])
        # regional features from Ht:
        df["Ht"][i] = get_Ht(gxx, gxy, gxz, gyy, gyz, gzz, df["coords"][i])
        # eigenvalues of Ht:
        df["L1"][i], df["L2"][i], df["L3"][i] = np.linalg.eigvalsh(df["Ht"][i])
        # Regional blob features:
        df["Rb_t"][i] = np.abs(df["L3"][i]) / np.sqrt(df["L1"][i] * df["L2"][i]) 
        df["R1_t"][i] = np.abs(df["L3"][i]) / np.abs(df["L1"][i]) 
        df["Ra_t"][i] = np.abs(df["L2"][i]) / np.abs(df["L1"][i])
        df["S_t"][i] = np.sqrt(df["L1"][i]**2 + df["L2"][i]**2 + df["L3"][i]**2)


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
                     random_state=0).fit(df[["area", "S_max", "Rb_t"]])
    
    # Prediction TP or FP:
    labels_vbgmm = vbgmm.predict(df[["area", "S_max", "Rb_t"]])
    array_vbgmm = np.reshape(labels_vbgmm, (len(labels_vbgmm), 1))

    # Add labels to dataframe:
    df[["labels_vbgmm"]]= array_vbgmm

    # score: 
    print('Total number of blob candidates:')
    print(count_blob_candidates)

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
    nib.save(glom_nifti, os.path.join("./results/" + sample + "_glom.nii.gz"))
    nib.save(non_glom_nifti, os.path.join("./results/" + sample + "_non_glom.nii.gz"))
   
    
if __name__ == "__main__":
    main()