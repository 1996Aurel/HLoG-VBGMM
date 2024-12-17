import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def LoG_kernel_3D(sigma):
    '''Function to create a LoG kernel in 3D
        input:  - sigma: kernel size
        output: - LoG kernel of size 6 * sigma'''
    # kernel size
    n = np.ceil(6 * sigma)
    z, y , x = np.ogrid[-n//2 : n//2+1, -n//2 : n//2+1, -n//2 : n//2+1]
    z_filter = np.exp(-(z**2 / (2. * sigma**2)))
    y_filter = np.exp(-(y**2 / (2. * sigma**2)))
    x_filter = np.exp(-(x**2 / (2. * sigma**2)))
    kernel = (-3 * sigma**2 + x**2 + y**2 + z**2) * x_filter * y_filter * z_filter * (1 / (sigma**7 * (2 * np.pi)**(3/2)))
    return kernel


def mask_without_edges(im, sigma):
    '''return a mask to filter the peaks near edges
        input:      - im
                    - sigma
        output:     - mask '''
    temp_im = 100 * (im > 1e-5) #remove np.uint8
    temp_im = gaussian_filter(temp_im, sigma) 
    # normalisation
    #temp_im = np.uint8(temp_im / np.max(temp_im))
    mask = (temp_im == np.max(temp_im))  # replace 0.99 by np.max(temp_im)
    return mask


def get_index_matrix_3D(im, mask):
    # dimensions
    Nx, Ny, Nz = im.shape

    # Hessian analaysis  NB: try with edge_order = 1 and edge_order = 2
    gx, gy, gz = np.gradient(im, edge_order=1)
    gxx, gxy, gxz = np.gradient(gx, edge_order=1)
    gyx, gyy, gyz = np.gradient(gy, edge_order=1)
    gzx, gzy, gzz = np.gradient(gz, edge_order=1)

    # initiate matrices
    I = np.zeros((Nx, Ny, Nz))
    lambda1_matrix = np.zeros((Nx, Ny, Nz))
    lambda2_matrix = np.zeros((Nx, Ny, Nz))
    lambda3_matrix = np.zeros((Nx, Ny, Nz))
    
    # No need to do calculation out of the cortex
    coord_x, coord_y, coord_z = np.where(mask > 0)        #np.where(mask == 1)

    for i in range(len(coord_x)):
        x, y, z = coord_x[i], coord_y[i], coord_z[i]
        # hessian matrix at pixel x, y, z 
        h = [[gxx[x, y, z], gxy[x, y, z], gxz[x, y, z]],
            [gyx[x, y, z], gyy[x, y, z], gyz[x, y, z]],
            [gzx[x, y, z], gzy[x, y, z], gzz[x, y, z]]]
        
        # The eigenvalues in ascending order
        lambda1, lambda2, lambda3 = np.linalg.eigvalsh(h)

        # store the eigenvalues
        lambda1_matrix[x, y, z] = lambda1
        lambda2_matrix[x, y, z] = lambda2
        lambda3_matrix[x, y, z] = lambda3
    
        # check if h is definite negative
        if (lambda3 < 0): 
            I[x, y, z] = 1 
    return I, lambda1_matrix, lambda2_matrix, lambda3_matrix

def get_S_value(lambda1, lambda2, lambda3, coords):
    s = []
    length = len(coords)
    for i in range(length):
        nx, ny, nz = coords[i]
        l1, l2, l3 = lambda1[nx, ny, nz], lambda2[nx, ny, nz], lambda3[nx, ny, nz]
        s_voxel = np.sqrt(l1**2 + l2**2 + l3**2)
        s.append(s_voxel)
    return s

def get_R_value(lambda1, lambda2, coords):
    r = []
    length = len(coords)
    for i in range(length):
        nx, ny, nz = coords[i]
        l1, l2 = lambda1[nx, ny, nz], lambda2[nx, ny, nz]
        r_voxel = np.abs(l1) / np.abs(l2)
        r.append(r_voxel)
    return r

def get_mean_value(im, coords):
    '''coords : list of coords of the voxels in the blob
        im = the image to get the mean intensity '''
    mean_log = 0.
    length = len(coords)
    for i in range(length):
        nx, ny, nz = coords[i]
        mean_log += im[nx, ny, nz]
    mean_log = mean_log / length
    return mean_log

def get_Ht(gxx, gxy, gxz, gyy, gyz, gzz, coords):
    r = []
    length = len(coords)
    Ht = np.zeros((3, 3))
    for i in range(length):
        x, y, z = coords[i]
        Ht += [[gxx[x, y, z], gxy[x, y, z], gxz[x, y, z]],
               [gxy[x, y, z], gyy[x, y, z], gyz[x, y, z]],
               [gxz[x, y, z], gyz[x, y, z], gzz[x, y, z]]]
    return Ht 

def z_score_normalisation(mri):
    '''z_score normalisation for non-zero voxels'''
    # z-score normalisation (non zero voxels only): 
    mean = np.mean(mri[np.where(mri>np.min(mri))])
    std = np.std(mri[np.where(mri>np.min(mri))])
    mri = (mri - mean) / (std)
    return mri 


def optimale_scale_identification(mri, mask, min_sigma=1.5, step=0.05, n_iter=3):
    
    # lists to store the results: 
    list_sigma = []
    list_avg_score = []

    # parameters
    BLoG_best = -100 
    max_sigma = min_sigma + n_iter * step 

    # iterate through sigma values
    i = min_sigma 
    while (i <= max_sigma):
        # remove edges
        mask_edge = mask_without_edges(mask, sigma=i)

        # LoG filtering
        LoG_temp =  ndimage.gaussian_laplace(mri, sigma=i)

        # normalisation of LoG_temp in [-1 ; 1] for fair comparaison
        xmin = np.min(LoG_temp)
        xmax = np.max(LoG_temp)
        LoG_temp =  2 * ((LoG_temp - xmin) / (xmax - xmin)) - 1

        # get the binary index matrix (convexity map) from Hessian analysis
        I_temp, lambda1_temp, lambda2_temp, lambda3_temp = get_index_matrix_3D(im=LoG_temp, mask= mask * mask_edge) # mask = mask * mask_edge (at 30-micron to remove the border) 
        
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
        i = round(i + step, ndigits=2)

    return list_sigma, list_avg_score, LoG_norm, I, lambda1, lambda2, lambda3
    

def hessian(im):
    gx, gy, gz = np.gradient(im, edge_order=1)
    gxx, gxy, gxz = np.gradient(gx, edge_order=1)
    _, gyy, gyz = np.gradient(gy, edge_order=1)
    _, _, gzz = np.gradient(gz, edge_order=1)
    
    return gxx, gxy, gxz, gyy, gyz, gzz
    


    

