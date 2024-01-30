import argparse

parser = argparse.ArgumentParser(description="HLoG Blob detector arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("mri_path", type=str, help="path to access to the mri data")
parser.add_argument("mask_path", type=str, help="path to access to the mask data")
parser.add_argument("resolution", type=float, help="resolution of the MRI (30u, 60u or 100u)")
parser.add_argument("--n_iter", type=int, default=3, help="number of iteration of sigma values")
parser.add_argument("--step_sigma", type=float, default=0.1, help="step used to iterate through the range of sigma values (0.1 at 30u, 0.05 at 60u, 0.03 at 100u")
args = parser.parse_args()
