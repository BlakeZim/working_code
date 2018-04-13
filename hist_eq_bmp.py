import scipy.misc as sm
import numpy as np
import PyCA.Common as common
import PyCACalebExtras.Display as cd

def histmatch(np_src, np_ref):

    src = np_src.ravel()
    ref = np_ref.ravel()
    refshape = np_src.shape

    # Extract the set of unique pixel values and thier locations from the full image
    o_vals, bin_idx = np.unique(src, return_inverse=True)

    # Only match images where they are not zero (useful for masked images)
    ref_nnzero = ref  # [np.where(ref != 0)]
    src_nnzero = o_vals  # [np.where(o_vals != 0)]
    numzeros = o_vals.size - src_nnzero.size

    # Extract the set of unique pixel values and their corresponding indicies and counts from the non-zero array
    s_vals, s_counts = np.unique(src_nnzero, return_counts=True)
    r_vals, r_counts = np.unique(ref_nnzero, return_counts=True)

    # Use cumsum of the counts and normalize by number pixels to get the emperical cumulative distribution functions
    s_quant = np.cumsum(s_counts).astype(np.float64)
    s_quant /= s_quant[-1]
    r_quant = np.cumsum(r_counts).astype(np.float64)
    r_quant /= r_quant[-1]

    # Linear Interpolate to find pixel values in reference image that correspond to the quantiles in the source image
    interp_r_vals = np.interp(s_quant, r_quant, r_vals)

    np_int = np.pad(interp_r_vals, numzeros, 'constant', constant_values=0)
    np_out = np_int[bin_idx].reshape(refshape)

    return np_out 


inDir = '/home/sci/blakez/korenbergNAS/3D_database/Working/Blockface/T2_registered/Full_deformation/M15/bmpfiles/'
histim = sm.imread('/home/sci/blakez/Hist_Reference.bmp')
source = sm.imread(inDir + 'M15_01_BFI_as_MRI_hd1_slice1112.bmp')

hist_matched = histmatch(source, histim)

common.DebugHere()

# file_path = sys.argv[1]
# scan_num = file_path[-25:-19]
