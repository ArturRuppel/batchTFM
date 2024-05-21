"""

@author: Artur Ruppel

"""

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, dilation
from skimage.transform import downscale_local_mean
import tifffile
import time
from pyTFM.grid_setup_solids_py import prepare_forces
from pyTFM.grid_setup_solids_py import grid_setup, FEM_simulation

def main(path_mask, path_TFM_data, savepath, pixelsize, downsamplerate):
    print("Loading image stack")
    
    mask_all = tifffile.imread(path_mask)[:, :, :]
    # load data
    t_x = np.load(path_TFM_data + "t_x.npy")    
    t_y = np.load(path_TFM_data + "t_y.npy")  
    
    # rescale masks to the resolution of the forcemaps
    mask_all = downscale_local_mean(mask_all, (1, downsamplerate, downsamplerate))
  
    # convert to m and rescale pixelsize
    forcemap_pixelsize = pixelsize * downsamplerate * 1e-6 
    
    # calculate cell/tissue stresses (MSM)
    stress_tensor_all = np.zeros((t_x.shape[0], t_x.shape[1], t_x.shape[2], 2, 2))
    
    no_frames = mask_all.shape[0]
    
    print("Starting stress calculations")
    for frame in range(no_frames):
        mask = dilation(mask_all[frame, :, :], disk(10)) > 0
        mask = binary_fill_holes(mask)

        t0 = time.perf_counter()
        # prepare_forces takes pixelsize in micron, traction stress in Pa and returns forces in N
        f_x, f_y = prepare_forces(t_x[frame, :, :], t_y[frame, :, :], forcemap_pixelsize * 1e6, mask)
    
        # construct FEM grid
        nodes, elements, loads, mats = grid_setup(mask, -f_x, -f_y, sigma=0.5)
        
        # performing FEM analysis. Returns stresses in N/pixel
        UG_sol, stress_tensor = FEM_simulation(nodes, elements, loads, mats, mask, verbose=False)
        stress_tensor[np.isnan(stress_tensor)] = 0
        stress_tensor_all[frame, :, :, :, :] = stress_tensor / forcemap_pixelsize
        t1 = time.perf_counter()
        print("Frame " + str(frame) + ": Stress calculations took " + str((t1-t0)/60) + " minutes")


        
    stress_tensor = stress_tensor_all
    np.save(savepath + "stress_tensor.npy", stress_tensor)


