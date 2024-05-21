"""

@author: Artur Ruppel

"""

import numpy as np
from skimage import img_as_float
from skimage import io
from plot_functions import *

def main(path_supp_stack, supp_stack_file_name, savepath, pixelsize, downsamplerate, d_max=2, p_max=2):
    
    pixelsize *= 1e-6
    forcemap_pixelsize = pixelsize * downsamplerate
    print("Loading image stack")
    supp_stack = io.imread(path_supp_stack)[:, :, :]  # / (2 ** 16)
    # load data
    d_x = np.load(savepath + "d_x.npy") * pixelsize * 1e6
    d_y = np.load(savepath + "d_y.npy") * pixelsize * 1e6
    t_x = np.load(savepath + "t_x.npy")    
    t_y = np.load(savepath + "t_y.npy")  
    no_frames = d_x.shape[0]

    print("Making plots")
    for frame in np.arange(no_frames):
        make_displacement_plots(d_x[frame, :, :], d_y[frame, :, :], d_max, savepath + "displacement" + str(frame) + ".png", forcemap_pixelsize, frame=frame)
        make_traction_plots(t_x[frame, :, :], t_y[frame, :, :], p_max, savepath + "traction" + str(frame), forcemap_pixelsize, frame=frame)
        plot_image_with_forces(2*img_as_float(supp_stack[frame, :, :]), t_x[frame, :, :], t_y[frame, :, :], p_max, savepath + supp_stack_file_name + "_forces" + str(frame), forcemap_pixelsize, frame=frame)


    make_movies_from_images(savepath, "displacement", no_frames)
    make_movies_from_images(savepath, "traction", no_frames)
    make_movies_from_images(savepath, supp_stack_file_name + "_forces", no_frames)

