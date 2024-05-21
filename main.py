# -*- coding: utf-8 -*-
import os
import numpy as np
import tifffile
from preprocess_images import main as preprocess_images
from measure_displacements import main as measure_displacements
from calculate_forces import main as calculate_forces
from make_TFM_movies import main as make_TFM_movies
from apply_MSM import main as apply_MSM
from make_MSM_movies import main as make_MSM_movies

# parameters to set
path = "C:/Users/aruppel/Desktop/batchTFM/test_data_minimal/position"
name_stack_stressed = "BK"
name_image_relaxed = "AK"
name_stack_optional_1 = "actin"
name_stack_optional_2 = None
name_stack_optional_3 = None
name_mask = "masks"

no_stacks = 1                               # number of stacks to be analyzed
finterval = 180                             # interval between frames in s
pixelsize = 0.108                           # size of a pixel in the bead image in µm
downsamplerate = 4                          # final forcemaps will have a resolution of the image divided by this number
radius = [0, 0, 0, 0]                       # radius of rolling ball for background substraction. Put 0 to disable
lower_threshold = [90, 70, 70, 70]          # lower threshold percentile for bead stack
upper_threshold = [99.9, 99, 99.9, 99.9]    # lower threshold percentile for bead stack
sigma_smooth = [1, 1, 1, 1]                 # radius of gaussian smoothing window
attachment = 30                             # optical flow parameter. lower values lead to smoother displacement maps
E = 19660                                   # rigidity of the cell substrate in Pa
nu = 0.5                                    # poisson ratio of the cell substrate
alpha = 1 * 1e-19                           # regularization parameter for force calculation
d_max = 2                                   # max displacement for map plts in µm
p_max = 3                                   # max traction for map plots in kPa
sigma_max = 5                               # mas stress for map plots in N/m


# Using the repr() function to convert the string values to their string representation
s_finterval = repr(finterval)
s_pixelsize = repr(pixelsize)
s_forcemap_pixelsize = repr(downsamplerate * pixelsize)
s_radius = repr(radius)
s_lower_threshold = repr(lower_threshold)
s_sigma_smooth = repr(sigma_smooth)
s_E = repr(E)
s_nu = repr(nu)
s_alpha = repr(alpha)

# Writing parameters to a .txt file
with open(path.replace("position", "TFM_parameters.txt"), "w") as file:
    # Writing the parameters to a text file
    file.write("Frame interval = " + s_finterval + " s \n" + "Pixel size = " + s_pixelsize + " µm \n" + "Forcemap pixel size = " + s_forcemap_pixelsize + " µm \n" +
               "Rolling ball radius for background substraction = " + s_radius + " pixel \n" + "Lower percentile threshold for contrast adjustment = " + s_lower_threshold + "\n" + 
               "Size of gaussian smoothing window = " + s_sigma_smooth + " pixel \n" + "Young's Modulus of TFM gel = " + s_E + " Pa \n" + 
               "Poisson's ratio of TFM gel = " + s_nu + "\n" + "Regularization parameter = " + s_alpha)

    # Closing the file
    file.close

for position in np.arange(no_stacks):
    path_mask = path + str(position) + "/" + name_mask + ".tif"
    path_stressed = path + str(position) + "/" + name_stack_stressed + ".tif"
    path_stressed_registered = path + str(position) + "/preprocessed_images/" + name_stack_stressed + "_registered.tif"
    path_relaxed = path + str(position) + "/" + name_image_relaxed + ".tif"
    path_relaxed_registered = path + str(position) + "/preprocessed_images/" + name_image_relaxed + "_registered.tif"


    path_stack_optional_1 = None
    path_stack_optional_2 = None
    path_stack_optional_3 = None
    if name_stack_optional_1:
        path_stack_optional_1 = path + str(position) + "/" + name_stack_optional_1 + ".tif"
        path_stack_optional_1_registered = path + str(position) + "/preprocessed_images/" + name_stack_optional_1 + "_registered.tif"
    if name_stack_optional_2:
        path_stack_optional_2 = path + str(position) + "/" + name_stack_optional_2 + ".tif"
        path_stack_optional_2_registered = path + str(position) + "/preprocessed_images/" + name_stack_optional_2 + "_registered.tif"
    if name_stack_optional_3:
        path_stack_optional_3 = path + str(position) + "/" + name_stack_optional_3 + ".tif"
        path_stack_optional_3_registered = path + str(position) + "/preprocessed_images/" + name_stack_optional_3 + "_registered.tif"


    savepath_preprocessing = path + str(position) + "/preprocessed_images/"
    savepath_TFM = path + str(position) + "/TFM_data/"
    savepath_MSM = path + str(position) + "/MSM_data/"

    if not os.path.exists(savepath_preprocessing):
        os.mkdir(savepath_preprocessing)

    if not os.path.exists(savepath_TFM):
        os.mkdir(savepath_TFM)
        
    if not os.path.exists(savepath_MSM):
        os.mkdir(savepath_MSM)

    ####################################################################################################################
    # Execute preprocessing
    ####################################################################################################################
    # preprocess_images(savepath_preprocessing, path_stressed, path_relaxed, path_stack_optional_1, path_stack_optional_2, path_stack_optional_3, 
    #                   name_image_relaxed, name_stack_stressed, name_stack_optional_1, name_stack_optional_2, name_stack_optional_3,
    #                   pixelsize, finterval, lower_threshold=lower_threshold, upper_threshold=upper_threshold, sigma_smooth=sigma_smooth, radius=radius)
    
    # measure_displacements(path_stressed_registered, path_relaxed_registered, savepath_TFM, finterval, pixelsize, downsamplerate=downsamplerate, attachment=attachment)
    
    # calculate_forces(savepath_TFM, pixelsize, downsamplerate=downsamplerate, E=E, nu=nu, alpha=alpha)
    
    # make_TFM_movies(path_stack_optional_1_registered, name_stack_optional_1, savepath_TFM, pixelsize, downsamplerate, d_max=d_max, p_max=p_max)
    
    # apply_MSM(path_mask, savepath_TFM, savepath_MSM, pixelsize, downsamplerate)
    
    make_MSM_movies(savepath_MSM, pixelsize, downsamplerate, sigma_max=sigma_max)