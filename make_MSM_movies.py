# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:36:52 2024

@author: aruppel
"""
# import matplotlib.pyplot as plt
import numpy as np
from plot_functions import *

def main(path_MSM_data, pixelsize, downsamplerate, sigma_max=0.5):

    forcemap_pixelsize = pixelsize * downsamplerate
    stress_tensor_all = np.load(path_MSM_data + "stress_tensor.npy")  
    no_frames = stress_tensor_all.shape[0]
    
    for frame in range(no_frames):
        make_stress_plots(stress_tensor_all[frame, :, :, 0, 0], sigma_max, path_MSM_data + "xx-Stress_" + str(frame), "xx-Stress", forcemap_pixelsize, frame=frame)
        make_stress_plots(stress_tensor_all[frame, :, :, 1, 1], sigma_max, path_MSM_data + "yy-Stress_" + str(frame), "yy-Stress", forcemap_pixelsize, frame=frame)
        make_stress_plots((stress_tensor_all[frame, :, :, 0, 0] + stress_tensor_all[frame, :, :, 1, 1]) / 2, sigma_max, path_MSM_data +
                          "avg-Stress_" + str(frame), "Avg. normal stress", forcemap_pixelsize, frame=frame)
    
    make_movies_from_images(path_MSM_data, "xx-Stress_", no_frames)
    make_movies_from_images(path_MSM_data, "yy-Stress_", no_frames)
    make_movies_from_images(path_MSM_data, "avg-Stress_", no_frames)