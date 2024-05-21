# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:14:04 2023

@author: aruppel
"""

import os
import time
import numpy as np
import tifffile
from skimage import exposure
from skimage import restoration
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from skimage import img_as_uint
from skimage.filters import gaussian

def remove_background(image, radius=100):
    if radius == 0:
        return image
    else:
        background = restoration.rolling_ball(image, radius=radius)
        return image - background


def stretch_contrast(image, p2, p98):
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale

def stretch_contrast_relative(image, lower_threshold=0.001, upper_threshold=99.99):
    p2 = np.percentile(image, lower_threshold)
    p98 = np.percentile(image, upper_threshold)
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale

def main(savepath, path_stressed, path_relaxed, path_stack_optional_1, path_stack_optional_2, path_stack_optional_3, 
         name_image_relaxed, name_stack_stressed, name_stack_optional_1, name_stack_optional_2, name_stack_optional_3,
         pixelsize, finterval, lower_threshold, upper_threshold, sigma_smooth, radius):

    # Load the TIFF files and set up variables for saving processed images, registered images and displacement vectors
    relaxed = tifffile.imread(path_relaxed)
    stressed = tifffile.imread(path_stressed)
        
    stressed_processed = np.zeros_like(stressed)
    stressed_registered = np.zeros_like(stressed)

    stack_optional_1 = None
    stack_optional_1_processed = None
    stack_optional_1_registered = None
    stack_optional_2 = None
    stack_optional_2_processed = None
    stack_optional_2_registered = None
    stack_optional_3 = None
    stack_optional_3_processed = None
    stack_optional_3_registered = None

    if path_stack_optional_1 is not None:
        stack_optional_1 = tifffile.imread(path_stack_optional_1)#[0:2, :, :]
        stack_optional_1_processed = np.zeros_like(stack_optional_1)
        stack_optional_1_registered = np.zeros_like(stack_optional_1)


    if path_stack_optional_2 is not None:
        stack_optional_2 = tifffile.imread(path_stack_optional_2)#[0:2, :, :]
        stack_optional_2_processed = np.zeros_like(stack_optional_2)
        stack_optional_2_registered = np.zeros_like(stack_optional_2)

    if path_stack_optional_3 is not None:
        stack_optional_3 = tifffile.imread(path_stack_optional_3)#[0:2, :, :]
        stack_optional_3_processed = np.zeros_like(stack_optional_3)
        stack_optional_3_registered = np.zeros_like(stack_optional_3)

    displacement_vectors = np.zeros((stressed.shape[0], 2))


    print("Removing background, stretching contrast for relaxed image")

    p2 = np.percentile(relaxed, lower_threshold[0])
    p98 = np.percentile(relaxed, upper_threshold[0])

    def apply_filters(image, radius, p2, p98, sigma_smooth):
        intermediate_image_1 = remove_background(image, radius)
        indermediate_image_2 = stretch_contrast(intermediate_image_1, p2, p98)
        result_image = img_as_uint(gaussian(indermediate_image_2, sigma=sigma_smooth))

        return result_image

    relaxed_processed = apply_filters(relaxed, radius[0], p2, p98, sigma_smooth[0])
    del relaxed

    for frame in np.arange(stressed.shape[0]):
        print("Removing background, stretching contrast for stressed images, frame " + str(frame))
        stressed_processed[frame, :, :] = apply_filters(stressed[frame, :, :], radius[0], p2, p98, sigma_smooth[0])
    del stressed

    if stack_optional_1 is not None:
        p2 = np.percentile(stack_optional_1, lower_threshold[1])
        p98 = np.percentile(stack_optional_1, upper_threshold[1])
        for frame in np.arange(stack_optional_1.shape[0]):
            print("Removing background, stretching contrast for optional images 1, frame " + str(frame))
            stack_optional_1_processed[frame, :, :] = apply_filters(stack_optional_1[frame, :, :], radius[1], p2, p98, sigma_smooth[1])
        del stack_optional_1

    if stack_optional_2 is not None:
        p2 = np.percentile(stack_optional_2, lower_threshold[2])
        p98 = np.percentile(stack_optional_2, upper_threshold[2])
        for frame in np.arange(stack_optional_2.shape[0]):
            print("Removing background, stretching contrast for optional images 2, frame " + str(frame))
            stack_optional_2_processed[frame, :, :] = apply_filters(stack_optional_2[frame, :, :], radius[2], p2, p98, sigma_smooth[2])
        del stack_optional_2

    if stack_optional_3 is not None:
        p2 = np.percentile(stack_optional_3, lower_threshold[3])
        p98 = np.percentile(stack_optional_3, upper_threshold[3])
        for frame in np.arange(stack_optional_3.shape[0]):
            print("Removing background, stretching contrast for optiomal images 3, frame " + str(frame))
            stack_optional_3_processed[frame, :, :] = apply_filters(stack_optional_3[frame, :, :], radius[3], p2, p98, sigma_smooth[3])
        del stack_optional_3


    # Loop over the time frames of stressed.tif and perform image registration for each image
    print("Starting translation correction")
    for frame in range(stressed_processed.shape[0]):
        # set a timer to measure how long the analysis takes for each frame
        t0 = time.perf_counter()
        
        if frame == 0:
            # Use the first frame as the reference for registration
            reference = stressed_processed[0]
            displacement, _, _ = phase_cross_correlation(reference, relaxed_processed, upsample_factor=100, normalization=None)
            relaxed_registered = shift(relaxed_processed, displacement)
            displacement = 0
        else:
            # Register the current frame to the reference frame using phase cross correlation
            displacement, _, _ = phase_cross_correlation(reference, stressed_processed[frame], upsample_factor=100, normalization=None)
            displacement_vectors[frame] = displacement
            
        # Shift the image and save the registered image
        stressed_registered[frame] = shift(stressed_processed[frame], displacement)

        t1 = time.perf_counter()
        print("Frame " + str(frame) + ": Translation correction took " + str((t1 - t0) / 60) + " minutes")

    # Apply the same displacement to the optional stacks
    if stack_optional_1_processed is not None:
        for frame in range(stack_optional_1_processed.shape[0]):
            stack_optional_1_registered[frame] = shift(stack_optional_1_processed[frame], displacement_vectors[frame])
    if stack_optional_2_processed is not None:
        for frame in range(stack_optional_2_processed.shape[0]):
            stack_optional_2_registered[frame] = shift(stack_optional_2_processed[frame], displacement_vectors[frame])
    if stack_optional_3_processed is not None:
        for frame in range(stack_optional_3_processed.shape[0]):
            stack_optional_3_registered[frame] = shift(stack_optional_3_processed[frame], displacement_vectors[frame])


    # Save the registered images and displacement vectors
    tifffile.imwrite(os.path.join(savepath, name_image_relaxed + '_registered.tif'),
                      relaxed_registered.astype("uint16"),
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'axes': 'YX'
                          }
                      )
    tifffile.imwrite(os.path.join(savepath, name_stack_stressed + '_registered.tif'),
                      stressed_registered.astype("uint16"),
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'finterval': finterval,
                          'axes': 'TYX'
                          }
                      )
    if stack_optional_1_registered is not None:
        tifffile.imwrite(os.path.join(savepath, name_stack_optional_1 + '_registered.tif'),
                          stack_optional_1_registered.astype("uint16"),
                          imagej=True,
                          resolution=(1/pixelsize, 1/pixelsize),
                          metadata={
                              'unit': 'um',
                              'finterval': finterval,
                              'axes': 'TYX'
                              }
                          )
    if stack_optional_2_registered is not None:
        tifffile.imwrite(os.path.join(savepath, name_stack_optional_2 + '_registered.tif'),
                          stack_optional_2_registered.astype("uint16"),
                          imagej=True,
                          resolution=(1/pixelsize, 1/pixelsize),
                          metadata={
                              'unit': 'um',
                              'finterval': finterval,
                              'axes': 'TYX'
                              }
                          )
    if stack_optional_3_registered is not None:
        tifffile.imwrite(os.path.join(savepath, name_stack_optional_3 + '_registered.tif'),
                          stack_optional_3_registered.astype("uint16"),
                          imagej=True,
                          resolution=(1/pixelsize, 1/pixelsize),
                          metadata={
                              'unit': 'um',
                              'finterval': finterval,
                              'axes': 'TYX'
                              }
                          )


    np.savetxt(os.path.join(savepath, 'displacement_vectors.txt'), displacement_vectors)

