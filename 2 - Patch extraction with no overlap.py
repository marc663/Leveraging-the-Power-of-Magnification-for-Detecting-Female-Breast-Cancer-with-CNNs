# The following extracts patches from the WSI. The patches are not overlapping. The patches are saved in an extra directory.
# As first step we create a dataframe which contains all the data and we then define a method to create patches. As last step we apply the method to our dataframe to create a defined number of patches

# The code is copied from: https://github.com/3dimaging/DeepLearningCamelyon/blob/master/2%20-%20Image%20Preprocess/PatchExtraction_No_Overlap.py
# and adapted for this project

import imageio
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import cv2
import keras
import keras_utils
from keras.utils.np_utils import to_categorical
from skimage import img_as_ubyte


slide_path = r'<<Insert_your_path>>'
BASE_TRUTH_DIR = r'<<Insert_your_path>>'

slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
BASE_TRUTH_DIRS.sort()
patch_path = r'<<Insert_your_path>>'

# Create dataframe for the data

sampletotal = pd.DataFrame([])
i=0
while i < len(slide_paths):
    base_truth_dir = Path(BASE_TRUTH_DIR)
    slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')
    
    with openslide.open_slide(slide_paths[i]) as slide: #iterate through the WSI images
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 299, slide.dimensions[1] / 299)) #Return an Image containing an RGB thumbnail of the slide
    
        thumbnail_grey = np.array(thumbnail.convert('L')) # convert the images to greyscale and create matrix a with the pixel values of the grey image
        thresh = threshold_otsu(thumbnail_grey) # Upper threshold value. All pixels intensities that less or equal of this value assumed as foreground, value 202
        binary = thumbnail_grey > thresh # Creates a matrix that contains True and False, True if pixel value > 202

        patches = pd.DataFrame(pd.DataFrame(binary).stack()) #Data frame which contains every single pixel
        patches['is_tissue'] = ~patches[0] #adds colums is_tissue to the dataframe, if column '0' = True, the is_tissue=False --> is the opposite
        patches.drop(0, axis=1, inplace=True)  # Drop the column with True/False with the otsu threhsold
        patches['slide_path'] = slide_paths[i] # adds the column "slide path" for every pixel - refers to the path of the slide
    
    if slide_contains_tumor:
        truth_slide_path = base_truth_dir / osp.basename(slide_paths[i]).replace('.tif', '_mask.tif') # Set path to the mask of the tumor slide
        with openslide.open_slide(str(truth_slide_path)) as truth:
            # Get a thumbnail of the tumor mask:
            thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / 299, truth.dimensions[1] / 299)) # Divide by 299, because our model will have images of the size 299x299 as input

        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack()) #Data frame which contains every single pixel
        patches_y['is_tumor'] = patches_y[0] > 250 #Adds column 'is tumor' --> True for every pixel value>250 (every white or very bright pixel)
        patches_y.drop(0, axis=1, inplace=True) # Drop the column '0' with pixel value

        samples = pd.concat([patches, patches_y], axis=1) #Creates data fames which contains data frames patches, patches_y

    else:
        samples = patches #When the slide does not contain tumor, only df patches is used
        samples['is_tumor'] = False # Set value in column 'is tumor' = False for every pixel in the df

    # if filter_non_tissue:
    samples = samples[samples.is_tissue == True] # Only keep pixels which contain tissue (remove background etc.)
    samples['tile_loc'] = list(samples.index) #Location in within te image of every pixel that contains tissue
    samples.reset_index(inplace=True, drop=True) # Reset the indizes within the dataframe samples

    sampletotal=sampletotal.append(samples, ignore_index=True) #Add the new samples to df for every iteration
        
    i=i+1

# randomly drop normal rows to match the number of tumor rows
idx=sampletotal.index[sampletotal['is_tumor'] == False].tolist() # create list with indizes of rows where ['is_tumor'] == False
idx_tum=sampletotal.index[sampletotal['is_tumor'] == True].tolist() # create list with the indizes of rows with cancer
# drop so many normal rows, that they match the tumor rows: (len(idx)-len(idx_tum))
drop_indices = np.random.choice(idx, (len(idx)-len(idx_tum)), replace=False)
sampletotal_subset = sampletotal.drop(drop_indices) #drop all the randomly select rows with normal pixels
# reorder the index
sampletotal_subset.reset_index(drop=True, inplace=True) #reorder the index

NUM_SAMPLES = 100000
sampletotal_subset= sampletotal_subset.sample(NUM_SAMPLES, random_state=42) #randomly select 100000 rows from the created subset
 
sampletotal_subset.reset_index(drop=True, inplace=True) #reorder the index

print(sampletotal_subset.is_tumor.value_counts()) #print the amount of tumor and normal instances


# Adapt tile location in order to extract patches at different zoom levels, patches scale up or down by a factor of 2 as the zoom levels are changed
# Only run the following four lines when we extract patches for other image levels than image level 0
# For image level 1 divide by 2 ; image level 2 divide by 4 and image level 3 divide by 8
i=0
while i < len(sampletotal_subset):
    sampletotal_subset['tile_loc'][i] = tuple(ti/2 for ti in sampletotal_subset['tile_loc'][i])
    i=i+1


#Define method to create patches
def gen_imgs(samples, num_patches, base_truth_dir=BASE_TRUTH_DIR, shuffle=True):
       
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1) # shuffle samples
        
        for offset in range(0, num_samples, num_patches):
            patch_samples = samples.iloc[offset:offset+num_patches]
        
            for _, patch_sample in patch_samples.iterrows():
                slide_contains_tumor = osp.basename(patch_sample.slide_path).startswith('tumor_')
                 
                with openslide.open_slide(patch_sample.slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=299, overlap=0, limit_bounds=False) #Set size of patches
                    img = tiles.get_tile(tiles.level_count-1, patch_sample.tile_loc[::-1])
                    im = np.array(img)
                    int1, int2= patch_sample.tile_loc[::-1]
                    if  patch_sample.is_tumor==True:
                        # Save the patches
                        # apply img_as_ubyte() to the patches and the masks to convert them to uint8 and suppress the loss of images due to conversion and avoid this warning: "Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning."
                        imageio.imwrite(r'<<Insert_your_path>>\%s_%d_%d.png' % (os.path.splitext(osp.basename(patch_sample.slide_path))[0], int1, int2), img_as_ubyte(im))
                    else:
                        imageio.imwrite(r'<<Insert_your_path>>\%s_%d_%d.png' % (os.path.splitext(osp.basename(patch_sample.slide_path))[0], int1, int2), img_as_ubyte(im))

                # only load truth mask for tumor slides
                if slide_contains_tumor:
                    truth_slide_path = osp.join(base_truth_dir, osp.basename(patch_sample.slide_path).replace('.tif', '_mask.tif'))
                    with openslide.open_slide(str(truth_slide_path)) as truth:
                        truth_tiles = DeepZoomGenerator(truth, tile_size=299, overlap=0, limit_bounds=False) #Set size of patches
                        mask = truth_tiles.get_tile(truth_tiles.level_count-1, patch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        mk = np.array(mask)
                        int1, int2= patch_sample.tile_loc[::-1]
                    if  patch_sample.is_tumor==True:
                        # Save the respective masks of the patches
                        imageio.imwrite(r'<<Insert_your_path>>\%s_%d_%d.png' % (os.path.splitext(osp.basename(patch_sample.slide_path))[0], int1, int2), img_as_ubyte(mk))
                    else:
                        imageio.imwrite(r'<<Insert_your_path>>\%s_%d_%d.png' % (os.path.splitext(osp.basename(patch_sample.slide_path))[0], int1, int2),img_as_ubyte(mk)) 
                else:
                    mask = np.zeros((299, 299))
                    mk = np.array(mask)
                    int1, int2= patch_sample.tile_loc[::-1]
                    imageio.imwrite(r'<<Insert_your_path>>\%s_%d_%d.png' % (os.path.splitext(osp.basename(patch_sample.slide_path))[0], int1, int2), img_as_ubyte(mk))

                       
            yield


number_of_patches=len(sampletotal_subset) # define how many patches we create - we create as many patches as we have instances in our dataframe

train_generator = gen_imgs(sampletotal_subset, number_of_patches) # Apply method to our created dataframe and specify how many patches we want to have 

next(train_generator) # Call train_generator and saves patches in respective folders