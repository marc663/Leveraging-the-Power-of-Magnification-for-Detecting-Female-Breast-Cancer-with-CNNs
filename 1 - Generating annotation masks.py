# The following code generates the annotation masks (masks where tumor regions are white and the rest is black) for our WSIs

# The code is from:  https://github.com/3dimaging/DeepLearningCamelyon/blob/master/1%20-%20WSI%20Visualization%20with%20Annotation/Mask%20Generation
# and is slightly adapted 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from pathlib import Path
import cv2 as cv2
import multiresolutionimageinterface as mir # To use this method we require Python 3.6. 
import glob
import torch
import openslide

reader = mir.MultiResolutionImageReader()

slide_path = r'<<Insert_your_path>>'
anno_path = r'<<Insert_your_path>>'
mask_path = r'<<Insert_your_path>>'
tumor_paths = glob.glob(osp.join(slide_path, '*.tif'))
tumor_paths.sort()
anno_tumor_paths = glob.glob(osp.join(anno_path, '*.xml'))
anno_tumor_paths.sort()

reader = mir.MultiResolutionImageReader()
i=0
while i < len(tumor_paths):
    mr_image = reader.open(tumor_paths[i])
    annotation_list=mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(anno_tumor_paths[i])
    xml_repository.load()
    annotation_mask=mir.AnnotationToMask()
    camelyon_type_mask = False
    # Annotations belonging to 0 and 1 represent tumor areas and annotations within group 2 respresent non-tumor areas
    label_map = {'metastases': 1, 'normal': 2} if camelyon_type_mask else {'_0': 255, '_1': 255, '_2': 0} #Annotation belonging to tumor regions get greyscal pixel value 255 (white) and normal regions pixel value 0 (black)
    conversion_order = ['metastases', 'normal'] if camelyon_type_mask else  ['_0', '_1', '_2']
    output_path= osp.join(mask_path, osp.basename(tumor_paths[i]).replace('.tif', '_mask.tif'))
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
    i=i+1