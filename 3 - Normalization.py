# This code applies Macenko normalization to all patches from a specified directory and stores the normalized patches in another specified directory
# Comment out the path and save command if it is not required (e.g. if we normalize the normal patches, comment the path for the tumor folders out and the respective line to save the patch)

# The code for the normalization step of a single patch is copied from this repository: https://github.com/bnsreenu/python_for_microscopists/blob/master/122_normalizing_HnE_images.py#L62 and we adapted it slightly to normalize all patches in a folder
# Furthermore, there is a YouTube video which gives some context to this method and explains the code: https://www.youtube.com/watch?v=yUrwEYgZUsA
# The paper to the Macenko normalization: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
import os.path as osp
import imageio

#slide_path = r'<<Insert_your_path>>'
slide_path = r'<<Insert_your_path>>'

slide_paths = glob.glob(osp.join(slide_path, '*.png'))
slide_paths.sort()

i=0
while i < len(slide_paths):
    img=cv2.imread(slide_paths[i],1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV reads images in as BGR but we want RGB --> convert it back
    
    Io = 240 # Transmitted light intensity, Normalizing factor for image intensities
    alpha = 1  #As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
    beta = 0.15 #As recommended in the paper. OD threshold for transparent pixels (default: 0.15)
    
    Io = 240 # Transmitted light intensity, Normalizing factor for image intensities
    alpha = 1  #As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
    beta = 0.15 #As recommended in the paper. OD threshold for transparent pixels (default: 0.15)
    
    HERef = np.array([[0.5626, 0.2159],
                  [0.7201, 0.8012],
                  [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])
    
    # extract the height, width and num of channels of image
    h, w, c = img.shape
    
    # reshape image to multiple rows and 3 columns.
    #Num of rows depends on the image size (wxh)
    img = img.reshape((-1,3))
    
    OD = -np.log10((img.astype(np.float)+1)/Io) #Use this for opencv imread
    #Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)

    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta
  
    #Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3]) #Dot product

    #find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1],That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
    
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the normalized image using reference mixing matrix 
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8) 
    
    name = os.path.splitext(os.path.basename(slide_paths[i]))[0]
    # path_tum = r'<<Insert_your_path>>\_'+str(name)+".png"
    path_norm = r'<<Insert_your_path>>\_'+str(name)+".png"
    
    #Save the picture
    # imageio.imwrite(path_tum, Inorm)
    imageio.imwrite(path_norm, Inorm)
    
    i=i+1