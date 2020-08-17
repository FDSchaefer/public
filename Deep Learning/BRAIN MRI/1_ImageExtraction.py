
"""
Rights: Franz D Schaefer
https://github.com/FDSchaefer/public
Please Give Credit if used


"""

import os
import numpy as np
from medpy.io import load
import scipy.ndimage as img
import matplotlib.pyplot as plt

## DIRECTORIES ###
root = os.getcwd()
dataDir = root+'/data'
processedDir = root+'/processed'



#Input Shape for Network
standardSize = np.array([99,99,99])

loadData = True
if loadData:
    #Load all data sets
    Files = os.listdir(dataDir)
    imageList = []

    for i in range (0,len(Files)):
        name = Files[i]

        if 'mha' in name :
            print('Working on: ' + name)
            imageOut, image_header = load(dataDir + '/' + name)
            curShape = np.asarray(imageOut.shape)

            if (curShape  != standardSize).any():
                zoomMag = standardSize/curShape
                imageOut = img.zoom(imageOut,zoomMag)

            ##Normaise
            imageOut = np.divide(imageOut,np.amax(imageOut))
            imageList.append(np.expand_dims(imageOut,0))

    imageList = np.array(imageList)
    np.save(processedDir + '/standardData', imageList)

else:
    imageList = np.load(processedDir + '/standardData.npy', mmap_mode=None, allow_pickle=True)

#View Random sample to confirm read in
if True:
    Selected = np.round(np.multiply(np.random.rand(4,1), len(imageList)-1)).astype(np.int)
    Selected2 = np.round(np.multiply(np.random.rand(4,1), standardSize[2])).astype(np.int)

    plt.figure
    for i in range (0,4):
        plt.subplot(2,2,i+1)
        plt.imshow(imageList[Selected[i,0],0,:,:,Selected2[i,0]],aspect = 'equal' , cmap = 'gray')

    plt.show()