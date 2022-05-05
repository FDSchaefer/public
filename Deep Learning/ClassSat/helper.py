import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
import h5py
from tqdm.auto import tqdm

import os
import numpy as np
from PIL import Image


def extractImage(source,jp2,target = None,pix = 1024):
    f = rasterio.open(source + jp2)  #Open Image
    CMYK = np.uint8(reshape_as_image(f.read())) #Convert to standard image format
    I = Image.fromarray(CMYK).resize((pix, pix))
    if target != None:
        I.save(target + jp2[:-4] + ".png",optimize=True,quality=20)
    return  np.array(I)

def load_sample(file_name):
    h5_file = h5py.File(file_name, "r")
    img = h5_file["IMG"][:]
    h5_file.close()
    return img

def imgRGBConverter(In):
    from PIL import Image
    image = Image.fromarray(np.uint8(In))
    image.mode = "CMYK"
    return(np.array(image.convert('RGB')))


def sampleLoader(dir,classes):
    listed = os.listdir(dir)
    samples = []
    labels = []

    for i in tqdm(listed):
        samples.append(load_sample(dir + i))
        labels.append(classes.index(i.split("_")[0]))

    return (samples,labels)