# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 08:58:42 2019

@author: iamav
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import cv2
import math

path = "MPIIGaze//Data//Original//p00//day01"
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    return images

images = load_images_from_folder(path)
#plt.imshow(images[0])
#plt.show

#img= images[0]

#px = img[300,600]
#print(np.mean(img))

mean_img=[]
for i in range(len(images)):
    mean_img.append(np.mean(images[i]))
print(math.ceil(min(mean_img)))
    
bins = np.linspace(math.ceil(min(mean_img)), 
                   math.floor(max(mean_img)), 20)

plt.hist(mean_img, bins=bins)
plt.title("Mean image histogram for p00 day1")
plt.ylabel("no. of images")
plt.xlabel("mean value")
plt.show()