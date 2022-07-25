import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
import matplotlib.pyplot as plt
import cv2

for i in range(50, 155):
    imgpath='/media/utkarsh/New Volume/archive/BraTS2020_training_data/content/data/volume_250_slice_'+str(i)+'.h5'
    f=h5py.File(imgpath,'r')
    img=f['image']
    imgArray=np.array(img)
    norm=cv2.normalize(imgArray,None, norm_type=cv2.NORM_MINMAX)
    fig = plt.figure(figsize=(6, 6))
    fig.add_subplot(1, 2, 1)
    plt.imshow(norm)
    plt.title("Original")
    img=f['mask']
    imgArray=np.array(img)
    norm=cv2.normalize(imgArray,None, norm_type=cv2.NORM_MINMAX)
    fig.add_subplot(1, 2, 2)
    plt.imshow(norm*255)
    plt.title("Mask")
    print(i)
    plt.show()