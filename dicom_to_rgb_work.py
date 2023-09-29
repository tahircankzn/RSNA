



import pydicom

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np




def read_dcm(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array

    #plt.imshow(img)
    #plt.show()
    #print('Shape:', img.shape)

    return "ID_0a0f3abd0", img

def Image_resize(img,len_x):
    img = img.resize((len_x, len_x), Image.ANTIALIAS)
    return img

_,img = read_dcm("ID_0a0f3abd0.dcm")
#print(img)



def To_16bit(img_arr):
    min_arr = np.amin(img_arr)
    max_arr = np.amax(img_arr)
    range_array = max_arr - min_arr

    return np.round((img_arr-min_arr)/range_array*(np.power(2,16)*3-1)) 


def To_RGB(img_arr): #Extend to 24 bit then segment by 8 bit
    min_arr = np.amin(img_arr)
    max_arr = np.amax(img_arr)
    range_array = max_arr - min_arr
    
    lenx, leny = img_arr.shape
    rgbArray = np.zeros((lenx,leny,3), 'uint8')

    arr2 = np.round((img_arr-min_arr)/range_array*(np.power(2,24)-1))
    rgbArray[:,:, 0] = arr2 % np.power(2,8)
    arr2 = (arr2-rgbArray[:,:, 0])/np.power(2,8)
    rgbArray[:,:, 1] = arr2 % np.power(2,8)
    arr2 = (arr2-rgbArray[:,:, 1])/np.power(2,8)
    rgbArray[:,:, 2] = arr2 % np.power(2,8)
    
    return rgbArray 
    
def To_RGB2(img_arr): #Extend to 16 bit then segment by 8 bit top(G), 8 bit bottom(R), 8 bit overall(B)
    min_arr = np.amin(img_arr)
    max_arr = np.amax(img_arr)
    range_array = max_arr - min_arr
    
    lenx, leny = img_arr.shape
    rgbArray = np.zeros((lenx,leny,3), 'uint8')

    arr2 = np.round((img_arr-min_arr)/range_array*(np.power(2,16)-1))
    rgbArray[:,:, 0] = arr2 % np.power(2,8)                 #8 bit bottom
    arr2 = np.floor(arr2/np.power(2,8))                     #8 bit top
    rgbArray[:,:, 1] = arr2 
    
    rgbArray[:,:, 2] = np.round((img_arr-min_arr)/range_array*(np.power(2,8)-1)) #8 bit overall
    
    
    return rgbArray




#rgb_arr = To_RGB2(img)
#rgb_arr = np.transpose(rgb_arr, axes=(2, 0, 1))
#print(rgb_arr.shape)
#print(rgb_arr[0].shape)
#print(rgb_arr[1].shape)
#print(rgb_arr[2].shape)
#plt.imshow(rgb_arr)
#plt.show()
#print('Image Shape:', rgb_arr.shape)
#print(list(rgb_arr.flatten()))

# https://www.kaggle.com/code/dipuk0506/dicom-files-to-rgb-512x512