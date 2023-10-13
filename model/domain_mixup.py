#!/usr/bin/env python
# coding: utf-8

# In[62]:


from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
import cv2
import pickle
from keras.regularizers import l2
from keras.models import Model
f=[]
g=[]
labels=[]
labels1=[]
bary_centers_mix=[]
mixup_input=[]
for i in range(len(cy1)):

    
    if((cy1[i]==cy1[i+1])&(cy2[i]==cy2[i+1]) ):
        
        f.append(c1[i])
        g.append(c2[i])

    else:
        
        f1=np.array(f)
        f2=np.array(g)
        labels.append(cy1[i])
        labels1.append(cy2[i])
    l=len(f1)
    l1=len(f2)
    if(l>l1):
        z=l1
    else:
        z=l

    if(z>=10):
            weights=np.full(20, 1/20)
            arr=np.array([f1[0:10],f2[0:10]])
            arr=arr.reshape(20,256,256,3)
            #arr=np.array(arr,dtype=np.float64)
            
    else:
            weights=np.full(2*z,1/(2*z))
            w=len(weights)
            w1=int(w/2)
            arr=np.array([f1[0:w1],f2[0:w1]])
            arr=arr.reshape(w,256,256,3)
            #arr=np.array(arr,dtype=np.float64)
    mixup_input.append([arr,cy1[i],cy2[i]])
f_mixup=f_mixup.reshape(6234,20,32,16)
dom1_label[0:100]
dom2_label[0:100]
#mixup between cam1 and cam2
import ot
#y_ground=np.argmax(y, axis=1)
#barycenter calculation
reg=0.002
f=[]
g=[]
labels=[]
labels1=[]
bary_centers_mix=[]

for i in range(len(dom1)):

    
    if((dom1_label[i]==dom1_label[i+1] )):
        
        f.append(dom1[i])
        g.append(dom2[i])

    else:
        
        f1=np.array(f)
        f2=np.array(g)
        labels.append(dom1_label[i])
        labels1.append(dom2_label[i])
        l=len(f1)
        l1=len(f2)
        if(l>l1):
            z=l1
        else:
            z=l
        if(z>0):

            if(z>=10):
                weights=np.full(10, 1/10)
                arr=np.array([f1[0:5],f2[0:5]])
                arr=arr.reshape(10,32,16)
                arr=np.array(arr,dtype=np.float64)
            else:

                weights=np.full(2*z,1/(2*z))
                w=len(weights)
                w1=int(w/2)
                arr=np.array([f1[0:w1],f2[0:w1]])
                arr=arr.reshape(w,32,16)
                arr=np.array(arr,dtype=np.float64)
            print("class-label",dom1_label[i])
            print("input shape",arr.shape)
            print("weights",weights.shape)
            print("# samples",i)
            #f1 = f1 / np.sum(f1)

            f2=ot.bregman.convolutional_barycenter2d(arr, reg,weights)
            bary_centers_mix.append(f2)
            print("bc",f2)
            f=[]
            g=[]
l1=np.array(labels)
l1.shape

