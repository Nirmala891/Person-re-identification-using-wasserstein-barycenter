#!/usr/bin/env python
# coding: utf-8

# In[4]:





# In[ ]:


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
from tensorflow.keras.applications.resnet50 import ResNet50
import os


# In[6]:


DATADIR="SYSU-MM01/cam2"
rgb_cat1=[]
entries = os.listdir(DATADIR)
for entry in entries:
  rgb_cat1.append(entry)
 
IMG_SIZE=256
rgb_cat1.sort()
cat=np.array(rgb_cat1)
cat1=[]
for i in range(len(cat)):
    cat1.append(int(cat[i]))


# In[9]:


cat1


# In[13]:


training_data1=[]


# In[14]:


def create_training_data():
    i=-1
    for category in rgb_cat1:  
        
        path = os.path.join(DATADIR,category)  
        
        i=i+1
        for img in tqdm(os.listdir(path)):
            try:
                #img_array = cv2.imread(os.path.join(path,img)) 
                img_array = cv2.imread(os.path.join(path,img)) 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                training_data1.append([new_array,cat1[i]-1])
                print(cat1[i])
                
            except OSError as e:
               print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
               print("general exception", e, os.path.join(path,img))


# In[15]:


create_training_data()


# In[ ]:


X1 = []
y1 = []

for features,label in training_data2:
    X1.append(features)
    y1.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
IMG_SIZE=256
X1 = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
y1=np.array(y)


# In[205]:


#ir data
X2 = []
y2 = []

for features,label in training_data3:
    X2.append(features)
    y2.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
IMG_SIZE=256
X2 = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
y2=np.array(y)


# In[17]:


X = []
y = []

for features,label in training_data1:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
IMG_SIZE=256
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
y=np.array(y)


# In[13]:


y.shape


# In[18]:


from tensorflow.keras.utils import to_categorical
#y = to_categorical(y,num_classes=1)
y = to_categorical(y, 332)
y.shape

