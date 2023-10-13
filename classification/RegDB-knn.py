#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from sklearn.model_selection import GridSearchCV, train_test_split


# In[3]:


import os
DATADIR="Visible"
rgb_cat1=[]
entries = os.listdir(DATADIR)
for entry in entries:
  rgb_cat1.append(entry)
 
#IMG_SIZE=256
rgb_cat1.sort()
cat=np.array(rgb_cat1)
cat1=[]
for i in range(len(cat)):
    cat1.append(int(cat[i]))


# In[69]:


rgb_cat1.sort()


# In[83]:


training_data2=[]
labels=[]


# In[84]:


def create_training_data1():
    i=-1
    for category in rgb_cat1:  
        
        path = os.path.join(DATADIR,category)  
        
        i=i+1
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img)) 
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                
                z =img_array[:, :,2]
                
                z=cv2.resize(z,(100,100))
                z = z / np.sum(z)
                training_data2.append(z)
                labels.append(cat1[i])
                #print(cat1[i])
                
                #b=np.array([z,z])
                #b1=ot.bregman.convolutional_barycenter2d(b, reg,weights)
                #bc.append(b1)
               # c=c+1
                print("cat1",cat1[i])
                
            except OSError as e:
               print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
               print("general exception", e, os.path.join(path,img))


# In[85]:


create_training_data1()


# In[86]:


training_data2


# In[87]:


labels


# In[88]:


X=np.array(training_data2)
y=np.array(labels
          )


# In[89]:


X.shape


# In[90]:


y.shape


# In[92]:


y[34]


# In[74]:


a=np.array(training_data2,dtype=object)
a.shape


# In[75]:


a[0].shape


# In[80]:


a[4010]


# In[46]:


X[0],y[0]


# In[51]:


X[10]


# In[53]:


plt.imshow(X[10])


# In[50]:


y[10]


# In[77]:


X = []
y = []

for features,label in training_data2:
    X.append(features)
    y.append(label)


# In[78]:


X=np.array(X)
y=np.array(y)
X.shape
y.shape


# In[81]:


y[0],y[4119]


# In[13]:


X.shape


# In[14]:


t=np.array(training_data2)


# In[19]:


import ot
reg=0.002


# In[33]:


t.shape


# In[36]:


t[2340]


# In[21]:


#barycenter calculation
f=[]
bary_centers=[]
for i in range(0,4120):
    if(i!=4120):
 
        if(t[i][1]==t[i+1][1]):
            f.append(t[i][0])

        else:
            f1=np.array(f)
            
            l=len(f1)
            weights=np.full(l, 1/(l))
            print("input shape",f1.shape)
            print("weights",weights.shape)
            print("# samples",i)
            f2=ot.bregman.convolutional_barycenter2d(f1, reg,weights)
            bary_centers.append(f2)
            print("bc",f2)
            f=[]


# In[95]:


y[4110]


# In[102]:


f=[]
for i in range(4110,4210):
    f.append(t[i][0])
    


# In[105]:


l=len(f)
weights=np.full(l, 1/(l))
f2=ot.bregman.convolutional_barycenter2d(f, reg,weights)
bary_centers.append(f2)


# In[106]:


bc=np.array(bary_centers)
bc.shape


# In[157]:


import pickle

with open("bary_center_regdb.txt", 'wb') as fh:
   pickle.dump(bary_centers, fh)


# In[107]:


cat.shape


# In[108]:


cat


# In[109]:


X.shape,y.shape


# In[110]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# In[111]:


X_test.shape


# In[131]:


cat1


# In[132]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  

#X_train, X_test, y_train, y_test = train_test_split(
            # bary_center, labels, test_size = 0.2, random_state=42)
X_train_bc=bc.reshape(412,10000) 
X_test_bc=X_test.reshape(824,10000) 
y_train_bc=cat1
y_test_bc=y_test
neighbors = np.arange(1, 2)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_bc = KNeighborsClassifier(n_neighbors=k)
    knn_bc.fit(X_train_bc, y_train_bc)
      
    
    #train_accuracy_bc[i] = knn_bc.score(X_train_bc, y_train_bc)
    #test_accuracy_bc[i] = knn_bc.score(X_test_bc, y_test_bc)


# In[133]:


from sklearn import metrics
Pred_y = knn_bc.predict(X_test_bc)
metrics.accuracy_score(y_test, Pred_y)


# In[126]:


y_test.dtype


# In[127]:


Pred_y=np.array(Pred_y,dtype=np.int64)


# In[134]:


print(y_test[:10], Pred_y[:10])


# In[136]:


X_train.shape


# In[137]:


#knn_vanilla
X_train_=X_train.reshape(3296,10000) 
X_test_=X_test.reshape(824,10000) 
y_train_=y_train
y_test_=y_test
neighbors = np.arange(1, 10)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_, y_train_)


# In[138]:


Pred_y = knn.predict(X_test_)
metrics.accuracy_score(y_test, Pred_y)


# In[154]:


case2.sort()


# In[144]:


case1=[]
case2=[]
case3=[]
case1_bc=[]
case2_bc=[]
case3_bc=[]

case0=[]
case0_bc=[]
def test(img):

    noise =  np.random.normal(loc=0, scale=1, size=img.shape)

    #
    noisy = np.clip((img + noise*0.2),0,1)
    noisy2 = np.clip((img + noise*0.4),0,1)

 
    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)


    img2 = img*2
    n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)
    
    pred= knn.predict(img.reshape(1,10000))
    pred1=knn_bc.predict(img.reshape(1,10000))
    case0.append(pred)
    case0_bc.append(pred1)
    pred= knn.predict(n4.reshape(1,10000))
    pred1=knn_bc.predict(n4.reshape(1,10000))
    case2.append(pred)
    case2_bc.append(pred1)
    pred= knn.predict(noisy4mul.reshape(1,10000))
    pred1=knn_bc.predict(noisy4mul.reshape(1,10000))
    case3.append(pred)
    case3_bc.append(pred1)
    


# In[145]:


i=0
for i in range(len(X_test)):
    
    test(X_test[i])


# In[156]:


metrics.accuracy_score(y_test, case3)


# In[149]:


metrics.accuracy_score(y_test, case0_bc)


# In[150]:


metrics.accuracy_score(y_test, case2_bc)


# In[151]:


metrics.accuracy_score(y_test, case3_bc)


# In[ ]:




