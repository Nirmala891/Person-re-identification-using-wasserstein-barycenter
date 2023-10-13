#!/usr/bin/env python
# coding: utf-8

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


# In[78]:


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


# In[79]:


bc=np.array(bary_centers_mix)
bc.shape


# In[80]:


l1=np.array(labels)
l1.shape


# In[ ]:





# In[81]:


#KNN-domain mix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  

#X_train, X_test, y_train, y_test = train_test_split(
            # bary_center, labels, test_size = 0.2, random_state=42)
X_train_bc=bc.reshape(119,512) 
#X_test_bc=dom1[0:6260].reshape(6260,512) 
y_train_bc=l1
#y_test_bc=y_ground[0:6260]
neighbors = np.arange(1, 2)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_bc = KNeighborsClassifier(n_neighbors=k)
    knn_bc.fit(X_train_bc, y_train_bc)


# In[83]:


X_.shape


# In[92]:


f1.shape


# In[113]:


from sklearn import metrics
X_=dom1[0:30]
X_=X_.reshape(30,512)
#X_=X_/np.sum(X_)
Pred_y = knn_bc.predict(X_)
metrics.accuracy_score(dom1_label[0:30], Pred_y)


# In[114]:


Pred_y


# In[115]:


dom1_label[0:100]


# In[29]:


import ot
y_ground=np.argmax(y, axis=1)
#barycenter calculation
reg=0.002
f=[]
labels=[]
bary_centers=[]

for i in range(len(inter_f)):

    
    if((y_ground[i]==y_ground[i+1]|((i+1)==6289)) ):
        
        f.append(inter_f[i])

    else:
        
        f1=np.array(f)
        labels.append(y_ground[i])
        l=len(f1)
        if(l>=10):
            weights=np.full(10, 1/10)
        else:
            weights=np.full(l,1/l)
        print("class-label",y_ground[i])
        print("input shape",f1.shape)
        print("weights",weights.shape)
        print("# samples",i)
        #f1 = f1 / np.sum(f1)
        w=len(weights)
        f2=ot.bregman.convolutional_barycenter2d(f1[0:w], reg,weights)
        bary_centers.append(f2)
        print("bc",f2)
        f=[]


# In[ ]:





# In[30]:


bc=np.array(bary_centers)
bc.shape


# In[31]:


l=np.array(labels)
l.shape


# In[32]:


l


# In[151]:


y_ground[6260]


# In[126]:


inter_f[0:254].shape


# In[127]:


l[0:8]


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  

#X_train, X_test, y_train, y_test = train_test_split(
            # bary_center, labels, test_size = 0.2, random_state=42)
X_train_bc=bc.reshape(258,512) 
X_test_bc=inter_f[0:6260].reshape(6260,512) 
y_train_bc=l
y_test_bc=y_ground[0:6260]
neighbors = np.arange(1, 2)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_bc = KNeighborsClassifier(n_neighbors=k)
    knn_bc.fit(X_train_bc, y_train_bc)


# In[34]:


from sklearn import metrics
#X_=X_test_bc[0:10]
#X_=X_/np.sum(X_)
#X_test_bc=X_test_bc/np.sum(X_test_bc)
Pred_y = knn_bc.predict(X_test_bc[0:3000])
metrics.accuracy_score(y_test_bc[0:3000], Pred_y)


# In[42]:


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
training_data2=[]
def create_training_data1():
    i=-1
    for category in rgb_cat1:  
        
        path = os.path.join(DATADIR,category)  
        
        i=i+1
        for img in tqdm(os.listdir(path)):
            try:
                #img_array = cv2.imread(os.path.join(path,img)) 
                img_array = cv2.imread(os.path.join(path,img)) 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                training_data2.append([new_array,cat1[i]-1])
                print(cat1[i])
                
            except OSError as e:
               print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
               print("general exception", e, os.path.join(path,img))
create_training_data1()


# In[45]:


X1 = []
y1 = []

for features,label in training_data2:
    X1.append(features)
    y1.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
IMG_SIZE=256
X1 = np.array(X1).reshape(-1, IMG_SIZE, IMG_SIZE,3)
y1=np.array(y1)


# In[46]:


plt.imshow(X[0])


# In[ ]:





# In[47]:


plt.imshow(X1[0])


# In[48]:


X[0]-X1[0]


# In[50]:


#cam2 test data 
feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="flatten").output,
)
intermediate_features_f1=[]
for i in range(len(X1[0:300])):
    c=X1[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    intermediate_features_f1.append(f0)

inter_f1=np.array(intermediate_features_f1)
    
inter_f1.shape


# In[51]:


inter_f1=inter_f1.reshape(300,512)


# In[53]:


#y_ground1=np.argmax(y1, axis=1)


# In[55]:


y_ground1=y1


# In[58]:


Pred_y = knn_bc.predict(inter_f1)
metrics.accuracy_score(y_ground1[0:300], Pred_y)


# In[59]:


Pred_y


# In[61]:


y_ground1[0:300]


# In[208]:


#cam3 test data
feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="flatten_8").output,
)
intermediate_features_f2=[]
for i in range(len(X2[0:300])):
    c=X2[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    intermediate_features_f2.append(f0)

inter_f2=np.array(intermediate_features_f2)
    
inter_f2.shape


# In[209]:


inter_f2=inter_f2.reshape(300,512)


# In[211]:


y_ground2=np.argmax(y2, axis=1)


# In[212]:


Pred_y = knn_bc.predict(inter_f2)
metrics.accuracy_score(y_ground2[0:300], Pred_y)


# In[214]:


Pred_y[100]


# In[220]:


y_ground2[100]


# In[221]:


a=X2[0]
plt.imshow(a)


# In[ ]:





# sample 0 and sample 29 belong to different classes. need to check which feature vector differentiates them well

# In[76]:


d1=np.array(inter_dense1)
d=np.array(inter_dense)
i=np.array(inter_conv4)
print("conv4-a",i[0])
print("conv4-b",i[29])
print("dense1-a",d1[0])
print("dense1-a",d1[29])
print("dense-a",d[0])
print("dense-a",d[29])


# barycenter computation with the dense(last dense layer) feature vectors
# 

# In[81]:


d=d.reshape(6289,332,1)


# In[85]:


d[29][1]


# In[86]:


y.shape


# In[146]:


i_conv4=np.array(inter_conv4)
i_conv4.shape


# In[151]:


i_conv4=i_conv4.reshape(6289,256,9)


# In[156]:


i_inter1=inter_inter1.reshape(6289,25,64)


# In[153]:


y_ground.shape


# In[157]:


import ot
y_ground=np.argmax(y, axis=1)
#barycenter calculation
reg=0.002
f=[]
labels=[]
bary_centers=[]

for i in range(len(i_inter1)):

    
    if((y_ground[i]==y_ground[i+1]|((i+1)==6289)) ):
        
        f.append(i_inter1[i])

    else:
        
        f1=np.array(f)
        labels.append(y_ground[i])
        l=len(f1)
        weights=np.full(5, 1/5)
        print("class-label",y_ground[i])
        print("input shape",f1.shape)
        print("weights",weights.shape)
        print("# samples",i)
        f1 = f1 / np.sum(f1)
        f2=ot.bregman.convolutional_barycenter2d(f1[0:5], reg,weights)
        bary_centers.append(f2)
        print("bc",f2)
        f=[]


# In[158]:


bc=np.array(bary_centers)
bc.shape


# In[112]:


f1=d[6269:6289]
l=len(f1)
weights=np.full(l, 1/l)
f1 = f1 / np.sum(f1)
f2=ot.bregman.convolutional_barycenter2d(f1, reg,weights)
bary_centers.append(f2)
labels.append(y_ground[6288])
print("bc",f2)


# In[164]:


bc=np.array(bary_centers)
bc.shape


# In[159]:


labels=np.array(labels)
labels.shape


# In[160]:


X_train, X_test, y_train, y_test = train_test_split(X[0:3730], y[0:3730], test_size=0.1,shuffle=False)


# In[205]:


y=np.argmax(y, axis=1)


# In[ ]:





# In[162]:


y_test.shape


# In[163]:


y_test=np.argmax(y_test, axis=1)


# In[165]:


25*64


# In[169]:


labels[153]


# In[206]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  

#X_train, X_test, y_train, y_test = train_test_split(
            # bary_center, labels, test_size = 0.2, random_state=42)
X_train_bc=bc.reshape(153,1600) 
X_test_bc=inter_inter1_test.reshape(10,1600) 
y_train_bc=labels[0:153]
y_test_bc=y[0:10]
neighbors = np.arange(1, 2)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_bc = KNeighborsClassifier(n_neighbors=k)
    knn_bc.fit(X_train_bc, y_train_bc)


# In[207]:


from sklearn import metrics
X_=X_test_bc[0:10]
X_=X_/np.sum(X_)
Pred_y = knn_bc.predict(X_)
metrics.accuracy_score(y[0:10], Pred_y)


# In[208]:


Pred_y


# In[209]:


y[0:10]


# In[143]:


y_test_bc.shape


# In[136]:


X_.shape


# In[ ]:





# In[140]:


sample=[]
for i in range(len(Pred_y)):
    indice1 = np.where(Pred_y[0] == np.amax(Pred_y[0]))
    sample.append(indice1)


# In[141]:


sample


# In[129]:


y_test_bc[0:10]


# In[38]:


from tensorflow.keras.models import save_model
save_model(model, "model_vanilla.h5")


# In[ ]:





# In[7]:


from tensorflow.keras.models import load_model
model=load_model( "model_vanilla.h5")


# In[39]:


import pickle

with open("sysu-mm01-correct.txt", 'wb') as fh:
   pickle.dump(training_data1, fh)


# In[26]:


import os
DATADIR="SYSU-MM01/cam2"
rgb_cat1=[]
entries = os.listdir(DATADIR)
for entry in entries:
  rgb_cat1.append(entry)
 
IMG_SIZE=256
rgb_cat1.sort()


# In[27]:


cat=np.array(rgb_cat1)
cat1=[]
for i in range(len(cat)):
    cat1.append(int(cat[i]))


# In[28]:


training_data3=[]


# In[29]:


def create_training_data():
    i=-1
    for category in rgb_cat1:  
        
        path = os.path.join(DATADIR,category)  
        
        i=i+1
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img)) 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                training_data3.append([new_array,cat1[i]-1])
                print(cat1[i])
                
            except OSError as e:
               print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
               print("general exception", e, os.path.join(path,img))


# In[30]:


create_training_data()


# In[31]:


tr=np.array(training_data3)


# In[34]:


tr[0][0]


# In[35]:


predicted_class=[]


# In[37]:


def test(c):
  a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  #b = np.array(d).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  x=model.predict(a)
  #prob =x
  #print(x)
  arr=np.array(x)
  #y=y.reshape(2,332)
  indice1 = np.where(x[0] == np.amax(x[0]))
  
  #indice2 = np.where(y[1] == np.amax(y[1]))
  #print(indice1)
  predicted_class.append(indice1)
  
  
    
    


# In[38]:


len(tr)


# In[39]:


i=0
for i in range(len(tr)):
    test(tr[i][0])
    


# In[42]:


predicted_class=np.array(predicted_class)
predicted_class=predicted_class.reshape(7528)


# In[51]:


predicted_class.shape


# In[48]:


ground=[]


# In[49]:


for i in range(len(tr)):
    ground.append(tr[i][1])
    


# In[50]:


ground=np.array(ground)
ground.shape


# In[52]:


from sklearn import metrics

metrics.accuracy_score(predicted_class, ground)


# In[58]:


t1=np.array(tr[0][0],dtype=np.float64)
t2=np.array(tr[1][0],dtype=np.float64)


# In[64]:


tr[0][0]


# In[66]:


X = []
y = []

for features,label in training_data3:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y=np.array(y)


# In[69]:


X=np.array(X,dtype=np.float64)


# In[75]:


X=X.reshape(7528,768,256)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


"""from tensorflow import keras
model = keras.models.load_model('model_vanilla.h5')"""


# In[42]:


model


# In[316]:


feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="dense").output,
)
intermediate_features=[]
for i in range(len(X)):
    c=X[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    intermediate_features.append(f0)

inter=np.array(intermediate_features)
    
inter.shape


# In[317]:


inter=inter.reshape(6289,4,512)


# In[293]:





# In[294]:





# In[295]:


inter.shape


# In[77]:


inter=inter.reshape(6289,4,256)


# In[68]:


y=np.argmax(y, axis=1)

    


# In[69]:


y.shape


# In[71]:


y


# In[72]:


X.shape


# In[160]:


import ot
reg=0.002


# In[161]:


inter=inter.reshape(6289,32,32)


# In[120]:


inter[64]


# In[121]:


y.shape


# In[325]:


inter=inter.reshape(6289,64,32)


# In[329]:


#barycenter calculation
reg=0.002
f=[]
labels=[]
bary_centers=[]
count=0
for i in range(len(inter)):

    
    if((y[i]==y[i+1])&(count<20) ):
        count=count+1
        f.append(inter[i])

    else:
        count=0
        f1=np.array(f)
        labels.append(y[i])
        l=len(f1)
        weights=np.full(l, 1/l)
        print("class-label",y[i])
        print("input shape",f1.shape)
        print("weights",weights.shape)
        print("# samples",i)
        #f1 = f1 / np.sum(f1)
        f2=ot.bregman.convolutional_barycenter2d(f1, reg,weights)
        bary_centers.append(f2)
        print("bc",f2)
        f=[]


# In[ ]:





# In[330]:


"""import pickle

with open("cam1-dense_features_barycenter.txt", 'wb') as fh:
   pickle.dump(bary_centers, fh)"""


# In[34]:


pickle_off = open ("cam1-dense_features_barycenter.txt", "rb")
bary_centers = pickle.load(pickle_off)


# In[35]:


b=np.array(bary_centers)
b.shape


# In[38]:


b[0]-b[20]


# In[212]:


bc[254]


# In[163]:


b=np.array(bary_centers)
b.shape


# In[172]:


zz=inter[6269:6289]


# In[173]:


len(zz)


# In[174]:


l


# In[175]:


weights.shape


# In[176]:


zz.shape


# In[177]:


l=len(zz)
weights=np.full(l, 0.01)
f1=zz
"""print("class-label",y[i])
print("input shape",f1.shape)
print("weights",weights.shape)
print("# samples",i)"""
f1 = f1 / np.sum(f1)
f2=ot.bregman.convolutional_barycenter2d(f1, reg,weights)
bary_centers.append(f2)
print("bc",f2)


# In[178]:


import pickle

with open("cam1-features_barycenter.txt", 'wb') as fh:
   pickle.dump(bary_centers, fh)


# In[ ]:





# In[179]:


bary_centers=np.array(bary_centers)
bary_centers.shape


# In[182]:


"""test_img=[]
for i in range(len(X_test)):
    z=X_test[i][:,:,2]
    z1=cv2.resize(z,(32,32))
    test_img.append(z1)"""
    


# In[183]:


test_img=np.array(test_img)
test_img.shape


# In[185]:


labels=np.array(labels)
labels.shape


# In[192]:


l=[]
l.append(labels)
l.append(y[6288])


# In[195]:


c=np.array(cat1)
c.shape


# In[196]:


y_test.shape


# In[197]:


y_test=np.argmax(y_test, axis=1)


# In[198]:


y_test.shape


# In[200]:


inter.shape


# In[1]:


bc=bary_centers


# In[216]:


bc[254]


# In[43]:


import math
bc=bary_centers


# In[44]:


for i in range(len(b)):
    for j in range(32):
        for k in range(32):
            if(math.isnan(bc[i][j][k])):
                print("hi")
                bc[i][j][k]=-1


# In[206]:


bc


# In[320]:


bc[0]-bc[2]


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[0:800], y[0:800], test_size=0.1, random_state=42,shuffle=True)


# In[47]:


feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="densez").output,
)
intermediate_features_test=[]
for i in range(len(X_test)):
    c=X_test[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    intermediate_features_test.append(f0)

inter_test=np.array(intermediate_features_test)
    
inter_test.shape


# In[50]:


bc=np.array(bc)
bc.shape


# In[51]:


inter_test=inter_test.reshape(80,64,32)


# In[52]:


y_test.shape


# In[54]:


labels=np.array(cat1)
labels.shape


# In[65]:


labels=[0,0,1,1,3,3,4,4,4,5,5,6,6,7,7,9,9,10,10,11,12,12,12,13,13,14,14,15,17,17,18,18,19,19,21,22,22,26,27,28,29,29,46,51,51,52,52,53,53,54,54,55,55,56]


# In[66]:


labels=np.array(labels)
labels.shape


# In[70]:


bc.shape


# In[80]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  

#X_train, X_test, y_train, y_test = train_test_split(
            # bary_center, labels, test_size = 0.2, random_state=42)
X_train_bc=bc.reshape(54,2048) 
X_test_bc=inter_test.reshape(80,2048) 
y_train_bc=labels
y_test_bc=y_test
neighbors = np.arange(1, 4)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_bc = KNeighborsClassifier(n_neighbors=k)
    knn_bc.fit(X_train_bc, y_train_bc)
      


# In[229]:


X_.shape


# In[83]:


from sklearn import metrics
X_=X_test_bc
#X_=X_/np.sum(X_)
Pred_y = knn_bc.predict(X_)
metrics.accuracy_score(y_test_bc, Pred_y)


# In[76]:


y_test_bc


# In[77]:


Pred_y


# In[227]:


X_test_bc[0]


# In[127]:


y[29]


# In[132]:


z=inter[29:54]
f1=z
l=len(f1)
weights=np.full(10, 1/10)
"""print("class-label",y[i])
print("input shape",f1.shape)
print("weights",weights.shape)
print("# samples",i)"""
f1 = f1 / np.sum(f1)
f2=ot.bregman.convolutional_barycenter2d(f1[0:10], reg,weights)
bary_centers.append(f2)
print("bc",f2)


# In[158]:


reg=0.002
weights=np.full(20, 0.01)
"""print("class-label",y[i])
print("input shape",f1.shape)
print("weights",weights.shape)
print("# samples",i)"""
f1 = f1 / np.sum(f1)
f2=ot.bregman.convolutional_barycenter2d(f1[0:20], reg,weights)
bary_centers.append(f2)
print("bc",f2)


# In[139]:


f1=f1.reshape(25,32,32)


# In[149]:


reg=0.2
weights=np.full(2, 1/2)
"""print("class-label",y[i])
print("input shape",f1.shape)
print("weights",weights.shape)
print("# samples",i)"""
f1 = f1 / np.sum(f1)
f2=ot.bregman.convolutional_barycenter2d(f1[23:25], reg,weights)
bary_centers.append(f2)
print("bc",f2)


# In[ ]:





# In[124]:


cd


# In[120]:


import re
training_data_market=[]
IMG_SIZE=256


# In[125]:


#market 1501
path="bary_center/bounding_box_train"
def create_training_data1():
    
     
        
    path = "bary_center/bounding_box_train"
    #i=i+1
    for img in tqdm(os.listdir("bary_center/bounding_box_train")):
        try:
            #img_array = cv2.imread(os.path.join(,img)) 
            img_array = cv2.imread(os.path.join(path,img)) 
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            res = re.findall("(\d+)_c", img)
            training_data_market.append([new_array,res[0]])
            print(img)
            print(res[0])

        except OSError as e:
           print("OSErrroBad img most likely", e, os.path.join(path,img))
        except Exception as e:
           print("general exception", e, os.path.join(path,img))


# In[126]:


create_training_data1()


# In[127]:


t_market=np.array(training_data_market)
t_market.shape


# In[170]:


X_m = []
y_m = []

for features,label in training_data_market:
    X_m.append(features)
    y_m.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
IMG_SIZE=256
X_m = np.array(X_m).reshape( -1,IMG_SIZE, IMG_SIZE,3)
y_m=np.array(y_m)


# In[140]:


y_m1=[]
for i in range(len(y_m)):
    y_m1.append(int(y_m[i]))
    
y_m1=np.array(y_m1)
    
    


# In[130]:


#labels of cam-1,2 from sysu-mm01 dataset
y_sysu=np.argmax(y, axis=1)


# In[168]:


X.shape


# In[171]:


X_m.shape


# In[164]:


y_total=[]
for i in range(len(y_sysu)):
    y_total.append(y_sysu[i])
    


# In[165]:


for i in range(len(y_m1)):
    y_total.append(y_m1[i])


# In[166]:


yy=np.array(y_total)
yy.shape


# In[167]:


from tensorflow.keras.utils import to_categorical
#y = to_categorical(y,num_classes=1)
yy = to_categorical(yy, 1832)
yy.shape


# In[172]:


X_total=[]
for i in range(len(X)):
    X_total.append(X[i])
for i in range(len(X_m)):
    X_total.append(X_m[i])
XX=np.array(X_total)
XX.shape


# In[187]:


model = keras.Model(inputs=resnet.input, outputs=model1)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from keras.losses import mean_squared_error, binary_crossentropy,categorical_crossentropy
"""opt=tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"
)"""
#opt = SGD(lr=0.001)
opt=Adam(lr=0.0002, beta_1=0.5)
#distance=Lambda(wasserstein_distance1,output_shape=wasserstein_dist_out_shape)([model1,model2])
#model.compile(loss =wasserstein_distance1, optimizer = opt,metrics="accuracy")
model.compile(loss=[categorical_crossentropy,earth_mover_loss,cont_loss],loss_weights=[1,0.1,0.01],optimizer=opt,metrics="accuracy")


# In[ ]:





# In[188]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.1, random_state=42,shuffle=True)


# In[180]:


y_test.shape


# In[190]:


history=model.fit(X_train,y_train,epochs=20,batch_size=64,validation_data=(X_test,y_test))


# In[193]:


#test data-cam1. available in X
feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="flatten_1").output,
)
features_1=[]
for i in range(len(X[0:6288])):
    c=X[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    features_1.append(f0)

f1=np.array(features_1)
    
f1.shape


# In[194]:


f1=f1.reshape(6288,512)


# In[197]:


y_ground=y[0:6288]


# In[198]:


y_ground=np.argmax(y_ground, axis=1)


# In[200]:


f1=f1.reshape(6288,32,16)


# In[201]:


#barycenter computation
bc_cam1=[]
import ot
y_ground=np.argmax(y, axis=1)
#barycenter calculation
reg=0.002
f=[]
labels=[]
bary_centers=[]

for i in range(len(f1)):

    
    if((y_ground[i]==y_ground[i+1]|((i+1)==6289)) ):
        
        f.append(f1[i])

    else:
        
        fz=np.array(f)
        labels.append(y_ground[i])
        l=len(fz)
        if(l>=10):
            weights=np.full(10, 1/10)
        else:
            weights=np.full(l,1/l)
        print("class-label",y_ground[i])
        print("input shape",fz.shape)
        print("weights",weights.shape)
        print("# samples",i)
        #f1 = f1 / np.sum(f1)
        w=len(weights)
        ff=ot.bregman.convolutional_barycenter2d(fz[0:w], reg,weights)
        bc_cam1.append(ff)
        print("bc",ff)
        f=[]


# In[202]:


bc_=np.array(bc_cam1)
bc_.shape


# In[203]:


labels=np.array(labels)
labels.shape


# In[204]:


X_train_bc=bc_.reshape(258,512) 
#X_test_bc=dom1[0:6260].reshape(6260,512) 
y_train_bc=labels
#y_test_bc=y_ground[0:6260]
neighbors = np.arange(1, 2)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train_bc, y_train_bc)


# In[210]:


y_ground.shape


# In[211]:


#X_=dom1[0:30]
#X_=X_.reshape(30,512)
#X_=X_/np.sum(X_)
test=f1.reshape(6288,512)
y_test=y_ground[0:6288]
Pred_y = knn_test.predict(test)
metrics.accuracy_score(y_test, Pred_y)


# In[207]:


f1.shape


# In[ ]:




