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
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten,Input 
from keras.layers import Conv2D, MaxPooling2D ,BatchNormalization
from keras import backend as K 
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os,cv2
import matplotlib.pyplot as plt





# In[19]:





# In[21]:


resnet = ResNet50(input_shape=X.shape[1:], weights='imagenet', include_top=False)


# In[22]:


for layer in resnet.layers:
  layer.trainable = False
  layer._name = layer._name + '_resnet'


# In[23]:


model1_feature=[]


# In[27]:


input_shape=X.shape[1:]
input1 = Input(shape=input_shape)
#input2 = Input(shape=input_shape)

conv_1a = Conv2D(32, (3,3), padding='same', activation='relu')
#conv_1b = Conv2D(32, (3,3), padding='same', activation='relu')
model1 = conv_1a(resnet.output)
#model2 = conv_1b(vgg2.output)


conv_2a = Conv2D(64, (3,3), padding='same', activation='relu')
pool_1a=MaxPooling2D(pool_size=(2,2),strides=1,padding='valid')
conv_3a = Conv2D(128, (3,3), padding='same', activation='relu')
pool_2a=MaxPooling2D(pool_size=(2,2),strides=1,padding='valid')
conv_4a = Conv2D(128, (3,3), padding='same', activation='relu')
model1=pool_1a(model1)
#model2=pool_1b(model2)

model1=conv_2a(model1)
#model2=conv_2b(model2)
model1=BatchNormalization()(model1)
#model2=BatchNormalization()(model2)
model1=pool_1a(model1)
#model2=pool_1b(model2)

model1=conv_3a(model1)
#model2=conv_3b(model2)
model1=BatchNormalization()(model1)
#model2=BatchNormalization()(model2)
model1=pool_2a(model1)
#model2=pool_2b(model2)

model1=conv_4a(model1)
#model2=conv_4b(model2)




conv1 = Conv2D(64, (3,3), padding='same', activation='relu',name="inter1") # shared
conv2 = Conv2D(128, (3,3), padding='same', activation='relu',name="inter2")
conv3=Conv2D(256, (3,3),padding='same',activation="relu",name="inter3") 
conv4=Conv2D(256, (3,3),padding='same',activation="relu",name="inter4") 

model1=conv1(model1)
#model2=conv1(model2)
model1=BatchNormalization()(model1)
#model2=BatchNormalization()(model2)
pool1=MaxPooling2D(pool_size=(2,2),strides=1,padding='valid')
model1=pool1(model1)
#model2=pool1(model2)

model1=conv2(model1)
#model2=conv2(model2)
model1=BatchNormalization()(model1)
#model2=BatchNormalization()(model2)
model1=pool1(model1)
#model2=pool1(model2)

model1=conv3(model1)
#model2=conv3(model2)
model1=BatchNormalization()(model1)
#model2=BatchNormalization()(model2)
model1=pool1(model1)
#model2=pool1(model2)

model1=conv4(model1)
#model2=conv4(model2)
model1=BatchNormalization()(model1)
#model2=BatchNormalization()(model2)
model1=pool1(model1)
#model2=pool1(model2)
model1_feature.append(model1)
model1=Dense(512,activation="relu",name="dense1")(model1)
#model2=Dense(512,activation="relu")(model2)

model1=Flatten()(model1)
#model2=Flatten()(model2)
model1 = Dense(250, activation='softmax',kernel_regularizer=l2(1e-3))(model1)
#model2 = Dense(332, activation='softmax',kernel_regularizer=l2(1e-3))(model2)


# In[29]:


model = keras.Model(inputs=resnet.input, outputs=model1)


# In[30]:


model.summary()


# In[186]:


def cont_loss(y_true,y_pred):#CONTRASTIVE LOSS FUNCTION

  margin = 1
  square_pred = K.square(y_pred)
  #square_pred=tf.cast(square_pred, tf.int64)
  margin_square = K.square(K.maximum(margin - y_pred, 0))
  #margin_square=tf.cast(margin_square, tf.int64)
  return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# In[ ]:


def earth_mover_loss(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


# In[23]:


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
model.compile(loss=[categorical_crossentropy,earth_mover_loss,cont_loss],loss_weights=[1,0.5,0.5],optimizer=opt,metrics="accuracy")


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,shuffle=True)


# In[71]:


"""X_train=X[0:5500]
y_train=y[0:5500]
X_test=X[5500:6288]
y_test=y[5500:6288]"""


# In[28]:


"""X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y=np.array(y)"""


# In[29]:


"""from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 332)
y_test=to_categorical(y_test,332)
#y1 = to_categorical(y1, 332)
y_train.shape"""


# In[276]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[31]:


y_train[0]


# In[32]:


X_test.shape,y_test.shape


# In[25]:


history=model.fit(X_train,y_train,epochs=50,batch_size=64,validation_data=(X_test,y_test))


# In[78]:


print(history.history.keys())


# In[79]:


history


# In[80]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[81]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[53]:


z=np.array(X[100]).reshape(-1, IMG_SIZE, IMG_SIZE,3)
p=model.predict(z)
p1=np.where(p[0] == np.amax(p[0]))
p1


# In[73]:


#prediction module to test the accuracy of random samples from train and test data
y_pred=[]
i=0
for i in range(len(X)):
    z=np.array(X[i]).reshape(-1, IMG_SIZE, IMG_SIZE,3)
    p=model.predict(z)
    p1=np.where(p[0] == np.amax(p[0]))
    y_pred.append(p1)
    


# In[39]:


X_train.shape


# In[41]:


X[i].shape


# In[74]:


y_pred=np.array(y_pred)
y_pred=y_pred.reshape(6289)
y_pred.shape


# In[75]:


y_pred.shape


# In[76]:


#accuracy check
from sklearn import metrics
y_pred=np.array(y_pred)
y_ground=np.argmax(y, axis=1) #converting one hot vector to an array(into its original format)
metrics.accuracy_score(y_pred,y_ground)


# In[27]:


c1=X[0:6288]
cy1=y[0:6288]
c2=X[6289:13817]
cy2=y[6289:13817]


# The model is able to achieve 98.21% accuracy when tested with the entire cam1 data(train+test images)
# The next step is to extract features from last convolution layer
# 

# In[37]:


feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="flatten").output,
)
features_1=[]
for i in range(len(c1)):
    c=c1[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    features_1.append(f0)

f1=np.array(features_1)
    
f1.shape


# In[87]:


f1.shape


# In[39]:


feature_extractor = keras.Model(
   inputs=resnet.input,
   outputs=model.get_layer(name="flatten").output,
)
features_2=[]
for i in range(len(c2)):
    c=c2[i]
    a = np.array(c).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    f0=feature_extractor(a)
    features_2.append(f0)

f2=np.array(features_2)
    
f2.shape


# In[32]:


cy1=np.argmax(cy1, axis=1)
cy2=np.argmax(cy2,axis=1)


# In[41]:


f1=f1.reshape(6288,32,16)
f2=f2.reshape(7528,32,16)


# In[43]:


dom1=f1[0:3000]
dom2=f2[0:3000]
dom1_label=cy1[0:3000]
dom2_label=cy2[0:3000]


# In[45]:


#mixup
"""f=[]
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
    mixup_input.append([arr,cy1[i],cy2[i]])"""


# In[133]:


f_mixup=f_mixup.reshape(6234,20,32,16)


# In[62]:





# In[63]:


f1.shape


# In[76]:


dom1_label[0:100]


# In[77]:


dom2_label[0:100]


# In[ ]:




