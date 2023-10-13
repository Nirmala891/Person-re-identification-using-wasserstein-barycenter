#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[6]:


from sklearn.model_selection import GridSearchCV, train_test_split


# In[7]:


#rmse example code

"""from sklearn.metrics import mean_squared_error
from math import sqrt
train_preds = knn.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
rmse"""


# In[8]:


#training_data1=[]


# In[9]:


"""pickle_off = open ("sysu-mm01.txt", "rb")
training_data2 = pickle.load(pickle_off)"""


# In[10]:


"""import numpy
IMG_SIZE=100
X = []
y = []

for features,label in training_data2:
    X.append(features[0:100,0:100])
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y=np.array(y)"""


# In[11]:


import os
DATADIR="SYSU-MM01/cam1"
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


# In[12]:


training_data2=[]


# In[13]:


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
                training_data2.append([z,cat1[i]-1])
                #print(cat1[i])
                
                #b=np.array([z,z])
                #b1=ot.bregman.convolutional_barycenter2d(b, reg,weights)
                #bc.append(b1)
               # c=c+1
                print("c1",i)
                
            except OSError as e:
               print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
               print("general exception", e, os.path.join(path,img))


# In[14]:


create_training_data1()


# In[15]:


X = []
y = []

for features,label in training_data2:
    X.append(features)
    y.append(label)


# In[16]:


X2=np.array(X)
X2.shape


# In[17]:


import math
math.sqrt(259)


# In[139]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  
X_train, X_test, y_train, y_test = train_test_split(
             X2, y, test_size = 0.2, random_state=42)
X_train=X_train.reshape(5031,10000) 
X_test=X_test.reshape(1258,10000) 
neighbors = np.arange(1,7 )
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
      
    
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[140]:


from sklearn import metrics
Pred_y = knn.predict(X_test)
metrics.accuracy_score(y_test, Pred_y)


# In[20]:


Pred_y1 = knn.predict(X_train)
metrics.accuracy_score(y_train, Pred_y1)


# In[ ]:





# In[ ]:





# In[324]:


#X1=X1.reshape(6289,10000)


# In[325]:


#neighs = KNeighborsClassifier(n_neighbors=259)
#neighs.fit(X, y)
"""X_embedded = TSNE(n_components=1,perplexity=20).fit_transform(X1)
neighs_tsne = KNeighborsClassifier(n_neighbors=259)
neighs_tsne.fit(X_embedded, y)
"""


# In[327]:


X1[10]


# In[328]:


y[10]


# In[335]:


a=X1[100].reshape(-1,1)
print(neighs_tsne.predict(a))


# In[336]:


print(neighs.predict(a))


# In[ ]:





# In[ ]:


Pred_y = knn.predict(X_test)
metrics.accuracy_score(y_test, Pred_y)


# In[342]:


X_train.shape


# In[344]:


a=a.reshape(1,10000)


# In[340]:


y[100]


# In[345]:


#a=X1[100].reshape(-1,1)
print(knn.predict(a))


# In[368]:


t[6288][:1]


# In[22]:


t=np.array(training_data2)
t.shape


# In[364]:


t[6288]


# In[370]:


f=[]
bary_centers=[]


# In[372]:


t[0][0]


# In[373]:


t[0][1]


# In[376]:


f=[]


# In[378]:





# In[379]:


f


# In[21]:


class KNearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)


    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        dists = np.sqrt(np.sum(np.square(self.X_train), axis=1) + np.sum(np.square(X), axis=1)[:, np.newaxis] - 2 * np.dot(X, self.X_train.T))
        pass
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.y_train[sorted_dist[0:k]])
            pass
            y_pred[i]= (np.argmax(np.bincount(closest_y)))
            pass
        return y_pred


# In[77]:


num_training = 5031
num_test=1258


# In[87]:


#barycenter knn
num_folds = 3
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50]

X_train, X_test, y_train, y_test = train_test_split(
             bc1, y, test_size = 0.2, random_state=42)
X_train=X_train.reshape(5031,10000) 
X_test=X_test.reshape(1258,10000) 
X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)
k_to_accuracies = {}



for k in k_choices:
    k_to_accuracies[k] = []
    for num_knn in range(0,num_folds):
        X_test = X_train_folds[num_knn]
        y_test = y_train_folds[num_knn]
        X_train = X_train_folds
        y_train = y_train_folds
        
        temp = np.delete(X_train,num_knn,0)
        X_train = np.concatenate((temp),axis = 0)
        y_train = np.delete(y_train,num_knn,0)
        y_train = np.concatenate((y_train),axis = 0)
        
        classifier = KNearestNeighbor()
        classifier.train(X_train, y_train)
        dists = classifier.compute_distances(X_test)
        y_test_pred = classifier.predict_labels(dists, k)

        num_correct = np.sum(y_test_pred == y_test)
        accuracy = float(num_correct) / num_test
#         print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
        k_to_accuracies[k].append(accuracy)


print("Printing our 5-fold accuracies for varying values of k:")
print()
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# In[88]:


for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# In[89]:


best_k = 7

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Computing and displaying the accuracy for best k found during cross-validation
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# In[90]:


#vanilla knn
num_folds = 3
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50]

X_train, X_test, y_train, y_test = train_test_split(
             X2, y, test_size = 0.2, random_state=42)
X_train=X_train.reshape(5031,10000) 
X_test=X_test.reshape(1258,10000) 
X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)
k_to_accuracies = {}



for k in k_choices:
    k_to_accuracies[k] = []
    for num_knn in range(0,num_folds):
        X_test = X_train_folds[num_knn]
        y_test = y_train_folds[num_knn]
        X_train = X_train_folds
        y_train = y_train_folds
        
        temp = np.delete(X_train,num_knn,0)
        X_train = np.concatenate((temp),axis = 0)
        y_train = np.delete(y_train,num_knn,0)
        y_train = np.concatenate((y_train),axis = 0)
        
        classifier = KNearestNeighbor()
        classifier.train(X_train, y_train)
        dists = classifier.compute_distances(X_test)
        y_test_pred = classifier.predict_labels(dists, k)

        num_correct = np.sum(y_test_pred == y_test)
        accuracy = float(num_correct) / num_test
#         print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
        k_to_accuracies[k].append(accuracy)


print("Printing our 5-fold accuracies for varying values of k:")
print()
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# In[91]:


for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# In[98]:


#test_accuracy
best_k = 7

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Computing and displaying the accuracy for best k found during cross-validation
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# In[ ]:





# In[23]:


t=np.array(training_data2)
t.shape


# In[44]:


t[0][0].dtype


# In[136]:





# In[138]:


len(f)


# In[142]:


#barycenter calculation
"""f=[]
bary_centers=[]
for i in range(0,6288):
    if(i!=6288):
 
        if(t[i][1]==t[i+1][1]):
            f.append(t[i][0])

        else:
            f1=np.array(f)
            print(f1.shape)
            l=len(f1)
            weights=np.full(l, 1/(l))
            print(f1.shape)
            print(weights.shape)
            print(i)
            f2=ot.bregman.convolutional_barycenter2d(f1, reg,weights)
            bary_centers.append(f2)
            f=[]"""
                


# In[143]:


"""bary_centers1=np.array(bary_centers)
bary_centers1.shape"""


# In[151]:


labels = np.arange(0,258)
labels


# In[24]:


f=[]
labels=[]
#labels.append(t[0][1])
for i in range(0,6289):
    if(i!=6288):
 
        if(t[i][1]==t[i+1][1]):
            continue

        else:
            labels.append(t[i][1])
            f=[]
labels.append(t[6288][1])


# In[162]:


t[6288][1]


# In[166]:


labels


# In[58]:


import pickle
pickle_off = open ("bary_center.txt", "rb")
bary_center = pickle.load(pickle_off)


# In[59]:


bary_center=np.array(bary_center)
bary_center.shape


# In[60]:


bary_center.dtype


# In[61]:


bary_center


# while calculating the barucenter for all samples in a class, we get 259 barycenters for 259 classes. therefore only one input representation for one class. hence we the 259 samples cannot be split into train and test data just like that, as none of the test data will have a common sample in the training data. this is because all 259 class samples are different. therefore the training samples will be all the barycenter samples(=259). test data samples are given as raw images(same as the vanilla knn)

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(
             X2, y, test_size = 0.2, random_state=42)
X_test1=np.array(X_test)
X_test1.shape


# In[28]:


y_test_bc=np.array(y_test)
y_test_bc.shape


# In[29]:


labels=np.array(labels)
labels.shape


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  

#X_train, X_test, y_train, y_test = train_test_split(
            # bary_center, labels, test_size = 0.2, random_state=42)
X_train_bc=bary_center.reshape(259,10000) 
X_test_bc=X_test1.reshape(1258,10000) 
y_train_bc=labels
y_test_bc=y_test_bc
neighbors = np.arange(1, 2)
train_accuracy_bc = np.empty(len(neighbors))
test_accuracy_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_bc = KNeighborsClassifier(n_neighbors=k)
    knn_bc.fit(X_train_bc, y_train_bc)
      
    
    train_accuracy_bc[i] = knn_bc.score(X_train_bc, y_train_bc)
    test_accuracy_bc[i] = knn_bc.score(X_test_bc, y_test_bc)
  


# In[31]:


train_accuracy_bc


# In[32]:


test_accuracy_bc


# In[33]:


from sklearn.metrics import mean_squared_error
from math import sqrt
pred1 = knn.predict(X_test.reshape(1258,10000))
mse = mean_squared_error(y_test, pred1)
rmse = sqrt(mse)
rmse


# In[34]:


pred2 = knn_bc.predict(X_test.reshape(1258,10000))
mse = mean_squared_error(y_test, pred2)
rmse = sqrt(mse)
rmse


# In[35]:


from sklearn import metrics
Pred_y = knn.predict(X_test.reshape(1258,10000))
metrics.accuracy_score(y_test, Pred_y)


# In[36]:


Pred_y1 = knn_bc.predict(X_test.reshape(1258,10000))
metrics.accuracy_score(y_test, Pred_y1)


# In[ ]:





# In[76]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
#img = cv2.imread("X_test[0]")[...,::-1]/255.0
noise =  np.random.normal(loc=0, scale=1, size=img.shape)

# noise overlaid over image
noisy = np.clip((img + noise*0.2),0,1)
noise2 = (noise - noise.min())/(noise.max()-noise.min())
img2 = img*2
n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.01)), (1-img2+1)*(1 + noise*0.01)*-1 + 2)/2, 0,1)
plt.figure(figsize=(20,20))
plt.imshow(np.vstack((np.hstack((img, noise2)),
                      np.hstack((img, n2)))))


# In[74]:


n=n2[:,:,2]
n=cv2.resize(n,(10000,1))

pred= knn.predict(n)
pred1=knn_bc.predict(n)
pred,pred1


# In[75]:


img1=img2[:,:,2]
##n=n2[:,:,2]

n=cv2.resize(img1,(10000,1))
pred= knn.predict(n)
pred1=knn_bc.predict(n)
pred,pred1


# In[63]:





# In[70]:


plt.imshow(X_test[0])


# In[71]:


plt.imshow(img1)


# In[72]:


plt.imshow(n)


# In[73]:


pred


# In[77]:


n


# In[78]:


n.shape


# In[81]:


plt.imshow(X_test[0])


# In[82]:


pred= knn.predict(X_test[0].reshape(1,10000))
pred1=knn_bc.predict(X_test[0].reshape(1,10000))
pred,pred1


# In[84]:


y_test[0]


# In[93]:


img = X_test[0]
# noise overlaid over image
noise =  np.random.normal(loc=0, scale=1, size=img.shape)

# noise overlaid over image
noisy = np.clip((img + noise*0.2),0,1)
noisy2 = np.clip((img + noise*0.4),0,1)

# noise multiplied by image:
# whites can go to black but blacks cannot go to white
noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

# noise multiplied by bottom and top half images,
# whites stay white blacks black, noise is added to center
img2 = img*2
n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)
"""plt.imshow(np.vstack((np.hstack((img, noise2)),
                      np.hstack((img, n2)))))"""


# In[89]:


n2.shape


# In[90]:


pred= knn.predict(n2.reshape(1,10000))
pred1=knn_bc.predict(n2.reshape(1,10000))
pred,pred1


# In[91]:


n2


# In[92]:


X_test[0]


# In[94]:


n4


# In[95]:


pred= knn.predict(n4.reshape(1,10000))
pred1=knn_bc.predict(n4.reshape(1,10000))
pred,pred1


# In[96]:


pred= knn.predict(noisy4mul.reshape(1,10000))
pred1=knn_bc.predict(noisy4mul.reshape(1,10000))
pred,pred1


# In[135]:


#test image 2
"""case1=[]
case2=[]
case3=[]
case1_bc=[]
case2_bc=[]
case3_bc=[]"""

case0=[]
case0_bc=[]
def test(img):

    """noise =  np.random.normal(loc=0, scale=1, size=img.shape)

    #
    noisy = np.clip((img + noise*0.2),0,1)
    noisy2 = np.clip((img + noise*0.4),0,1)

 
    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)


    img2 = img*2
    n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)"""
    
    pred= knn.predict(img.reshape(1,10000))
    pred1=knn_bc.predict(img.reshape(1,10000))
    case0.append(pred)
    case0_bc.append(pred1)
    
"""pred= knn.predict(n4.reshape(1,10000))
pred1=knn_bc.predict(n4.reshape(1,10000))
case2.append(pred)
case2_bc.append(pred1)
pred= knn.predict(noisy4mul.reshape(1,10000))
pred1=knn_bc.predict(noisy4mul.reshape(1,10000))
case3.append(pred)
case3_bc.append(pred1)"""
    


# In[136]:


i=0
for i in range(len(X_test)):
    
    test(X_test[i])


# In[112]:


case1=np.array(case1)
case1.shape


# In[116]:


case2=np.array(case2)
case3=np.array(case3)
case1_bc=np.array(case1_bc)
case2_bc=np.array(case2_bc)
case3_bc=np.array(case3_bc)


# In[114]:


c=np.array(y_test)
c.shape


# In[115]:


from sklearn import metrics


metrics.accuracy_score(c, case1)


# In[117]:


metrics.accuracy_score(c, case2)


# In[118]:


metrics.accuracy_score(c, case3)


# In[119]:


metrics.accuracy_score(c, case1_bc)


# In[120]:


metrics.accuracy_score(c, case2_bc)


# In[121]:


metrics.accuracy_score(c, case3_bc)


# In[137]:


case0=np.array(case0)
case0_bc=np.array(case0_bc)
metrics.accuracy_score(c, case0)


# In[138]:


metrics.accuracy_score(c, case0_bc)


# In[108]:


y_test[2]


# In[99]:


y_test[1]


# In[100]:


X_test[1]


# In[103]:


noise2


# In[123]:


img=cv2.imread("0001-class5.jpg")[:,:,2]
img.shape


# In[127]:


img = img/ np.sum(img)


# In[128]:


img=cv2.resize(img,(100,100))
pred= knn.predict(img.reshape(1,10000))
pred1=knn_bc.predict(img.reshape(1,10000))
pred,pred1


# In[125]:





# In[126]:


bary_center[4]


# In[209]:


import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# In[154]:


labels.shape


# In[117]:


x_train.shape


# In[210]:


x_train=x_train[:,:,:,2]


# In[119]:


x_train.shape


# In[120]:


y_train.shape


# In[211]:


x_test=x_test[:,:,:,2]
x_test.shape


# In[212]:


y_train=y_train.reshape(50000,)
y_test=y_test.reshape(10000,)


# In[125]:


a=y_train
#corret method to append arrays
a1=np.append(a,y_test,0)


# In[ ]:





# In[126]:


a1.shape


# In[127]:


cifar_data=[]
cifar_data.append(x_train)
cifar_data=np.array(cifar_data)
cifar_data.shape


# In[128]:


cifar_data=cifar_data.reshape(50000,32,32)


# In[162]:


cifar_data1=np.append(cifar_data,x_test,0)
cifar_data1.shape


# In[208]:


X_test_cifar.shape


# In[215]:


x_train=x_train.reshape(50000,1024)
x_test=x_test.reshape(10000,1024)


# In[219]:


X_train_cifar=x_train 
X_test_cifar=x_test 
y_train_cifar=y_train
y_test_cifar=y_test
neighbors = np.arange(1, 5)
train_accuracy_cifar = np.empty(len(neighbors))
test_accuracy_cifar = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_cifar = KNeighborsClassifier(n_neighbors=k)
    knn_cifar.fit(X_train_cifar, y_train_cifar)
      
    
    train_accuracy_cifar[i] = knn_cifar.score(X_train_cifar, y_train_cifar)
    test_accuracy_cifar[i] = knn_cifar.score(X_test_cifar, y_test_cifar)
  


# In[217]:


train_accuracy_cifar


# In[218]:


test_accuracy_cifar


# In[ ]:


train_accuracy_cifar


# In[ ]:


test_accuracy_cifar


# In[160]:


cifar_input=[]
cifar_input.append(X_train_cifar)
cifar_input.append(X_test_cifar)
cifar_labels=[]
cifar_labels.append(y_train_cifar)
cifar_labels.append(y_test_cifar)


# In[190]:


cifar_labels=np.array(cifar_labels)
cifar_input=np.array(cifar_input)


# In[192]:


cifar_labels[0].shape


# In[21]:


a1.shape


# In[22]:


a1[3728]


# In[28]:


x_train


# In[31]:


cifar_data1.shape


# In[163]:


f1=[]
f2=[]
f3=[]
f4=[]
f5=[]
f6=[]
f7=[]
f8=[]
f9=[]
f10=[]
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]
l6=[]
l7=[]
l8=[]
l9=[]
l10=[]
bary_centers_cifar=[]
for i in range(0,60000):
    
    if(a1[i]==0):
        f1.append(cifar_data1[i])
        l1.append(a1[i])
    elif(a1[i]==1):
        f2.append(cifar_data1[i])
        l2.append(a1[i])
    elif(a1[i]==2):
        f3.append(cifar_data1[i])
        l3.append(a1[i])
    elif(a1[i]==3):
        f4.append(cifar_data1[i])
        l4.append(a1[i])
    elif(a1[i]==4):
        f5.append(cifar_data1[i])
        l5.append(a1[i])
    elif(a1[i]==5):
        f6.append(cifar_data1[i])
        l6.append(a1[i])
    elif(a1[i]==6):
        f7.append(cifar_data1[i])
        l7.append(a1[i])
    elif(a1[i]==7):
        f8.append(cifar_data1[i])
        l8.append(a1[i])
    elif(a1[i]==8):
        f9.append(cifar_data1[i])
        l9.append(a1[i])
    elif(a1[i]==9):
        f10.append(cifar_data1[i])
        l10.append(a1[i])
        

   


# In[143]:


ll=[]
ll.append(l1)
ll.append(l2)
ll.append(l3)
ll.append(l4)
ll.append(l5)
ll.append(l6)
ll.append(l7)
ll.append(l8)
ll.append(l9)
ll.append(l10)


# In[33]:


f1=np.array(f1)
f1.shape


# In[ ]:





# In[164]:


cifar_input=[]
cifar_input.append(f1)
cifar_input.append(f2)
cifar_input.append(f3)
cifar_input.append(f4)
cifar_input.append(f5)
cifar_input.append(f6)
cifar_input.append(f7)
cifar_input.append(f8)
cifar_input.append(f9)
cifar_input.append(f10)


# In[165]:


cifar_input=np.array(cifar_input)
cifar_input.shape


# In[78]:


f1


# In[144]:


cifar_input=np.array(cifar_input)
cifar_input.shape


# In[145]:


ll=np.array(ll)
ll.shape


# In[146]:


ll


# In[147]:


f1=np.array(f1)
f1.shape


# In[38]:


len(cifar_input[0])


# In[39]:


for i in range(0,3):
    print(i)


# In[55]:


cifar_input[0][0:10].shape


# In[191]:


import ot
reg=0.002
bary_centers_cifar=[]
for i in range(0,10):
    l=len(cifar_input[i][0:10])
    
    weights=np.full(l, 1/(l))
    data=[]
    data.append(cifar_input[i][0:10])
    data=np.array(data)
    #data = data / np.sum(data)
    data=data.reshape(10,32,32)
    print(weights.shape)
    print(i)
    z=ot.bregman.convolutional_barycenter2d(data, reg,weights)
    bary_centers_cifar.append(z)
    print(z)


# In[201]:


a=np.array(X_train_cifar_bc)
a.shape


# In[202]:


b=np.array(X_test_cifar_bc)
b.shape


# In[206]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
  
X_train_cifar_bc=bary_centers_cifar
X_train_cifar_bc=np.array(X_train_cifar_bc)
X_train_cifar_bc=X_train_cifar_bc.reshape(10,1024)
X_test_cifar_bc=x_test 
X_test_cifar_bc=np.array(X_test_cifar_bc)
X_test_cifar_bc=X_test_cifar_bc.reshape(10000,1024)
y_train_cifar_bc=np.arange(0,10)
y_test_cifar_bc=y_test
neighbors = np.arange(1, 10)
train_accuracy_cifar_bc = np.empty(len(neighbors))
test_accuracy_cifar_bc = np.empty(len(neighbors))
  

for i, k in enumerate(neighbors):
    knn_cifar_bc = KNeighborsClassifier(n_neighbors=k)
    knn_cifar_bc.fit(X_train_cifar_bc, y_train_cifar_bc)
      
    
    train_accuracy_cifar_bc[i] = knn_cifar_bc.score(X_train_cifar_bc, y_train_cifar_bc)
    test_accuracy_cifar_bc[i] = knn_cifar_bc.score(X_test_cifar_bc, y_test_cifar_bc)
  


# In[207]:


test_accuracy_cifar_bc


# In[194]:


cifar_input[4][0]


# In[72]:


data=[]
data.append(cifar_input[0][0:100])
    
data = data / np.sum(data)


# In[73]:


data.shape


# In[64]:


cifar_input[0][0]


# In[66]:


bary_centers_cifar=np.array(bary_centers_cifar)
bary_centers_cifar[1]


# In[57]:


cifar_input[0]


# In[43]:


weights.dtype


# In[46]:


cifar_input=np.array(cifar_input,dtype=np.float64)


# In[47]:


cifar_input.dtype


# In[49]:


cifar_input[0].shape


# In[50]:


cifar_input.shape


# In[51]:


cifar_input=cifar_input.reshape(10,6000,32,3)


# In[77]:


plt.imshow(cifar_input[0])


# In[107]:


f1


# In[134]:


cifar_data1=np.array(cifar_data1,dtype=np.float64)
cifar_data1.shape


# In[149]:


f1.dtype


# In[150]:


x_train.dtype


# In[167]:


cifar_input[0][0]


# In[168]:


cifar_input=np.array(cifar_input,dtype=np.float64)
cifar_input[0][0]


# In[174]:


f1=np.array(f1,dtype=np.float64)


# In[187]:


#l=len(cifar_input[i][0:100])
reg=0.002
sample=cifar_input[0][0:20]
l=len(sample)
weights=np.full(l, 1/(l))
"""data=[]
data.append(cifar_input[0][0:11])
data=np.array(data)"""
sample = sample/ np.sum(sample)
#data=data.reshape(l,32,3)
print(weights.shape)
#print(data.shape)
#print(i)
z=ot.bregman.convolutional_barycenter2d(sample, reg,weights)
#bary_centers_cifar.append(z)
print(z)


# In[181]:


weights


# In[109]:


sample


# In[110]:


x_train.shape


# In[111]:


import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# In[113]:


x_train.shape


# In[114]:


x_train1=x_train[:,:,:,2]


# In[115]:


x_train1.shape


# In[151]:


f1[0].shape


# In[152]:


plt.imshow(f1[0])


# In[161]:


cifar_data1[0]


# In[160]:


f1[0]


# In[ ]:




