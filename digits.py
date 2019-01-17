#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m

#import the dataset
dataset=pd.read_csv('train.csv')
X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0:1].values

#categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
Y=onehotencoder.fit_transform(Y).toarray()

#splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/4,random_state=0)

'''#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#fitting the model into training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)
'''
#ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

classifier=Sequential()

#input layer
classifier.add(Dense(output_dim=20,init='uniform',activation='relu',input_dim=784))

#hidden layer
classifier.add(Dense(output_dim=20,init='uniform',activation='relu'))

#output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,Y_train,epochs=100,batch_size=10)


ar=[]
#predicting the test set results
pred=classifier.predict(X_test)

for i in range(0,10499):
    for j in range(0,10):
     if pred[i][j]==pred[i].max() :
      pred[i][j]=1
     else:
      pred[i][j]=0
      

Y_pred=onehotencoder.inverse_transform(pred)

Y_test=onehotencoder.inverse_transform(Y_test)    

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)


s=int(m.sqrt(cm.size))
sum1=0
sum2=0 

for i in range(0,s):
    for j in range(0,s):
            if i==j:
                sum1 = sum1 + cm[i][j]
            else:
                sum2 = sum2 + cm[i][j]
                
total=sum1+sum2                
Accuracy=(sum1/total)*100            
print("The accuracy for the given test set is " + str(float(Accuracy)) + "%")

dataset1=pd.read_csv('test.csv')
X1=dataset1.iloc[:,].values

pred1=classifier.predict(X1)



for i in range(0,10499):
    for j in range(0,10):
     if pred1[i][j]==pred1[i].max() :
      pred1[i][j]=1
     else:
      pred1[i][j]=0

y_pred1=onehotencoder.inverse_transform(pred1)


a = np.asarray(y_pred1)
np.savetxt("answer.csv", np.dstack((np.arange(1, a.size+1),a))[0],"%d,%d",header="ImageId,Label")