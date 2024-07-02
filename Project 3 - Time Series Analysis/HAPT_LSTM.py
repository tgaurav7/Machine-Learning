# -*- coding: utf-8 -*-
"""
@author: gaur2

#the final result after testing gave 100% acccuracy while the test data gives 99.74% accuracy

Epoch 200/200
 - 1s - loss: 9.5430e-04 - acc: 1.0000
score
[0.02510403720871454, 0.9973924380704041]

"""

import numpy as np
import pandas as pd

import keras.models as km
import keras.layers as kl
import shelve
import keras.utils as ku
#load model_selection libraries for splitting test and training data
from sklearn.model_selection import train_test_split


#######################################################################


# Specify some data file names.
datafile0 = 'allrecipes.txt'
datafile = 'recipes.data'
shelvefile = 'recipes.metadata.shelve'
modelfile = 'recipes.model.h5'


#######################################################################
# Read data from file 'HAPT.data.csv'
data = pd.read_csv('HAPT.data.csv')

#splitting into a smaller dataset to be used here
traindata, testdata = train_test_split(data, test_size=0.01)

df = testdata
arx1 = []
arx2 = []
ary = []
arx3 = []
arx4 = []
arx5 = []
arx6 = []

#separating data to create the operlapping datasets
for i in range (1, 13):
    data_t = df[df.ID==i].values
#    print(data_t)
    arx1 = np.append(arx1, data_t[:,0])
    arx2 = np.append(arx2, data_t[:,1])
    ary = np.append(ary, data_t[:, 6])
    arx3 = np.append(arx3, data_t[:,2])
    arx4 = np.append(arx4, data_t[:,3])
    arx5 = np.append(arx5, data_t[:,4])
    arx6 = np.append(arx6, data_t[:,5])
    

sentence_length = 5
num_words = 6
x1_data = []
x2_data = []
x3_data = []
x4_data = []
x5_data = []
x6_data = []
y_data = []

#creating the overlapping data
for j in range(0, len(ary) - sentence_length):
    #to separate if result changes from one position to another
    if(ary[j] == ary[j+sentence_length-1]):
        x1 = arx1[j: (j + sentence_length)]
        x2 = arx2[j: (j + sentence_length)]
        x3 = arx3[j: (j + sentence_length)]
        x4 = arx4[j: (j + sentence_length)]
        x5 = arx5[j: (j + sentence_length)]
        x6 = arx6[j: (j + sentence_length)]
        y0 = ary[j: (j + sentence_length)]
        
        x1_data = np.append(x1_data, x1)
        x2_data = np.append(x2_data, x2)
        x3_data = np.append(x3_data, x3)
        x4_data = np.append(x4_data, x4)
        x5_data = np.append(x5_data, x5)
        x6_data = np.append(x6_data, x6)
        y_data = np.append(y_data, y0)


num_sentences = len(x1_data)

#converting data into one file for separating test and train data
x = np.zeros((num_sentences, sentence_length, num_words+1), dtype = np.float64)  
for l in range (0, num_sentences):
    for j in range (0, sentence_length):
      
        x[l,j,0] = x1_data[l]
        x[l,j,1] = x2_data[l]    
        x[l,j,2] = x3_data[l]
        x[l,j,3] = x3_data[l]
        x[l,j,4] = x4_data[l]
        x[l,j,5] = x5_data[l]
        x[l,j,6] = y_data[l]

np.save('data/' + datafile + '.x1.npy', x_data)
 
#print('Creating metadata shelve file.')
g = shelve.open('data/' + shelvefile)
g['sentence_length'] = sentence_length
g['num_words'] = num_words
g.close()
   
#Dividing the dataset in test and training sets
traindata, testdata = train_test_split(x, test_size=0.2)

#train data
x_train = traindata[:,:, 0:num_words]
y_train = traindata[:,:, num_words]
#converting results into categorical data
y_train2 = ku.to_categorical(y_train[:,0])

#test data
x_test = testdata[:,:, 0:num_words]
y_test = testdata[:,:, num_words]        

#converting results into categorical data
y_test2 = ku.to_categorical(y_test[:,0])
#print('Building network.')
model = km.Sequential()
model.add(kl.LSTM(256, input_shape = (sentence_length, num_words)))
model.add(kl.Dense(13, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

#model fitting with train 
fit = model.fit(x_train, y_train2, epochs = 200, batch_size = 128, verbose = 2)

# Save the model so that we can use it as a starting point.
model.save('data/' + modelfile)
#evaluating the model with test data
score = model.evaluate(x_test, y_test2, verbose = 2)
print('The score on final data is', score, 'with only 0.1% of the data')
