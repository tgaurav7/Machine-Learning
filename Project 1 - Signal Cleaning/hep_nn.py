# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:21:42 2019

@author: gaur2
"""

# -*- coding: utf-8 -*-
"""
This code models a neural network with different parameters to model prediction of 'signal' 
or 'background data based on 27 f values for each case.

The data is read as dataframe 'data' and split into training and testing data(80:20) using the train_test_split library from model_selection. 

3 initial tests were performed to see the effect of number of nodes in the hidden layer(numnodes) namely for 10, 50 and 100 with sigmoid-activation, 
sgd-optimization and mean_squared_test-loss functions without any regularization, dropout and batchnormalization. 
Since, 10 nodes gave the best result with test data among the three and was fastest, the remaining tests were performed with 10 nodes. 
These include:
    1. With regularization (lam = 0.001)
    2. With dropouts (d_rate = 0.2)
    3. With batchnormalization
    4. In addition to batchnormalization, with 'tanh' as hidden layer activation function and 'softmax' for output layer
    5. In addition to above, the 'adam' optimization algorithm and 'categorical_crossentropy' loss function
    
The results for the tests are given as(score[2]):
    Running simple sigmoid function and Sgd optimization - numnodes 10
    0.815250
    numnodes 50
    0.814300
    numnodes 100
    0.814175
    Running with regularizationregularization_nunodes10_epochs_500_batchsize_5000
    0.813850
    Running with dropouts
    0.812200
    Running with updated activation functions
    0.814525
    Running with batchnormalization and updated activation functions
    0.816050
    Running with batchnormalization, updated activation, optimization and loss functions
    0.817125
    
The best results were with 10 numnodes was with including all the varied parameters i.e. the last model. 

However, the model with 50 nodes was also trained and tested. The results with 50 numnodes are given as:
    Running simple sigmoid function and Sgd optimization - numnodes 10
    0.819600
    numnodes 50
    0.819250
    numnodes 100
    0.818875
    Running with regularizationregularization_nunodes50_epochs_500_batchsize_5000
    0.819175
    Running with dropouts, drop rate = 40%
    0.819075
    Running with updated activation functions
    0.820625
    Running with batchnormalization and updated activation functions
    0.819575
    Running with batchnormalization, updated activation, optimization and loss functions
    0.818075
    
As seen, the results with 50 nodes with the time spent considerably higher are slighly better (3rd order of decimal only)
with the best of 0.819600 with the initial case i.e. sigmoid functions for hidden and output layers and no other parameters. 

@author: gaur2
"""

# Load the Pandas libraries with alias 'pd'
import pandas as pd
#load model_selection libraries for splitting test and training data
from sklearn.model_selection import train_test_split

#load plotting libraries
import matplotlib.pyplot as plt

#load keras 
import keras.utils as ku
import keras.models as km
import keras.layers as kl
import keras.regularizers as kr

#simple sigmoid activationfunction
def get_model_sigmoid(numnodes):
    #numnodes - int, number of nodes in hidden layer
    model = km.Sequential()
    model.add(kl.Dense(numnodes, input_dim = 27, activation = 'sigmoid', name = 'hidden'))
    model.add(kl.Dense(2, activation = 'sigmoid', name = 'output'))
    return model

#with regularization
def get_model_regularized(numnodes, lam=0.0):
    #lam - float, value of the regularizer parameter
    model = km.Sequential()
    model.add(kl.Dense(numnodes, input_dim = 27, activation = 'sigmoid', kernel_regularizer = kr.l2(lam), name = 'hidden'))
    model.add(kl.Dense(2, activation = 'sigmoid',kernel_regularizer = kr.l2(lam) , name = 'output'))
    return model

    
#with dropout
def get_model_dropout(numnodes, d_rate = 0.4):
    #d_rate - float, fraction of nodes to be dropped out by the dropout procedure
    model = km.Sequential()
    model.add(kl.Dense(numnodes, input_dim = 27, activation = 'sigmoid', name = 'hidden'))
    #add dropout to the hidden layer
    model.add(kl.Dropout(d_rate, name = 'dropout'))
    model.add(kl.Dense(2, activation = 'sigmoid', name = 'output'))
    return model

#with batchnormalizations
def get_model_batchnormalization(numnodes):
    model = km.Sequential()
    #add hidden layers
    model.add(kl.Dense(numnodes, input_dim = 27, activation = 'tanh', name = 'hidden'))
    #add batch normalization
    model.add(kl.BatchNormalization(name = 'batch_normalization'))
    #add output layer
    model.add(kl.Dense(2, activation = 'softmax', name = 'output'))
    return model
 

#updated actication functions - tanh and softmax at hidden and output layers
def get_model_tanh(numnodes):
    model = km.Sequential()
    #using tanh functoin for hidden layer
    model.add(kl.Dense(numnodes, input_dim = 27, activation = 'tanh', name = 'hidden'))
     #add batch normalization
    model.add(kl.BatchNormalization(name = 'batch_normalization'))
    #using tanh functoin for output layer
    model.add(kl.Dense(2, activation = 'softmax', name = 'output'))
    return model

# Read data from file 'HEPMASS_small.csv'
data = pd.read_csv("HEPMASS_small.csv")
#data.head()

#splitting into training and test datasets in ratio of 80:20
traindata, testdata = train_test_split(data, test_size=0.2)

#getting the training and testing data sets from dataframe
train = traindata.values
test = testdata.values

#splitting train and test x and y 
trainx = train[:, 1:28]
ytrain = train[:, 0]

trainy = ku.to_categorical(ytrain, 2)
 

testx = test[:, 1:28]
ytest = test[:, 0]

testy = ku.to_categorical(ytest, 2)
 


f= open("results_50.txt","w+")

print('Running simple sigmoid function and Sgd optimization')
f.write('Running simple sigmoid function and Sgd optimization')
#decalring model
model = get_model_sigmoid(10)
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('numnodes 10\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('simplewithsigmoid2_10nodes.png')


#decalring model
model = get_model_sigmoid(50)
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('numnodes 50\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('simplewithsigmoid2_50nodes.png')


#decalring model
model = get_model_sigmoid(100)
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('numnodes 100\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('simplewithsigmoid2_100nodes.png')



print('Running with regularization')
f.write('Running with regularization')
#decalring model
model = get_model_regularized(50, lam = 0.001)
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('regularization_nunodes50_epochs_500_batchsize_5000\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('regularized_50.png')




print('Running with dropouts')
#decalring model
model = get_model_dropout(50, d_rate = 0.4)


f.write('Running with dropouts, drop rate = 40%')
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('dropout_50.png')



print('Running with updated activation functions')
f.write('Running with updated activation functions')
#decalring model
model = get_model_tanh(50)
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('updatedactivationfunctions_50.png')




print('Running with batchnormalization and updated activation functions')
f.write('Running with batchnormalization and updated activation functions')
#decalring model
model = get_model_batchnormalization(50)
#compiling the model
model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = "mean_squared_error")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('batchnormalizedandupdatedactivationfunctions_50.png')


print('Running with batchnormalization, updated activation, optimization and loss functions')
f.write('Running with batchnormalization, updated activation, optimization and loss functions')
#decalring model
model = get_model_batchnormalization(50)
#compiling the model
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = "categorical_crossentropy")
#fitting the model
fit = model.fit(trainx, trainy, epochs = 500, batch_size = 500, verbose = 2)
score = model.evaluate(testx, testy)
print(score)
print('\n\n')
f.write('\n')
f.write("%f" %score[1])
f.write('\n')
plt.plot(fit.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig('mostupdated_50.png')

f.close();
