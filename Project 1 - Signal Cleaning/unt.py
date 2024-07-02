# -*- coding: utf-8 -*-
"""
Created on Wed May  8 01:48:34 2019

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

plt.savefig('simplewithsigmoid_10nodes.png')


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

plt.savefig('simplewithsigmoid_50nodes.png')


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

plt.savefig('simplewithsigmoid_100nodes.png')



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
f.write('regularization_nunodes10_epochs_500_batchsize_5000\n')
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
