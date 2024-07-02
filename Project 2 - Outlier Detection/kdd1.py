# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:38:29 2019

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:22:18 2019

This program is for training an autoencoder for outlier detection. 

Effect of varying parameters:
    1. HIgher separation between normal and outlier data was seen at higher accuracy of autoencoding. 
    2. Epoch: Varying the number of eqpochs above 150 didn't affect the accuracy.
    3. Activation Functions: Varying the sigmoid activation function at decoder decreased accuracy. While changing relu to tanh function also affected the accuracy positively. 
    4. Encoder dimensions: Decreasing the encoder dimensions from 64 to 32 and 16 decreased accuracy.
    5. Regularization was included. 
    6. Loss function: mean squared value and binary_corssentropy loss functions were used. Using binary cross_entropy loss function resulted in low accuracy of autoencoding for validation data(0.80).
    7. Optimization algorithm: adam and adadelta optimization algorithm was used. Using adam optimization algorithm yielded better accuracy than with adadelta optimization algorithm

Threshold was defined as square root of means of mse for both the datasets. 

With only one layer of encoder and decoder
The mean MSE values for the validation data was too close as given. 
        Normal_Data
count  19450.000000
mean       0.000364
std        0.001252
min        0.000065
25%        0.000173
50%        0.000212
75%        0.000314
max        0.081299
       Outlier_Data
count    200.000000
mean       0.000780
std        0.002199
min        0.000180
25%        0.000266
50%        0.000266
75%        0.000343
max        0.029659

With two layers of encoder and decoders
Results with Test data set:
MSE distribution for outlier validation data (label = 1)
  Outlier_Data
count    178.000000
mean       0.001747
std        0.021552
min        0.000009
25%        0.000020
50%        0.000020
75%        0.000148
max        0.287647

MSE distribution for normal validation data (label = 0)
    Normal_Data
count  19472.000000
mean       0.000095
std        0.001114
min        0.000001
25%        0.000007
50%        0.000015
75%        0.000034
max        0.096018

Hence, a threshold value of threshold = 0.0005 was chosen. 

The confusion_matrix thus printed is:
    [[19151   299]
 [  176    24]]

Hence, the autoencoder predicts 323 cases of outlying data while only 200 were present. 

With three layers:
    
mean mse for normal validation data 0.00011564250628393803
This is mean mse for normal validation data 0.04487900964259461
        Normal_Data
count  19455.000000
mean       0.000116
std        0.001630
min        0.000003
25%        0.000022
50%        0.000033
75%        0.000065
max        0.149547
       Outlier_Data
count    195.000000
mean       0.044879
std        0.029570
min        0.000381
25%        0.002337
50%        0.068718
75%        0.068886
max        0.068886
[[19330   128]
 [  191     1]]

With three layers a much higher difference between the normal and outlier data was identified. Of the 192 data, 129 were identified.

 
@author: gaur2
"""
import numpy as np
# Load the Pandas libraries with alias 'pd'
import pandas as pd
#load model_selection libraries for splitting test and training data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


#load keras 
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers as kr

# Read data from file 'HEPMASS_small.csv'
data = pd.read_csv("Ass2.kddcup.csv")
data.head()
data.shape #size


#separate normal data from outlier data
#normal_data = data[data.label == 0]
#outlier_data = data[data.label == 1]


#splitting into training and test datasets in ratio of 80:20
traindata1, testdata = train_test_split(data, test_size=0.2)
#for validation dataset, separate the training again for 20% of total as validation dataset
traindata, valdata = train_test_split(traindata1, test_size=0.25)


#getting the training and testing data sets from dataframe
train1 = traindata.values
test1 = testdata.values
val1 = valdata.values

#splitting train and test x and y 
trainx = train1[:, 0:34]
valx = val1[:, 0:34]
testx = test1[:, 0:34]

#scaling the training data
scaler_train = MinMaxScaler()
print(scaler_train.fit(trainx))
train = scaler_train.transform(trainx)

scaler_val = MinMaxScaler()
scaler_val.fit(valx)
val = scaler_val.transform(valx)

#adding the autoencoder layers and parameters
dim = train.shape[1] #number of columns, 34
encoder_dim = 128
deep_encoder_dim = 64
deep_encoder2_dim = 32
#input layer
input_layer = Input(shape = (dim, ))
#add a Dense layer with L1 activity regularizer

#the encoder
encoder = Dense(encoder_dim, activation = "tanh", activity_regularizer=kr.l1(1e-5))(input_layer)
encoder = Dense(deep_encoder_dim, activation = 'tanh')(encoder)
encoder = Dense(deep_encoder2_dim, activation = 'tanh')(encoder)

#the decoder
decoder = Dense(deep_encoder2_dim, activation = 'tanh')(encoder)
decoder = Dense(deep_encoder_dim, activation = 'tanh')(decoder)
decoder = Dense(dim, activation = 'sigmoid')(encoder)


autoencoder  = Model(inputs = input_layer, outputs = decoder)

#model autoencoder to use mean_squared_error loss and adam optimizer
autoencoder.compile(metrics = ['accuracy'], loss='binary_crossentropy', optimizer = 'adam')

autoencoder.fit(train, train, epochs = 120, batch_size = 64, shuffle = True, verbose =2, validation_data = (val, val))
#save models
autoencoder.save('autoencoder.h5')

# mean mse for validation data without separation = 0.00015266318322697237
# separating the validation data between normal and outlier data for setting the threshold
normal_data = valdata[valdata.label ==0]
outlier_data = valdata[valdata.label==1]

normal_val = normal_data.values
outlier_val = outlier_data.values

#scaling the validation data
scaler_valnorm = MinMaxScaler()
scaler_valout = MinMaxScaler()

scaler_valnorm.fit(normal_val)
scaler_valout.fit(outlier_val)

#performing scaling
val_normal_scaled = scaler_valnorm.transform(normal_val)
val_outlier_scaled = scaler_valout.transform(outlier_val)

#removing the label
valnormdata = val_normal_scaled[:, 0:34]
valoutdata = val_outlier_scaled[:, 0:34]

#using the trained autoencoder to predict for normal and outlier validation data
normdata_predictions = autoencoder.predict(valnormdata)
outdata_predictions = autoencoder.predict(valoutdata)

#calculating MSE
mse_norm = np.mean(np.power(valnormdata - normdata_predictions, 2), axis=1)
mse_out = np.mean(np.power(valoutdata - outdata_predictions, 2), axis=1)
mean_error_norm = np.mean(mse_norm)
mean_error_out = np.mean(mse_out)
print('mean mse for normal validation data', mean_error_norm)
print('This is mean mse for normal validation data', mean_error_out)

#converting errors into dataframes for descriptive analysis
error_normal_df = pd.DataFrame({'Normal_Data':mse_norm, })
error_outlier_df = pd.DataFrame({'Outlier_Data':mse_out, })

#printing descriptive analysis of mse for both datasets
print(error_normal_df.describe())
print(error_outlier_df.describe())

#setting up the threshold point 
threshold =np.sqrt(mean_error_norm*mean_error_out)

#separating the test dataset
test = testdata.values

testx = test[:, 0:34]
testy = test[:, 34]

#scaling test data
scaler_test = MinMaxScaler()
scaler_test.fit(testx)

test_x = scaler_test.transform(testx)

test_predictions = autoencoder.predict(test_x)
test_mse = np.mean(np.power(test_x-test_predictions, 2), axis=1)
test_error_df = pd.DataFrame({'Test_Error':test_mse, 'True_Class':testy})

#printing confusion matrix
result = [1 if e > threshold else 0 for e in test_error_df.Test_Error.values]
print(confusion_matrix(test_error_df.True_Class, result))

