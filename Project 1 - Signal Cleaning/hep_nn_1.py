import numpy as np
# Load the Pandas libraries with alias 'pd'
import pandas as pd

# Read data from file 'HEPMASS_small.csv'

data = pd.read_csv("HEPMASS_small.csv")

data.head()

#splitting into training and test datasets in ratio of 80:20
split = pd.DataFrame(np.random.randn(len(data), 2))
splitper = np.random.rand(len(split)) < 0.8
train = split[splitper]
test = split[~splitper]


