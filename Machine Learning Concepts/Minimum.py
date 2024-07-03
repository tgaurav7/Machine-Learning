# importing the libraries
import numpy as np
import scipy as sp

# creating list of the number of spacings, N for different runs
space = [10, 100, 1000, 10000, 100000, 1000000]

# finding y for each x and the printing the x corresponding to minimum y for each case

# running the loop for each spacings given 
for N in space:
    
    # creating x wiht N equally spaced points
    x=np.linspace(0., 10., N)
    
    # determining y corresponding to each x
    y=x-15*x/(1+x)
    
    # printing out the N and x for minimum y
    print(" The minimum with ",N, " divisions in x, for x = ", x[(y-(np.min(y)))<10e-16], "\n")
