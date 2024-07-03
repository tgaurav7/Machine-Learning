# importing the libraries
import numpy as np
import scipy as sp

# defining the function given with x as the only variable
# since x is the only variable I have used the scalar minimization in optimize library in SciPy
# the results differed when using Brent and Bounded methods in the 8th decimal point

def f(x):
    return (x-15*x/(1+x))

# finding the x for minimum value of f using the Brent method 
root=sp.optimize.minimize_scalar(f)
print("The minimum calculated using the Brent method for Scalar Minimization is ",root.x, "\n")


def f(x):
    return (x-15*x/(1+x))

root=sp.optimize.minimize_scalar(f, bounds=(0.0, 10.0), method='bounded')
print("The minimum calculated using the Bounded method for Scalar Minimization is "root.x, "\n")

# on comparing: That calculated using the SciPy library with those calculated by finding done earlier
# the results in earlier part differs to the actual value 2.87298334621 after the third decimal digit
# that calculated using SciPy library is same till 8th digit, hence much higher in accuracy 