# importing the libraries
import numpy as np

from scipy.integrate import odeint
import matplotlib.pyplot as plt

# defining the function to use method odeint imported from SciPy library 
def system(y,x):
    
    dydx=r*x*y;
    return dydx

# defining the function to calculate exact solution
def solreal(x):
    y=2*np.exp(-x**2)
    return (y)

# defining the function to calculate the solution with Runge Kutta method
def rungekutta(x, y, h, r, num):
    for i in range(0, num-1):
        k1=r*x[i]*y[i]
        k2=r*(x[i]+h/2)*(y[i]+h/2*k1)
        k3=r*(x[i]+h/2)*(y[i]+h/2*k2)
        k4=r*(x[i]+h)*(y[i]+h*k3)
        y[i+1]=y[i]+h/6*(k1+2*k2+2*k3+k4)
    return (y)


# perforing the calculations using the functions defined above
# coeff of x*y
r=-2

# creating num number of divisions in x 
num=1000
t=np.linspace(0, 3, num)

# calculating the solution with odeint method
sol=odeint(system, [0, 2], t)

# calculating the exact solution
yreal=solreal(t)

# runge kutta 4
# yn=yo+h/6(k1+2k2+2k3+k4)
# defining h
h=(3-0)/num
# defining initial condition for Runge Kutta method 
yr=np.zeros(num)
yr[0]=2

# calculating the solution using Runge Kutta method
yrk=rungekutta(t, yr, h, r, num)

# plotting the results
# subplotting to include both the plots in one pdf
plt.subplot(2,1,1)
plt.plot(t, sol[:,1], label='solution using ODEINT method', lw=3)
plt.plot(t, yrk, label='Runge Kutta method', lw=3)
plt.plot(t, yreal, label='Exact solution', lw=3)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
#plt.show(hold=False)

# calculating the errors with exact solution
err_odeint=sol[:,1] -yreal
err_rungekutta= yrk - yreal


#plotting errors
plt.subplot(2,1,2)
plt.plot(t, err_odeint, label='ODEINT method', lw=3)
plt.plot(t, err_rungekutta, label='Runge Kutta Method', lw=3)
plt.xlabel('x', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.legend()
#plt.show(hold=False)

plt.savefig('my_rk4.pdf')

# its observed that the error with Runge Kutta method is much higher while that
# with ODEINT method is very small in comparison
    
 
    