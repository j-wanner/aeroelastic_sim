import numpy as np

def rk4(f,x,t,dt,u):
    k1 = np.array(f(x, t, u))
    k2 = np.array(f(x + dt/2 * k1, t, u))
    k3 = np.array(f(x + dt/2 * k2, t, u))
    k4 = np.array(f(x + dt * k3, t, u))
    xf = x+dt/6*(k1 +2*k2 +2*k3 +k4)
    return xf

def euler_newton(f,x,t,dt,u):
    return x + dt*np.array(f(x,t,u))