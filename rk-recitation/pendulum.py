from typing import Callable
from numpy import sin

# F builder
def get_f(alpha:float, beta:float):
    return lambda u,v: -alpha*v - beta*sin(u)

# step with rk4
def step(u:float, v:float, f:Callable, h:float = 1e-2):
    k1 = f(u      , v       )
    k2 = f(u+v*h/2, v+k1*h/2)
    k3 = f(u+v*h/2, v+k2*h/2)
    k4 = f(u+v*h  , v+k3*h  )
    k  = (k1+2*k2+2*k3+k4)/6

    return u + v*h, v + k*h

# Solve for some time t
def solve(T:float, u:float, v:float, f:Callable, h:float = 1e-2):
    U = [u]
    V = [v]
    i = 0
    
    while i < int(T/h):
        u_new, v_new = step(U[-1],V[-1], f, h)
        U.append(u_new)
        V.append(v_new)
        i += 1

    return U,V

# Solve pendulum
def solve_pendulum(alpha:float, beta:float, T:float, u:float, v:float, h:float = 1e-2):
    f = get_f(alpha,beta)
    return solve(T,u,v,f,h)