from numpy import array, linspace, savetxt
from multiprocessing import Pool, cpu_count
from time import perf_counter
from os import getpid

def get_f(a,b,c):
    def f(x):
        return array([a*(x[1] - x[0]), x[0]*(b-x[2]) - x[1], x[0]*x[1] - c*x[2]])
    return f

def step(x, f, h:float = 1e-2):
    k1 = f(x)
    k2 = f(x + h*k1/2)
    k3 = f(x + h*k2/2)
    k4 = f(x + h*k3)

    return x + (k1 + 2*k2 + 2*k3 + k4)*h/6

def worker(crange):    
    print(f"[{perf_counter():.3f}]--> PID=[{getpid()}] start at c={crange[0]:4.2f}")
    steps   = 10000
    h       = 1e-2
    a, b    = 10., 28.
    points  = []
    
    for c in crange:
        f = get_f(a,b,c)
        x0 = array([1.,1.,1.])

        for _ in range(steps-1):
            x = step(x0, f, h)
            if x[0]*x0[0] < 0:
                points.append(array([c, x0[2] + (x[2] - x0[2])/(x[0] - x0[0])*(-x0[0])]))
            x0 = x
    print(f"[{perf_counter():.3f}]--> PID=[{getpid()}] ended at c={crange[-1]:4.2f}")

    return points


if __name__=="__main__":
    c_min   = 0.35
    c_max   = 0.65
    N       = 100
    cpus    = cpu_count()
    delta   = (c_max - c_min) / cpus
    cranges = [linspace(c_min + i*delta,c_min + (i+1) * delta, N) for i in range(cpus)] 
    ints    = []

    initial_time = perf_counter()
    with Pool(cpus) as pool:
        for i in pool.map(worker, cranges): ints += i
    ints = array(ints)
    print(f"Execition time: {perf_counter() - initial_time:.6f} seconds")

    savetxt('intersections.csv', ints.T, delimiter=',')
