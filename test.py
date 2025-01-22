import pyswarm
from pyswarm import pso
from scipy.optimize import differential_evolution

def obj_func(x, sel):
    if sel:
        return x**2
    else:
        return -x**2

sel = True
result = differential_evolution(obj_func, [(-10, 10)], args=(sel,), popsize=10000, maxiter=100, mutation=0.8, recombination=0.7)

print(result)