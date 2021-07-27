import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


ls = [5,10,20, 30, 40]
ls = [5,]
def fit_func1(num, gene_length):
    return (num / (2 ** gene_length)) ** 10


def fit_func2(num, gene_length):
    return (1 - (num / (2 ** gene_length))) ** 10

def make_arrays(r, ls):
    a1all = list()
    a2all = list()
    strlen_a = list()
    xarray = list()
    lim = 33
    for l in ls:
        a1 = [fit_func1(x+1, l) for x in range(0, lim)]
        a2 = [fit_func2(x+1, l) for x in range(0, lim)]
        xarray.append([x+1 for x in range(0, lim)])
        a1all.append(a1)
        a2all.append(a2)
        strlen_a.append(l)
    return a1all, a2all, strlen_a, xarray
lgnd = ['func1', 'func2']

a1all, a2all, strl, xa = make_arrays(200, ls)
plt.figure(1)
plt.title('Objective function 1 for the GA to maximize')
for a1,x in zip(a1all, xa):
    plt.plot(x, a1)
plt.xlabel('numerical value')
plt.ylabel('output')
plt.legend(strl)


plt.figure(2)
plt.title('Objective function 2 for the GA to maximize')
for a2,x in zip(a2all, xa):
    plt.plot(x, a2)
plt.xlabel('numerical value')
plt.ylabel('output')
plt.legend(strl)
plt.show()