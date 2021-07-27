"""
    Purpose: originally created as a proof of concept for a Bioinspired computing course
             Now stands as a test bed for the tools generated during the coursework to go further
"""


import os
import sys
import numpy as np


# the length of the genetic string/chromosome
ls = [30,40]
#ls = [40]

# number of ?
Ns = [10, 30,50,100,]

# probability of mutation
pms = [-1, .25, .5, .05, ]

# probability of crossover
pcs = [.30, .50, .99, ]

# number of generations
Gs = [10,50,100,]

# ?
rnge = 5
for l in ls:
    for N in Ns:
        for pm in pms:
            for pc in pcs:
                for G in Gs:
                    # negative one means use the default pm value
                    if pm == -1:
                        pm = 1/N
                        # for the given range generate
                        # the given number of tests
                    for i in range(1, rnge):
                        seed = np.random.choice(list(range(1001)), 1)[0]
                        #                                        r                    l N  pm pc G  sd show_plot
                        os.system('python Iterative_Test_Runs.py {} project4_main4.py {} {} {} {} {} {} {}'.format(1,l, N, pm, pc,G, seed, 0))
                    seed = np.random.choice(list(range(1000)), 1)[0]
                    os.system('python Iterative_Test_Runs.py {} project4_main4.py {} {} {} {} {} {} {}'.format(1,l, N, pm, pc,G, seed, 0))
