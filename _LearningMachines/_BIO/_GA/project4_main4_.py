import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib.pyplot as plt
from GA_TOOLS4 import *
#new_challenge = False
new_challenge = True
use_default = False
learning_mode = False
learning_mode = True
show_it = True
show_it = False
print(sys.argv)
if len(sys.argv) == 1:

    if use_default:
        l = 20         # length of string             ---default
        N = 30           # number of genes to test
        pm = 1/N        # probability of mutation
        pc = .6         # probability of cross over
        G = 10           # number of generations to run
        seed = 191  # random seed
    else:
        l = 20          # length of string
        N = 10           # number of genes to test
        #pm = 1/N        # probability of mutation
        pm = .1       # probability of mutation
        pc = .5         # probability of cross over
        G = 50           # number of generations to run
        seed = np.random.choice(1001, 1)[0]     # random seed
        #show_it = True
        show_it = show_it
        #seed = None     # random seed
else:
    l = int(sys.argv[1])          # length of string
    N = int(sys.argv[2])          # number of genes to test
    pm = float(sys.argv[3])       # probability of mutation
    pc = float(sys.argv[4])       # probability of cross over
    G = int(sys.argv[5])          # number of genrations to run
    seed = int(sys.argv[6])      # random seed
    if seed == -1:
        seed = None
    show_it = 1
    if len(sys.argv) > 7:
        show_it = int(sys.argv[7])   # show the plot?
    if len(sys.argv) > 8:
        learning_mode = int(sys.argv[8])   # show the plot?
        if learning_mode > 0:
            learning_mode = True
        else:
            learning_mode = False
    if len(sys.argv) > 9:
        new_challenge = int(sys.argv[9])   # show the plot?
        if new_challenge > 0:
            new_challenge = True
        else:
            new_challenge = False

if learning_mode:
    #l = 20
    #G = 50
    #pc = .5
    #pm = .1
    ol=0
print('seed: {}'.format(seed))
ga_mod = GA_Model_Tester(pop_size=N, gene_length=l, pm=pm, pc=pc, seed=seed, num_gen=G, learning_mode=learning_mode,
                         new_challenge=new_challenge)
#p(ga_mod.ga_pop)
p('the probability of mutation'.format(ga_mod.pm))
#p(ga_mod.ga_pop[0] and ga_mod.ga_pop[1])
#ga_mod.calculate_fitness()
#print('gene probe {}'.format(ga_mod.gene_prob))
#p('total fitness {}'.format(ga_mod.total_fitness))

#p('the running tally numbers\n{}'.format(ga_mod.gene_prob_tally))
print(ga_mod.run_generations())

pmstr =str(np.around(pm, 3)).strip('.')
pcstr =str(np.around(pc, 3)).strip('.')
filename = 'GA_Testing_N_{}_l_{}_G_{}_pm_{}_pc_{}.xlsx'.format(N, l, G, pmstr, pcstr)

#print_log(filename, ga_mod, seed,  l=l, N=N, G=G, pm=pm, pc=pc, save_img=True, show_it=False, dir_name='l{}'.format(l) + '\\')
if not learning_mode:
    print_log(filename, ga_mod, seed,  l=l, N=N, G=G, pm=pmstr, pc=pcstr, save_img=True, show_it=show_it,
              dir_name='l{}_EVOLVE_Challenge'.format(l) + '\\', new_challenge=new_challenge)
else:
    print_log(filename, ga_mod, seed,  l=l, N=N, G=G, pm=pmstr, pc=pcstr, save_img=True, show_it=show_it,
              dir_name='l{}_NURTURE_Challenge'.format(l) + '\\', new_challenge=new_challenge)


print('the final population:')
print(ga_mod.ga_pop)
print('The most probable incorrect bits are:')
print(ga_mod.prob_wrong_dict)


