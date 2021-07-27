import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib.pyplot as plt
import argparse
from _GA.GA_TOOLS4 import *
new_challenge = True
use_default = False
learning_mode = False
#learning_mode = True
show_it = True
show_it = False


parser = argparse.ArgumentParser(description="Run a set of GA tests for a given set up")
# population size
parser.add_argument('N',metavar='N',
                    type=int, help='the number of agents in the population',)


# Optional
# probability of mutation
parser.add_argument('--pm', metavar='pm', type=float,
                    help='The probability of mutation\n.' +
                         '-1 leads to 1/N, should be a float value (0,1)')
# probability of crossover
parser.add_argument('--pc', metavar='pc', type=float,
                    help='The probability of crossover\n.' +
                         '-1 leads to 1/N, should be a float value (0,1)')
# number of generations to test
parser.add_argument('--G',metavar="G",
                    type=int, help='The number of generations to run')

parser.add_argument('--l', type=int, help='the length of the chromosomes')
parser.add_argument('--seed', '--s', metavar='seed', dest='seed', type=int,
                    help='the random seed, if not provided None is used')
parser.add_argument('--verbose', '-v', dest='verbose',
                    action='store_false',
                    help='How much of the activity you want to see on stdout')
parser.add_argument('--showIt', '--show_it', '--showit', action='store_false',
                    dest='showit',
                    help='if you want to see the resulting training plot or not',)

parser.add_argument('--store', action='store_false',
                    dest='store',
                    help='if you want to save the results into figs and files',)

parser.add_argument('--learning_mode', action='store_false',
                    dest='--learning_mode',
                    help='turns learning mode on or off, default off',)
parser.add_argument('--file_name',
                    type=str,
                    help="the name of the log and image files stored")

parser.add_argument('--source_files1', type=str, )
parser.add_argument('--source_files2', type=str, )


def arg_handeler(parsed_args):
    N = parsed_args.N         # size of population
    if parsed_args.l:
        l = parsed_args.l         # length of the genes
    else:
        l = 10
    if parsed_args.G:
        G = parsed_args.G         # number of generations to run
    else:
        G = 50
    if parsed_args.pm:
        pm = parsed_args.pm       # mutation
    else:
        pm = 1/N
    if parsed_args.pc:
        pc = parsed_args.pc       # crossover
    else:
        pc = 1/(N*.5)

    if parsed_args.seed:
        seed = parsed_args.seed
    else:
        seed = None

    # these have default values so just use what is there
    verbose = parsed_args.verbose
    show_it = parsed_args.showit
    store = parsed_args.store

    if parsed_args.file_name:
       file_name = parsed_args.file_name
    else:
        file_name = "GA_Test_N-{}_L-{}_G-{}_pm-{:.2f}_pc-{:.2f}_".format(N,
                                                                l,
                                                                G,
                                                                pm,
                                                                pc, )

    if parsed_args.source_files1:
       agents_source1 = parsed_args.source_files1
    else:
        agents_source1= None
    if parsed_args.source_files2:
        agents_source2 = parsed_args.source_files2
    else:
        agents_source2 = None
    return l, G, N, pm, pc, agents_source1, agents_source2, seed


l, G, N, pm, pc, s1, s2, seed = arg_handeler(parser.parse_args())
print(("l: {}, G: {}, N: {}, pm: {}, pc: {}, s1: {}, s2: {}, seed: {}").format(l, G, N, pm, pc, s1, s2, seed))
quit()
"""
#print(sys.argv)
if len(sys.argv) == 1:

    if use_default:
        l = 20         # length of string             ---default
        N = 30           # number of genes to test
        pm = 1/N        # probability of mutation
        pc = .6         # probability of cross over
        G = 10           # number of generations to run
        seed = 191  # random seed
    else:
        l = 20           # length of string
        N = 20           # number of genes to test
        #pm = 1/N        # probability of mutation
        pm = .1          # probability of mutation
        pc = .55         # probability of cross over
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

if learning_mode:
    #l = 20
    #G = 50
    #pc = .5
    #pm = .1
    ol=0
"""

print('seed: {}'.format(seed))
ga_mod = GA_Model_Tester(pop_size=N, gene_length=l, pm=pm, pc=pc, seed=seed, num_gen=G, learning_mode=learning_mode)
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
    print_log(filename, ga_mod, seed,  l=l, N=N, G=G, pm=pmstr, pc=pcstr, save_img=True, show_it=show_it, dir_name='l{}_EVOLVE'.format(l) + '\\')
else:
    print_log(filename, ga_mod, seed,  l=l, N=N, G=G, pm=pmstr, pc=pcstr, save_img=True, show_it=show_it,
              dir_name='l{}_NURTURE'.format(l) + '\\',)


print('the final population:')
print(ga_mod.ga_pop)
print('The most probable incorrect bits are:')
print(ga_mod.prob_wrong_dict)


