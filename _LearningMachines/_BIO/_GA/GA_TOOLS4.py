"""
    Purpose: Collection of tools for performing GA optimization tasks
    Created By: Gerald Jones
    Created: 7/19/21
    Last Edit: 7/1921
    Dependent files:
    Required files: _utils
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ._utils import sort_dict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Genetic Algorithm Optimization tool. Class only performs the GA
# related functions such as:
#                     * creating the initial population of chromosomes
#                     * using a returned score mapping of chromosomes to
#                       objective function scores to perform
#                       "survival of the fittest" selection of breeding pairs
#                       as well as use the breeding pairs to generate the next generation of
#                       chromosomes
#                     * perform any necessary crossover, or mutation on the new generation
#                     * keep track of each generation's performance
# The GAOPTM class takes as an input arguments:
#                     * population_size: the number of chromosomes in a population
#                     * strlen: the lenght of the chromosomes
#                     * solverfunc: a
class GAOPTM:
    def __init__(self, population_size, strlen, solverfunc, pm=.06, pc=None, verbose=False, generations=100,
                 init_func=None, oneprob=None, Nopen=None, show_first_gen=True,
                 **kwargs):
        self.population_size=population_size
        self.pm=pm
        self.pc=pc
        self.strlen=strlen
        self.verbose=verbose
        self.generations=generations
        self.solver=solverfunc
        if init_func is not None:
            self.chromosomes=init_func(population_size, strlen, **kwargs)
        else:
            self.chromosomes=self.init_func(population_size, strlen, oneprob, Nopen)

        if show_first_gen:
            print("\n------------------The First Epoch's Generation---------------------")
            minv = self.strlen*2
            maxv = -1
            for chro in self.chromosomes:
                print('\n',list(chro).count(1))
                maxv = max(list(chro).count(1), maxv)
                minv = min(list(chro).count(1), minv)
                print(list(chro).count(0), '\n')
            print("-------The First Epoch's Generation-Max: {}, Min: {}--------------\n".format(maxv, minv))

    def init_func(self, population_size, strlen, oneprob=None,Nopen=None):
        if oneprob is not None:
            return self.ginit_func(population_size, strlen, oneprob)
        else:
            return self.sinit_generation(population_size, strlen, Nopen)
    def ginit_func(self, population_size, strlen, oneprob):
        """
            Initializes each chromsome with a given probability of ones.
        :param population_size: number of chromosomes to add to the list
        :param strlen:          how long each chromosome will be
        :param oneprob:         probability of ones, allows for varied number of ones,
                                with a soft range/prog of seeing them
        :return: a list of numpy arrays where each array is a chromosom/solution
        """
        solutions = list()
        # print("oneprob ", oneprob)
        for i in range(population_size):
            solutions.append(np.random.default_rng().choice([0, 1], strlen, p=[1 - oneprob, oneprob]))
        return solutions
    def sinit_generation(self, population_size, strlen, Nopen):
        """
                Generates a set of chromosomes with a set number of ones to start
        :param population_size:
        :param strlen:
        :param Nopen:
        :return:
        """
        print("\n\t\t\t\tsiniting.......\n\n\n")
        population_init = list()
        if Nopen is None:
            Nopen=int(strlen/2)
        for p in range(population_size):
            # get the desired length of chromosome
            chromo = np.zeros(strlen)
            # intialize the given number of DG's as open by getting some random
            # indices and setting them to 1
            opens = np.random.default_rng().choice(list(range(strlen)), Nopen, replace=False)
            chromo[opens] = 1
            population_init.append(chromo)
        return population_init

    def generate_SelectionProbs(self, scores_dict, mode="MIN"):
        print("Mode: ", mode)
        if mode.upper() == "MIN":
            return self.generate_SelectionProbsMIN(scores_dict, )
        elif mode.upper() == "MIN2":
            return self.generate_SelectionProbsMIN2(scores_dict)
        else:
            return self.generate_SelectionProbsMAX(scores_dict)
    def generate_SelectionProbsMAX(self, scores_dict, ):
        # if self.verbose:
        #     print("Total score sum: {}".format(sumZ))
        prob_dict = {}

        # if OPTIMZ.upper() == 'MIN':
        # get the sum of the scores
        sumZ = np.sum(list(scores_dict.values()))
        #p(scores_dict)
        #print(sumZ)
        for k in scores_dict:
            prob_dict[k] = scores_dict[k]/sumZ
            #print(prob_dict[k])


        # now sort this based on the "cost(i)/total_cost"
        # as a representation of the probability of selection for breeding
        # But for this task the "best score" is the one with the lowest cost
        # so another step needs to be taken

        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1]))


        psum = np.sum(list(prob_dict.values()))
        if psum != 1 and self.verbose:

            # adjust the highest value to add what ever the difference between the sum and 1 is
            #         if psum < 1:
            #             diff = 1 - psum
            #         else:
            #             diff = 1 - psum
            diff = 1 - psum
            lastIDXKEY = list(prob_dict.keys())[-1]
            prob_dict[lastIDXKEY] = prob_dict[lastIDXKEY] + diff
        psum = np.sum(list(prob_dict.values()))


        # get the reversed version of the probabilities. This
        # will be used to reassign the probabilites so that the one that had
        # the highest score, thus the highest probability will be reassigned
        # to have the lowest and so on
        # rprobs = sorted(list(prob_dict.values()), reverse=True)
        # rprob_dict = {}
        #
        # # reassign the probabiliest so that the highest cost chromosome will
        # # have the least probability of selection
        # for k, p in zip(prob_dict, rprobs):
        #     rprob_dict[k] = p
        #     if self.verbose:
        #         print('true: {}: {}'.format(k, prob_dict[k]))
        #         print("reverse: {}: {}".format(k, rprob_dict[k]))
        #         print("----")

        return prob_dict
    def generate_SelectionProbsMIN2(self, scores_dict,):
        # if self.verbose:
        #     print("Total score sum: {}".format(sumZ))
        prob_dict = {}
        # if OPTIMZ.upper() == 'MIN':
        # get the sum of the scores
        for k in scores_dict:
            scores_dict[k] = 1/scores_dict[k]

        sumZ = np.sum(list(scores_dict.values()))
        # now sort this based on the "cost(i)/total_cost"
        # as a representation of the probability of selection for breeding
        # But for this task the "best score" is the one with the lowest cost
        # so another step needs to be taken
        for k in scores_dict:
            prob_dict[k] = scores_dict[k]/sumZ
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1]))

        psum = np.sum(list(prob_dict.values()))
        if psum != 1 and self.verbose:
            # adjust the highest value to add what ever the difference between the sum and 1 is
            #         if psum < 1:
            #             diff = 1 - psum
            #         else:
            #             diff = 1 - psum
            diff = 1 - psum
            lastIDXKEY = list(prob_dict.keys())[-1]
            prob_dict[lastIDXKEY] = prob_dict[lastIDXKEY] + diff
        psum = np.sum(list(prob_dict.values()))
        # print(psum)
        # print(prob_dict)
        # quit()
        # get the reversed version of the probabilities. This
        # will be used to reassign the probabilites so that the one that had
        # the highest score, thus the highest probability will be reassigned
        # to have the lowest and so on
        # rprobs = sorted(list(prob_dict.values()), reverse=True)
        # rprob_dict = {}
        #
        # # reassign the probabiliest so that the highest cost chromosome will
        # # have the least probability of selection
        # for k, p in zip(prob_dict, rprobs):
        #     rprob_dict[k] = p
        #     if self.verbose:
        #         print('true: {}: {}'.format(k, prob_dict[k]))
        #         print("reverse: {}: {}".format(k, rprob_dict[k]))
        #         print("----")

        return prob_dict
    def generate_SelectionProbsMIN(self, scores_dict):
        sumZ = np.sum(list(scores_dict.values()))

        prob_dict = {}
        #if OPTIMZ.upper() == 'MIN':
        # get the sum of the scores
        for k in scores_dict:
            prob_dict[k] = scores_dict[k] / sumZ

        # now sort this based on the "cost(i)/total_cost"
        # as a representation of the probability of selection for breeding
        # But for this task the "best score" is the one with the lowest cost
        # so another step needs to be taken

        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1]))

        psum = np.sum(list(prob_dict.values()))
        if psum != 1 and self.verbose:
            # adjust the highest value to add what ever the difference between the sum and 1 is
            #         if psum < 1:
            #             diff = 1 - psum
            #         else:
            #             diff = 1 - psum
            diff = 1 - psum
            lastIDXKEY = list(prob_dict.keys())[-1]
            prob_dict[lastIDXKEY] = prob_dict[lastIDXKEY] + diff
        psum = np.sum(list(prob_dict.values()))

        # get the reversed version of the probabilities. This
        # will be used to reassign the probabilites so that the one that had
        # the highest score, thus the highest probability will be reassigned
        # to have the lowest and so on
        rprobs = sorted(list(prob_dict.values()), reverse=True)
        rprob_dict = {}

        # reassign the probabiliest so that the highest cost chromosome will
        # have the least probability of selection
        for k, p in zip(prob_dict, rprobs):
            rprob_dict[k] = p

        return rprob_dict

    def check_scores(self, current_best, perspective_best, mode="MIN"):
        if mode.upper() in ["MIN", "MIN2"]:
            return current_best > perspective_best
        else:
            return current_best < perspective_best

    def run_generations(self,pm=None, pc=.08, gthresh=1, max_pop=500,
                        probability_gen=None, pair_selector=None,child_gen=None,
                        generations=100, window=3, grate=.9,
                        mode="MIN", verbose=True, **kwargs):
        if "adaptive" in kwargs:
            adaptive=kwargs["adaptive"]
        else:
            adaptive=False
        avg_score = list()
        best_scores = list()
        solutions = list()
        best_score = np.inf
        best_solution = None
        if mode.upper() not in  ["MIN", "MIN2"]:
            eval=max
        else:
            eval=min
        if probability_gen is None:
            probability_gen = self.generate_SelectionProbs
        if pair_selector is None:
            pair_selector = self.survivalOfTheFittest
        if child_gen is None:
            child_gen = self.generateNextGeneration

        print("The game of life begins....\n")
        rtt = 2
        for i in range(generations):
            # Score this set of chromosomes
            scores_dict = self.solver.score_population(self.chromosomes)
            # store this generations max/min score
            bestThis =eval( list(scores_dict.values()))
            # print("\n\n\n")
            # print("\t\t\t\t\teval ", eval)
            # print("the bes? ", bestThis)
            # print("\n\n\n")

            if adaptive:
                # get the difference between the last timewindow sampls and the last one
                # if they are below the work hard threshold start to grow population by
                # given growth rate
                if len(best_scores) > window+1:
                    lastavg = np.mean(best_scores[-(window+1):-1])
                    lastone = best_scores[-1]
                    diff = np.around(lastone - lastavg, 3)
                    # if diff >= 0:
                    #     self.pc += (lastavg/lastone)*.1
                    #     self.pc = min(.1, self.pc)
                    #     pc = self.pc
                    #     print('mutation rate is now: {}'.format(pc))
                else:
                    diff = 0
                    lastavg=None
                    lastone=None
                print("diff ", diff)
                print("last {}, this {}".format(lastavg, lastone))
                # if gthresh is not None and abs(diff) > gthresh:
                if gthresh is not None and abs(diff) > gthresh:
                    self.population_size += int(self.population_size*grate/i  + diff*.000001)
                    if self.population_size < 10:
                        self.population_size = 10
                    self.population_size = min(self.population_size, max_pop)
                    print("increase of population size to: ", self.population_size)
                    self.pc +=  -(diff*.0000001)/i
                    self.pc = abs(self.pc)
                    self.pc = min(.25, abs(self.pc))

                    self.pm += (diff * .0000001)/i
                    self.pm = abs(self.pm)
                    self.pm = min(.1, abs(self.pm))

                    pc = self.pc
                    pm = self.pm
            # get the index of the best score this run to compare it to the
            # current best overall
            bestSCR = eval( list(scores_dict.values()))
            best_scores.append(bestSCR)
            bestIdx = list(scores_dict.values()).index(bestSCR)
            bestSolution = list(scores_dict.keys())[bestIdx]
            avg_score.append(np.mean(list(scores_dict.values())))
            solutions.append(self.chromosomes[bestSolution])
            if best_score > best_scores[-1]:
                best_score = best_scores[-1]
                best_solution = solutions[-1]
                if verbose:
                    print("\t\t\t-----------NEW BEST SCORE!!: {:.6f}, GEN: {}".format(best_score, i))

            if self.solver.threshold_check is not None:
                if self.solver.threshold_check(best_score):
                    return best_score, best_solution
            # if the threshold has not been breached
            if i%rtt == 0 and verbose:
                print("\nGeneration: {}".format(i))
                print('best_score this run: {}'.format(best_scores[-1]))
                #print("scores dict: {}\n".format(scores_dict))
                print("best score so far: {}".format(best_score))
                print("best solution so far: {}".format(best_solution))
                print("pc: {}, pm: {}, pop: {}\n".format(pc, pm, self.population_size))

            # prob_dicc = self.generate_SelectionProbs(scores_dict, mode=mode)
            # breeding_pairs = self.survivalOfTheFittest(pop=list(prob_dicc.keys()), probs=list(prob_dicc.values()))
            # self.chromosomes = self.generateNextGeneration(breeding_pairs, self.chromosomes, crossoverrate=pc, mutation_rate=pm)
            prob_dicc = probability_gen(scores_dict, mode=mode)
            #breeding_pairs = pair_selector(list(prob_dicc.keys()), list(prob_dicc.values()))
            breeding_pairs = pair_selector(list(prob_dicc.keys()), list(prob_dicc.values()))
            self.chromosomes = child_gen(breeding_pairs, self.chromosomes, crossoverrate=pc, mutation_rate=pm)

        return best_scores, avg_score, best_score, best_solution

    # make sure we do not breed a solution with it self
    def checkForRepeat(self, current_pairs, pair):
        for current_pair in current_pairs:
            if pair[0] in current_pair and pair[1] in current_pair:
                # print("t\t\t\t\ there is a repeat: {}, {}".format(pair, current_pair))
                # if there is a repeat return true
                return True
        return False

    def survivalOfTheFittest(self, pop, probs):
        # get the initial pair
        new_pair = list(list(np.random.default_rng().choice(pop, size=2, p=probs, replace=False)))
        breeding_pairs = list()
        breeding_pairs.append(new_pair)

        # get limit on how many unique combinations to try
        N=len(probs)
        lims = 0
        while N > 0:
            N -= 1
            lims += N

        # breeding_pairs = list()
        # print(breeding_pairs)
        used_up = [list(breeding_pairs[-1])]
        # the last size-1 pairs
        # create pairs until you have enough to replace last generation
        for i in range(  min(self.population_size, lims) - 1):
            # get two unique pairs based on the given probability list
            while new_pair in breeding_pairs and self.checkForRepeat(used_up, new_pair):
                #        while new_pair in breeding_pairs:
                new_pair = list(np.random.default_rng().choice(pop, size=2, p=probs, replace=False))
            #             print("new possible: {}".format(new_pair))
            #             print("Couples: ", breeding_pairs)
            # print("accepted new pair: {}".format(new_pair))
            used_up.append(new_pair)
            breeding_pairs.append(new_pair)
        #         print('up: ', used_up)
        #         print("Couples: ", breeding_pairs)
        return breeding_pairs

    # breeding pair selector pop=list(prob_dicc.keys()), probs=list(prob_dicc.values())
    def pair_pop(self, populationidx, gene_prob_tally):
        pairs = list()
        popsize = len(populationidx)
        while len(pairs) < len(populationidx):
            cycle_cnt = 0
            parent1 = self.get_upperRNG(populationidx, gene_prob_tally)
            parent2 = self.get_upperRNG(populationidx, gene_prob_tally)
            while parent1 != parent1:
                parent1 = self.get_upperRNG(populationidx, gene_prob_tally)
                parent2 = self.get_upperRNG(populationidx, gene_prob_tally)
                if cycle_cnt > len(populationidx) * 2:
                    # if self.convergence_check(ga_pop):
                    #     print('--------------------convergence ---------------------')
                    if np.random.default_rng().choice([True, False], 1)[0]:
                        parent2 = (parent2 + np.random.choice(range(popsize), 1)[0]) % popsize
                    else:
                        parent1 = (parent1 + np.random.choice(range(popsize), 1)[0]) % popsize
                else:
                    parent1 = self.get_upperRNG(populationidx, gene_prob_tally)
                    parent2 = self.get_upperRNG(populationidx, gene_prob_tally)
                    cycle_cnt += 1
            pairs.append((parent1, parent2))
        return pairs

    # for of a probability generator
    def get_prob_tally(self, fitness_arrayd, **kwargs):
        gene_prob = self.calculate_gene_prob(list(fitness_arrayd.values()))
        retd = {k:v for k, v in zip(fitness_arrayd, gene_prob)}
        # retd = {k:v for k, v in zip(fitness_arrayd,self.calculate_gene_prob_tally(gene_prob))}
        retd = dict(sorted(retd.items(), key=lambda x:x[1]))
        # rpobs
        lilend = sorted(list(retd.values()), reverse=True)
        retd = {k:v for k, v in zip(list(retd.keys()),self.calculate_gene_prob_tally(lilend))}

        # if "mode" in kwargs:
        #     if kwargs["mode"].upper() in ["MIN", "MIN1"]:
        #         # reverset the probs
        #         sgene_probT = dict(sorted(retd.items(), key=lambda x: x[1]))
        #         # get the sorted tallys and reverse them

        return retd
    def calculate_gene_prob(self, fitness_array, ):
        gene_prob = (np.array(fitness_array) / np.array(fitness_array).sum())
        return gene_prob
    def calculate_gene_prob_tally(self, gene_prob):
        gene_prob_tally = np.cumsum(gene_prob)
        return gene_prob_tally


    def get_upperRNG(self, populationidx, gene_prob_tally, **kwargs):
        rng_num = np.random.uniform(0, 1, 1)
        idx = -1
        parent = -np.inf
        #print('tally?')
        #print(self.gene_prob_tally)
        # print('gene pop size {}'.format(self.pop))
        while parent <= rng_num and idx < len(populationidx):
            idx += 1
            parent = gene_prob_tally[idx]
        if idx == len(populationidx):
            idx -= 1
        return populationidx[idx]

    def generateNextGeneration(self, breeding_pairs,
                               chromosomes, crossoverrate=.16,
                               mutation_rate=None, fiffifsplt=False):
        if chromosomes is None:
            chromosomes = self.chromosomes

        # go through pair list generating the kids from the pairings
        kid_array = list()
        cnt = 0
        # if not given set the rate of mutation
        if mutation_rate is None:
            mutation_rate = np.around(1 / len(chromosomes))

        while len(kid_array) < self.population_size:
            # decide if crossover will occur, if so get the crossover point randomly
            if np.random.choice([False, True], 1, p=[1 - crossoverrate, crossoverrate])[0]:
                # if we need to crossover get a crossover point
                if fiffifsplt:
                    cp=int(len(chromosomes)/2)
                else:
                    cp = np.random.choice(list(range(1, len(chromosomes)-1)), 1)[0]
            else:
                cp = None
            # each iteration the kid array is passed to the procreate function along with the
            # "breeding pair" reprsented by indices into the chromosom array.
            #  p1, p2, ga_pop, cp, pm,
            kida, kidb = self.procreate(breeding_pairs[cnt][0], breeding_pairs[cnt][1],
                                       chromosomes, cp, mutation_rate,)
            kid_array.append(kida)
            if len(kid_array) < len(chromosomes):
                kid_array.append(kidb)
            cnt += 1
        return kid_array

    def mutate(self, genome, pm):
        #print("mutation: {}".format(pm))
        #oneIdx, zeIdx = list(), list()
        seek = 0
        # get the index of the zeros and 1's
        # for g in genome:
        #     #print("G: {}".format(g))
        #     #quit()
        #     if g == 1:
        #         oneIdx.append(cnt)
        #     else:
        #         zeIdx.append(cnt)
        #     cnt += 1

        # decide if mutation occurs, is so flip two random bits
        # fliping two to maintain number of on bits
        # if true it is all zeros so just flip the bits with the given prob
        for idx in range(self.strlen):
            # go through genome flipging(mutating) bits with given prob
            if np.random.choice([False, True], 1, p=[1 - pm, pm])[0]:
                # idx = np.random.default_rng().choice(list(range(len(genome))), 1)[0]
                if genome[idx] == 1:
                    genome[idx] = 0
                else:
                    genome[idx] =1
        # for idx in range(self.strlen):
        #     # attempst to do a balanced flip of a bit at  at time
        #     if np.random.choice([False, True], 1, p=[1 - pm, pm])[0]:
        #         set_null = np.random.default_rng().choice(oneIdx, 1)[0]
        #         set_one = np.random.default_rng().choice(zeIdx, 1)[0]
        #         genome[int(set_null)] = 0
        #         genome[int(set_one)] = 1
        return genome
    # This one sucks, needs to be altered or removed
    def mutate2(self, genome, pm):
        for g in range(len(genome)):
            if np.random.default_rng().choice([True, False], 1, p=[pm, 1-pm])[0]:
                if genome[g] == 1:
                    genome[g] = 0
                else:
                    genome[g] = 1
        return genome
    def procreate(self, p1, p2, ga_pop, cp, pm, kid_array=None):
        """
                Performs the crossover function
        :param p1: index of parent 1
        :param p2: index of parent 2
        :param cp: crossover point
        :return:
        """
        # print("p1: ", p1)
        # print("p2: ", p2)
        # print("ga: ", ga_pop)
        # if cp is not None then a crossover at the crossover point
        # cp needs to occur
        # print("p1: {}, p2: {}, cp: {}".format(p1, p2, cp))
        # print("ga_pop: {}".format(ga_pop[p1][:cp]))
        # print("ga_pop: {}".format(ga_pop[p1]))
        # print("ga_pop2: {}".format(ga_pop[p2][cp:]))
        # print("ga_pop2: {}".format(ga_pop[p2]))
        # print(type(ga_pop))

        if cp is not None:
            # print('kids are created at the cp point of {}'.format(cp))
            # print('the parents are indices {} and {} shown below'.format(p1, p2))
            # print('parent 1: {}'.format(self.ga_pop[p1]))
            # print('parent 2: {}'.format(self.ga_pop[p2]))
            # replace parent 1, with kida
            a1 =ga_pop[p1][:cp]
            a2 = ga_pop[p2][cp:]
            kida = list(a1) + list(a2)
            kidb = list(ga_pop[p2][:cp]) + list(ga_pop[p1][cp:])
            # print('\n--------------------------------------------')
            # print('kida before mutation\n{}'.format(kida))
            # print('kidb before mutation\n{}'.format(kidb))
            # print('--------------------------------------------')
            # Make sure there are not all zeros kids
            while sum(kida) == 0:
                kida = self.mutate(kida, pm)
            while sum(kidb) == 0:
                kidb = self.mutate(kidb, pm)
            # print('--------------------------------------------')
            # print('kida after mutation\n{}'.format(kida))
            # print('kidb after mutation\n{}'.format(kidb))
        else:
            # print('parents will possibly mutate and reincarnate')
            # print('parent 1: {}'.format(self.ga_pop[p1]))
            # print('parent 2: {}'.format(self.ga_pop[p2]))
            kida = ga_pop[p1]
            kidb = ga_pop[p2]
            # print('\n--------------------------------------------')
            # print('kida before mutation\n{}'.format(kida))
            # print('kidb before mutation\n{}'.format(kidb))
            # print('--------------------------------------------')
            kida = self.mutate(kida, pm)
            kidb = self.mutate(kidb, pm)
            # print('--------------------------------------------')
            # print('kida after mutation\n{}'.format(kida))
            # print('kidb after mutation\n{}'.format(kidb))
        # ga_pop2[p1] = np.array(kida.copy())
        # ga_pop2[p2] = np.array(kidb.copy())
        # kid_array.append(np.array(kida.copy()))
        # kid_array.append(np.array(kidb.copy()))
        # print('--------------------------------------------\n')
        #return kid_array
        return kida, kidb


################################################################
#      NOTE: Below are the example GASolver class objects
class GASOLVER:
    """Base class for the GA solver objects"""
    def __init__(self, threshold=0.00000, verbose=False, mode="MIN",
                 objective_func=None):
        self.threshold=threshold
        self.verbose = verbose
        self.fitness_array = None
        self.total_fitness=None
        self.gene_prob_tally=None
        self.gene_prob=None
        self.populationsize=None
        self.mode = mode.upper()
        if objective_func is None:
            self.obj_func=self.score_population
        else:
            self.obj_func=objective_func

    def score_population(self, chromosomes, **kwargs):
        score_dict = {}
        for cnt, chromosome in enumerate(chromosomes):
            if self.verbose:
                print(cnt, chromosome)
            score_dict[cnt] = self.obj_func(chromosome)
        self.fitness_array = list(score_dict.values())
        self.total_fitness = np.sum(self.fitness_array)
        self.calculate_gene_prob()
        self.calculate_gene_prob_tally()
        return score_dict

    def convergence_check(self, ga_pop):
        converge = True
        for gdi in range(len(ga_pop-1)):
            for gdi2 in range(gdi + 1, len(ga_pop)):
                if not np.array_equal(ga_pop[gdi], ga_pop[gdi2]):
                    return False
        return True

    def pair_pop(self, popsize, ga_pop):
        pairs = list()
        while len(pairs) < popsize:
            cycle_cnt = 0
            parent1 = self.get_upperRNG(np.random.uniform(0, 1, 2), popsize)
            parent2 = self.get_upperRNG(np.random.uniform(0, 1, 2), popsize)
            while parent1 != parent2:
                parent1 = self.get_upperRNG(np.random.uniform(0, 1, 2), popsize)
                parent2 = self.get_upperRNG(np.random.uniform(0, 1, 2), popsize)
                if cycle_cnt > popsize * 2:
                    if self.convergence_check(ga_pop):
                        print('--------------------convergence ---------------------')
                    if np.random.default_rng().choice([True, False], 1)[0]:
                        parent2 = (parent2 + np.random.choice(range(popsize), 1)[0]) % popsize
                    else:
                        parent1 = (parent1 + np.random.choice(range(popsize), 1)[0]) % popsize
                else:
                    n1, n2 = np.random.uniform(0, 1, 2)
                    parent1 = self.get_upperRNG(n1, popsize)
                    parent2 = self.get_upperRNG(n2, popsize)
                    cycle_cnt += 1
            pairs.append((parent1, parent2))
        return pairs

    def SuvivalOfTheFittest(self, popsize, ga_pop, pm, pc):
        kid_array = list()
        pairs = self.pair_pop(popsize, ga_pop)
        cnt = 0
        while len(kid_array) < popsize:
            # Natural Selection, the most fit carry on
            p1 = pairs[cnt][0]
            p2 = pairs[cnt][1]
            # crossover or not
            crossover = np.random.default_rng().choice([True, False], 1, p=[pc, 1-pc])[0]

    def get_upperRNG(self, rng_num, popsize):
        idx = -1
        parent = -np.inf
        #print('tally?')
        #print(self.gene_prob_tally)
        # print('gene pop size {}'.format(self.pop))
        while parent <= rng_num and idx < popsize:
            idx += 1
            parent = self.gene_prob_tally[idx]
        if idx == popsize:
            idx -= 1
        return idx

    def threshold_check(self, best_score, mode="Min"):
        """
                Used to check if set threshold has been breached
        :param best_score: the current best score
        :param mode: The mode the solver is in.
                        * MIN: desire the lowest value so seek
                               to find a value lower than the threshold
                        * MIN2: see above, but uses slightly different
                                minimization algorithm
                        * MAX: seek to find the highest so want to see if we have
                               gone beyond some threshold
        :return:
        """
        if mode.upper() in ["MIN", "MIN2"]:
            if best_score <= self.threshold:
                return True
        else:
            if best_score >= self.threshold:
                return True
        return False

    def calculate_gene_prob(self):
        self.gene_prob = (self.fitness_array / self.total_fitness)
        # reverse the stream
        # sort it
        self.gene_prob = dict(sorted(self.gene_prob.items(), key=lambda x: x[1]))
        rdict = {}
        rprob = sorted(self.gene_prob.values(), reverse=True)
        for k, v in zip(self.gene_prob, rprob):
            self.gene_prob[k] = v
        return

    def calculate_gene_prob_tally(self):
        self.gene_prob_tally = np.cumsum(self.gene_prob)
        return

class FEATURESELECTOR_SOLVER(GASOLVER):
    def __init__(self, features, target, datafile, Learner, fitparams, scorer,
                 cv=2, verbose=True, threshold=.0000, to_drop=None, size=None):
        super().__init__(threshold=threshold, verbose=verbose)
        if target in features:
            del features[features.index(target)]
        self.fitparams = fitparams
        self.features = features
        self.strlen = len(features)
        self.target = target
        self.Learner = Learner
        self.cv = cv
        self.scorer = scorer
        self.dataset = pd.DataFrame(pd.read_excel(datafile, usecols=features+[target], nrows=size).dropna(),
                                    columns=features+[target])
        self.N = self.dataset.shape[0]
        self.D = self.strlen
        if to_drop is not None:
            self.dataset.drop(columns=to_drop, inplace=True)

    def select_features(self, chromosome):
        retl = list()
        for cnt, chrs in enumerate(chromosome):
            print(cnt)
            if chrs == 1:
                retl.append(self.features[cnt])
        return retl

    def score_population(self, chromosomes):
        scores_dict = {}
        for cnt, chromosome in enumerate(chromosomes):
            test_features = list()
            # use chromsome to select features
            for i, bit in enumerate(chromosome):
                if bit == 1:
                    test_features.append(self.features[i-1])
            # print("\nthe feates: ",test_features, "\n")
            X = self.dataset[test_features]
            Y = self.dataset[[self.target]]
            #print(X.head())
            #print(Y.head())
            scores = cross_val_score(self.Learner, X, Y.values.ravel(),
                                     scoring=self.scorer,
                                     cv=self.cv, fit_params=self.fitparams)
            # print('score: {}'.format(np.mean(np.abs(scores))))
            scores_dict[cnt] = np.mean(np.abs(scores))
            # print("score dict: {}".format(scores_dict))
        return scores_dict

class DGS_SOLVER(GASOLVER):
    def __init__(self, dgfile, facilityfile, distancefile=None,
                 threshold=100.0, verbose=False, mode="MIN"):
        super().__init__(threshold=threshold, verbose=verbose, mode=mode)
        self.dgset = self.loadDG(dgfile)
        self.nodes = self.loadNodes(facilityfile, distancefile)

    def loadDG(self, dgfile):
        """
            Loades the given excel file reprsenting DG's into a data frame
        :param dgfile: the path/name of an excel file assumes the following columns:

        :return: the DGS object with the data loaded
        """
        return DGS(dgfile)

    def loadNodes(self, nodefile, distancefile):
        return Facilties(sourcefile=nodefile, distancefile=distancefile)

    def score_population(self, chromosomes, **kwargs):
        scores_dd = dict()
        cnt = 0
        # score the solutions
        # for each solution use it as a solution for assignement
        for chromosome in chromosomes:
            # use chromosome to assign dg's to nodes
            self.nodes.assignDGs(self.dgset, chromosome)

            # calculate the cost of this assignment
            dgcost = self.dgset.getCost(chromosome)
            facCost = self.nodes.getCost()
            cost = dgcost + facCost
            #         print("----ch, dg, fac, total-------------")
            #         print(chromosome)
            #         print(dgcost)
            #         print(facCost)
            #         print(cost)
            #         print("-----------------\n")
            scores_dd[cnt] = cost
            cnt += 1
        return scores_dd

####################################################################################
#      NOTE: Below are combinations of GAOPTM and Solver objects in one class
#            This will allow for a singular analysis of the DGS assignement problem
class DG_Optimizer:
    def __init__(self, population_size,  pm, pc,
                 dgfile, nodefile, distancefile, generations=60, strlen=None,
                 budgetfile=None, verbose=False, init_func=None,
                 threshold=.00001, Nopen=None, oneprob=None, mode="MIN"):
        self.population_size=population_size
        self.strlen=strlen
        self.pm=pm
        self.pc=pc
        self.generations=generations
        self.mode=mode
        self.dgSolver = DGS_SOLVER(dgfile=dgfile, facilityfile=nodefile, mode=mode,
                                   distancefile=distancefile, threshold=threshold,
                                   verbose=verbose)
        if strlen is None:
            self.strlen = self.dgSolver.dgset.N
            strlen = self.strlen
        print("strlen: ", strlen)
        self.gaoptmzr = GAOPTM(population_size, strlen, self.dgSolver, pm=pm, pc=pc,
                               verbose=verbose, generations=generations,
                               init_func=init_func, oneprob=oneprob, Nopen=Nopen)
        self.budget=self.loadBUDG(budgetfile)
        self.verbose = verbose
        self.threshold = threshold
        self.Nopen = Nopen
        self.oneprob = oneprob
        self.best_score=None
        self.best_scores=None
        self.best_solution=None
        self.avg_scores=None


    def loadBUDG(self, budgetfile):
        if budgetfile is None:
            return budgetfile
        return pd.read_csv(budgetfile, low_memory=False)

    def optimize(self, probability_gen=None, pair_selector=None,child_gen=None,):
        best_scores, avg_scores, best_score, best_solution = self.gaoptmzr.run_generations(
                                                            pm=self.pm,
                                                            pc=self.pc,
                                                            probability_gen=probability_gen,
                                                            pair_selector=pair_selector,
                                                            child_gen=child_gen,
                                                            generations=self.generations,
                                                            mode=self.mode, eval=min)
        self.best_score=best_score
        self.best_scores=best_scores
        self.best_solution=best_solution
        self.avg_scores=avg_scores

    def show_results(self,  figsize=(20, 20), popsize="",
                     fontdict={"size":20, 'weight':'bold'}, prop={"size":20},
                     ):
        cnt=0
        title = "Average Score vs Generation" + "\npm:{}, pc:{}, pop: {}, open:{}".format(self.pm, self.pc,
                                                                                          self.population_size, self.Nopen)
        axbsg = pd.DataFrame({"Population_Average": self.avg_scores,
                              "Generation": range(len(self.avg_scores))}).plot(
            "Generation", "Population_Average", title=title, figsize=figsize)
        axbsg.set_title(title, fontdict=fontdict)
        axbsg.set_xlabel("Generation", fontdict)
        axbsg.set_ylabel("Average Score", fontdict)
        axbsg.legend(prop=prop)

        title2 = "Best Score vs Generation" + "\npm:{}, pc:{}, pop: {}, open:{}".format(self.pm, self.pc,
                                                                                        self.population_size, self.Nopen)
        axbsg2 = pd.DataFrame({"best": self.best_scores,
                               "Generation": range(len(self.best_scores))}).plot("Generation", "best",
                                                                                                 title=title2,
                                                                                                 figsize=figsize)
        t = axbsg2.set_title(title2, fontdict=fontdict)
        axbsg2.set_xlabel("Generation", fontdict)
        t = axbsg2.set_ylabel("Best Score", fontdict)

        plt.show()




def p(s):
    print(s)
    return

def plot_ga_analysis(df, seed,  N, l, G, pm, pc,show_it=False, save_img=False, dir_name='', nurture=False, ga_mod=None):
    """ used to plot the different runs stored in the data frame passed"""
    xarray = df['Gen'.format(seed)]

    run_num = df['runs'][0]

    legend = [str(x+1) for x in range(run_num)]

    nurt = ""
    if nurture:
        nurt = '(Nurtured)'
    print(legend)
    for y in df.columns.tolist():
        if 'Gen' not in y:
            print(y)
            if 'avgcorrect' in y:
                print('should be avgcorrect {}'.format(y))
                # grab the seed for the legend
                plt.figure(1)
                plt.plot(xarray, df[y])
            elif 'mostfit' in y:
                print('should be mostfit {}'.format(y))
                plt.figure(2)
                plt.plot(xarray, df[y])
            elif 'avgfit' in y:
                print('should be avgfit {}'.format(y))
                plt.figure(3)
                plt.plot(xarray, df[y])
    plt.figure(1)
    plt.xlabel('generation')
    plt.ylabel('average correct per generaton')
    plt.title('Generation vs. The average Correct {}\nl:{}, N:{}, pc:{}, pm:{}, G:{}, seed:{}'.format(nurt,
                                                                                                      l, N, pc, pm, G, seed))
    plt.figlegend(legend)
    plt.savefig(dir_name + 'avg_corr_l{}_N{}_G{}_pm{}_pc{}.png'.format(l, N, G, pm, pc))

    plt.figure(2)
    plt.xlabel('generation')
    plt.ylabel('most fit per generation')
    plt.title('Generation vs. MostFit {}\nl:{}, N:{}, pc:{}, pm:{}, G:{}, seed:{}'.format(nurt, l, N, pc, pm, G, seed))
    plt.figlegend(legend)
    plt.savefig(dir_name + 'mostfit_l{}_N{}_G{}_pm{}_pc{}.png'.format(l, N, G, pm, pc))

    plt.figure(3)
    plt.xlabel('generation')
    plt.ylabel('average fitness per generation')
    plt.title('Generation vs. Average Fitness {}\nl:{}, N:{}, pc:{}, pm:{}, G:{}, seed:{}'.format(nurt, l, N, pc, pm, G, seed))
    plt.figlegend(legend)
    plt.savefig(dir_name + 'avg_fit_l{}_N{}_G{}_pm{}_pc{}.png'.format(l, N, G, pm, pc))

    plt.figure(4)
    plt.bar(ga_mod.prob_wrong_dict.keys(), ga_mod.prob_wrong_dict.values())
    plt.xlabel('Index of incorrect bit')
    plt.ylabel('probability of incorrectness')
    #plt.xticks(ga_mod.prob_wrong_dict.keys())
    plt.title('The most probable incorrect bit {}\nl:{}, N:{}, pc:{}, pm:{}, G:{}'.format(nurt, l, N, pc, pm, G))
    plt.savefig(dir_name + 'prob_wrng_bits_l{}_N{}_G{}_pm{}_pc{}.png'.format(l, N, G, pm, pc))
    if show_it:
        plt.show()

def get_run(strg):
    return strg[0:strg.index('_')]

def grab_all_run(df):
    desired_labels = ['avgcorrect','avgfit','mostfit',]

def assign_dgs(dgs, nodes):
    pass


##################################################################
#  The two main classes for the DG optimization. The represent the
#  Buildings with energy demands (Facilities) and the set of possible
#  distributed generation sites (DGS)
class Facilties:
    def __init__(self, sourcefile, distancefile, verbose=False, **kwargs):
        self.verbose = verbose
        self.kwargs = kwargs
        self.nodes = pd.read_csv(sourcefile, low_memory=False).dropna()
        self.distances = pd.read_csv(distancefile, low_memory=False).dropna()
        # self.distances.drop(columns=["demand_node"], inplace=True)
        self.N = self.nodes.shape[0]
        self.assignments = {i: 0 for i in self.nodes.index.tolist()}
        self.nodes['suppliedPower'] = np.zeros(self.N)
        self.reset()

    def reset(self):
        self.assignments = {i:0 for i in self.nodes.index.tolist()}
        self.nodes['suppliedPower'] = np.zeros(self.N)

    def showAssignments(self):
        for id in self.assignments:
            print(id, self.assignments[id])

    def get_closest(self, distances, fid, opens):
        bddict = dict()
        for c in opens:
            bddict[c] = distances.loc[fid, distances.columns.tolist()[c + 1]]
        bddict = dict(sorted(bddict.items(), key=lambda x: x[1]))
        if self.verbose:
            print("closest array ",bddict)
        return bddict

    def getUnmetDemandCost(self,):
        cost = 0
        for fid in range(len(self.nodes)):
            supply =int(self.nodes.loc[fid, 'suppliedPower'])
            if supply == 0:
                non_assign_pen = 100000
                print("\t\t\t\t\t\t\t\tnon assignment")
            else:
                non_assign_pen = 1

            diff_demand_supplied =self.nodes.loc[fid, "demand"] - self.nodes.loc[fid, 'suppliedPower']
            diff_demand_supplied = max(0, diff_demand_supplied)
            cost += diff_demand_supplied * self.nodes.loc[fid, 'penalty']
        return cost

    def getTransmissionCost(self, ):
        totalcost = 0
        for fid in self.assignments:
            #print("Transmission cost params: fid{}, assign{}".format(fid, self.assignments[fid]))
            #print("distance: {}".format(self.distances.iloc[fid, self.assignments[fid]+1]))
            totalcost += self.distances.iloc[fid, self.assignments[fid]+1]
        return totalcost

    def getCost(self, ):
        return self.getUnmetDemandCost() + self. getTransmissionCost()

    def setAssignment(self, idx, assignmentId):
        self.assignments[idx] = assignmentId

    def assignDGs(self, dgs, onlist):
        # reset all assignments
        self.reset()
        dgs.reset()

        # using the chromosomes called an onlist here,
        # get the indices of those that are open
        # according to the onlist
        dgNodes = dgs.getOn(onlist).index.tolist()

        # check for an empty list if the list is empty return max cost for each
        # set the empty flag and return max cost
        # go through the facilities/nodes and get the closest open
        # dg
        for fid in self.nodes.index.tolist():
            # get a sorted list of the Dg's that are open from closest
            # to farthest
            dgfound=False
            for dgId in self.get_closest(self.distances, fid, dgNodes):
                # if this one has enough power left to run it
                opt =dgs.dg_df.loc[dgId, 'output']
                rpwr =dgs.dg_df.loc[dgId, 'rated_power']
                #print("DG {}: output: {}, rated power: {} suitable?: {}".format(dgId, opt, rpwr, opt < rpwr))
                if dgs.dg_df.loc[dgId, 'output'] < dgs.dg_df.loc[dgId, 'rated_power']:
                    dgs.setAssignment(dgId, fid)
                    #dgs.assignments[dgId].append(fid)
                    #self.assignments[fid].append(dgId)
                    self.setAssignment(fid, dgId)
                    # if this one has enough for the required demand
                    if dgs.dg_df.loc[dgId, 'rated_power'] - dgs.dg_df.loc[dgId, 'current_output'] > self.nodes.loc[fid,'demand']:
                        # add this nodes demand to what the dg needs to(is) output(ing)
                        dgs.dg_df.loc[dgId, 'current_output'] += self.nodes.loc[fid, 'demand']
                        # set this nodes to be fully supplied
                        self.nodes.loc[fid, 'suppliedPower'] =  self.nodes.loc[fid, 'demand']
                    else:
                        # set this dg to be outputting its max
                        dgs.dg_df.loc[dgId, 'current_output'] = dgs.dg_df.loc[dgId, 'rated_power']
                        # This node gets what ever the dg had left to give
                        self.nodes.loc[fid, 'suppliedPower'] = dgs.dg_df.loc[dgId, 'rated_power'] - dgs.dg_df.loc[dgId, 'output']
                    # once we have found a suitable DG break the loop and assign for the next facility
                    break
                #print("")
                #print("")
        #print("")
        #print("\n\ndgs.assignments:\n{}".format(dgs.assignments))
        return

class DGS:
    """Set of DG's"""
    def __init__(self, source_file, excessFactor='LOW', initialize=False):
        excessFactor = excessFactor.upper()
        self.excessCol = ""
        if excessFactor.upper() in ["LOW", "MEDIUM", "HIGH"]:
            self.excessCol = 'excess_penetration_cost_' + excessFactor.upper()
        else:
            print("unknown excess cost type: {}".format(excessFactor))
            print("using low by default")
            self.excessCol = 'excess_penetration_cost_' + "LOW"

        self.initialize = initialize
        self.dg_df = pd.read_csv(source_file, low_memory=False).dropna()

        # below step makes sure the indices will match the ID
        self.dg_df = pd.DataFrame(self.dg_df.values, columns=self.dg_df.columns.tolist())
        self.N = self.dg_df.shape[0]
        self.opens = list()
        self.dg_df['costs'] = np.zeros(self.N)
        self.dg_df['current_output'] = np.zeros(self.N)
        self.on_list = list() # set of indices that are open for operation
        self.assignments = {idx:[] for idx in self.dg_df.index.tolist()}
        self.excessFactor = excessFactor
        self.features = self.dg_df.columns.tolist()
        self.reset()

    def reset(self):
        #self.dg_df['output'] = np.zeros(self.N)
        self.assignments = {i:[] for i in range(self.dg_df.shape[0])}
        self.dg_df['costs'] = np.zeros(self.N)
        if self.initialize:
            self.dg_df['current_output'] = np.zeros(self.N)
        else:
            self.dg_df['current_output'] = self.dg_df[['output']].values
        self.opens = list()

    def setOn(self, onlist):
        self.on_list = onlist

    def showAssignments(self):
        for id in self.assignments:
            if len(self.assignments[id]) > 0:
                print(id, self.assignments[id])

    def setAssignment(self, idx, assignmentId):
        self.assignments[idx].append(assignmentId)

    def getOn(self,on_list=None):
        if on_list is not None:
            self.opens = on_list
            # print("opens then: ", self.opens)
        return self.dg_df.loc[ [v == 1 for v in self.opens], :]

    def getCost(self, on_list=None):
        # set up an empty fitness array
        #self.fittness_array = np.zeros(len(self.))
        self.costs = {}
        cost = 0
        # pull the open DG's
        dgNodes = self.getOn(on_list=on_list)
        # the onlist allows you to only look at those dg's that
        # are in opertion
        for id in dgNodes.index.tolist():
            #dg = dgNodes.dg_df.iloc[id, :]
            investmentCost = dgNodes.loc[id,"rated_power"] * dgNodes.loc[id, 'investment_cost']
            operationCost = dgNodes.loc[id, "current_output"] * dgNodes.loc[id, 'o&m_cost']
            # self.opens.append(id)
            # cost += investmentCost + operationCost
            # penalize wasted power
            # if the dg's rated power is less than the current output
            # penalize the
            if dgNodes.loc[id, 'rated_power'] < dgNodes.loc[id, 'current_output']:
                excessCost = dgNodes.loc[id, 'rated_power'] - dgNodes.loc[id, 'current_output']
                excessCost *= dgNodes.loc[id,self.excessCol]
            else:
                excessCost = 0
            dgcost = investmentCost + operationCost + excessCost
            cost += dgcost
            self.costs[id] = dgcost


        # sort the dg index keyed dictionary by the cost values by their
        #self.costs = dict(sorted(self.costs.items(), key=lambda x:x[1]))
        # if there was not cost produced
        if cost == 0:
            # print("\n\n\n\t\t\t------Found an empty -------\n\n\n")
            # max_power =dgNodes.loc[:, 'rated_power'].max()
            # max_invc = dgNodes.loc[:, 'investment_cost'].max()
            # max_out =dgNodes.loc[:, 'output'].max()
            # om_cst =dgNodes.loc[:, 'o&m_cost'].max()
            # ex_cst = dgNodes.loc[:, self.excessCol].max()
            # print("power_r: {}, invc: {}, out: {}, exc: {}, N: {}".format(
            #     max_power, max_invc, max_out, om_cst, ex_cst, self.N
            # ))
            # cost = dgNodes.loc[:, 'rated_power'].max() * dgNodes.loc[:, 'investment_cost'].max() * self.N
            # cost +=dgNodes.loc[:, 'output'].max() * dgNodes.loc[:, 'o&m_cost'].max() * dgNodes.loc[:, self.excessCol].max() * self.N
            #cost = max_power * max_invc * self.N
            #cost += max_out * om_cst *  ex_cst * self.N
            cost = 10000000000
            print('returning max cost: ', cost)
        return cost


def print_log(filename, ga_mod, seed, N, l, G, pm, pc, show_it=False, save_img=True, dir_name='',
              new_challenge=False, nurture=False):
    if os.path.isfile(dir_name + filename):
        # save the data for a big plot
        df = pd.read_excel(dir_name + filename)
        run = df['runs'][0]+1
        df['{}_avgfit_{}'.format(run, seed)] = ga_mod.avg_fitness_gen
        print('the run')
        print(get_run(df.columns.tolist()[0]))
        df['{}_mostfit_{}'.format(run, seed)] = ga_mod.mostfit_of_gen
        df['{}_avgcorrect_{}'.format(run, seed)] = ga_mod.avg_correct_gen
        df['Gen'.format(seed)] = ga_mod.generationX
        df['runs'] = list([run]*len(ga_mod.avg_correct_gen))
        df.to_excel(dir_name + filename, index=False)
        '''
        # get the xarray
        xarray = df['{}_Gen'.format(seed)]
        for y in df.columns.tolist():
            if 'Gen' not in y:
                print(y)
                if 'avgcorrect' in y:
                    plt.figure(1)
                    plt.xlabel('generation')
                    plt.ylabel('average correct per generaton')
                    plt.title('Generation vs. The average Correct ')
                    plt.plot(xarray, df[y])
                elif 'mostfit' in y:
                    plt.figure(2)
                    plt.xlabel('generation')
                    plt.ylabel('most fit per generation')
                    plt.title('Generation vs. MostFit')
                    plt.plot(xarray, df[y])
                elif 'avgfit' in y:
                    plt.figure(3)
                    plt.xlabel('generation')
                    plt.ylabel('average fitness per generation')
                    plt.title('Generation vs. Average Fitness')
                    plt.plot(xarray, df[y])
        plt.show()
        '''
    else:
        df = pd.DataFrame({
            '{}_avgfit_{}'.format(1, seed): ga_mod.avg_fitness_gen,
            '{}_mostfit_{}'.format(1, seed): ga_mod.mostfit_of_gen,
            '{}_avgcorrect_{}'.format(1, seed): ga_mod.avg_correct_gen,
            'Gen': ga_mod.generationX,
            'runs':list([1]*len(ga_mod.avg_correct_gen)),
        })
        df.to_excel(dir_name + filename, index=False)
    plot_ga_analysis(df, seed, N=N, l=l, G=G, pm=pm, pc=pc, show_it=show_it, save_img=save_img, dir_name=dir_name,
                     nurture=nurture, ga_mod=ga_mod)

class GA_Model:
    """

    """
    def __init__(self, fitness_method, pop_size=30, gene_length=20, pm=None, pc=.06, seed=1911,
                 num_gen=10,verbose=True):
        np.random.seed(seed)
        self.verbose=verbose
        # set the given fitness method as the one to use
        # needs to set
        self.fitness_method = fitness_method
        self.gen_limit = 100000
        self.current_gen = 0
        self.pop = pop_size  # store the population of the genes
        self.gene_length = gene_length  # store the length
        self.pm = pm  # mutation probability
        if self.pm is None:
            self.pm = np.around(1 / self.pop, 3)
        self.pc = pc  # crossover probabilty
        self.seed = seed
        self.num_gen = num_gen
        self.breeding_runs = int(np.ceil(self.pop / 2, ))  # get how many iterations of breeding to perform
        self.ga_pop = np.array([np.random.choice([0, 1], gene_length, replace=True) for i in range(pop_size)])
        #self.ga_pop2 = np.array([np.random.choice([0, 1], gene_length, replace=True) for i in range(pop_size)])
        self.gene_prob, self.gene_prob_tally = np.zeros(pop_size), np.zeros(pop_size)
        self.total_fitness = 0
        self.fitness_array = np.zeros(pop_size)  # stores the fitness of each gene of the current population
        self.generation = 0  # used to keep track of what generation the population is on
        # TODO: This is unneeded, can just create on the fly if needed
        self.generationX = list([])  # just a list of the number of the generation used for plotting
        self.avg_fitness_gen = list()  # used to track average fitness of current generation
        #self.avg_correct_gen = list()  # used to track the average number of ones per generation
        self.mostfit_of_gen = list()
        self.kid_array = list([])
        self.prob_wrong_dict = dict()
        self.prob_wrng_t_cnt = 0

    def calculate_fittness(self, **kwargs):
        self.fitness_method(self.ga_pop, **kwargs)

    def run_generations(self, verbose=None, **kwargs):
        if verbose is None:
            verbose = self.verbose
        for g in range(self.num_gen):
            self.current_gen = g + 1
            if verbose:
                print('')
                print('-----------   Generation {}   ------------------'.format(self.generation + 1))
                print('')
            # was calculate fitness
            self.fitness_array = self.fitness_method(self.ga_pop, **kwargs)
            # TODO: make this a thing
            self.select_pairs()
            self.generation += 1
            if verbose:
                print("---------------------------------------------------------\n")

    def select_pairs(self):
        # set up empty list of GA pairs to procreate
        pairings = list([])
        self.kid_array = list([])
        #
        # breeding loop
        for i in range(self.breeding_runs):
            # calculate current generations fitness for later plotting
            # get two random number between 0(inclusive) and 1(exclusive)
            # these will be used to as a range to look between
            consanguinity = True  # assume inbreeding, and keep trying until not
            parent1, parent2 = 0, 0
            # avoid the inbreeding
            # np.random.seed(None)
            # n1 = np.random.uniform(0, 1, 1)[0]
            # n2 = np.random.uniform(0, 1, 1)[0]
            n1, n2 = np.random.uniform(0, 1, 2)
            parent1 = self.get_upper(n1)
            parent2 = self.get_upper(n2)
            cycle_cnt = 0
            # check for inbreeding
            while consanguinity:
                # np.random.seed(None)
                # n1, n2 = np.random.uniform(0, 1, 2)
                # print('night {} ;)'.format(i+1))
                # print('the first num {}'.format(n1))
                # print('the second num {}'.format(n2))
                # print('-------------------------------------')
                # for each of the two number find the minimum value
                # n1 = np.random.uniform(0, 1, 1)[0]
                # n2 = np.random.uniform(0, 1, 1)[0]
                # parent1 = self.get_upper(n1)
                # parent2 = self.get_upper(n2)
                # p('parent 1: {}'.format(parent1))
                # p('parent 2: {}'.format(parent2))
                # if not np.array_equal(self.ga_pop[parent1], self.ga_pop[parent2]):
                if parent1 != parent2:  # make sure we get unique parent indices
                    # parent1 = self.get_upper(n1)
                    # parent2 = self.get_upper(n2)
                    # print('\nThey are not related!!!!')
                    # print('parent 1 ({}):\n{}'.format(parent1, self.ga_pop[parent1]))
                    # print('parent 2 ({}):\n{}\n'.format(parent2, self.ga_pop[parent2]))
                    consanguinity = False
                else:
                    # print('\nUh Oh!, kissing cousins, lets try again')
                    # print('num 1: {}'.format(n2))
                    # print('num 2: {}'.format(n1))
                    # print('parent1: {}'.format(parent1))
                    # print('parent2: {}'.format(parent2))
                    # print('parent 1 genes:\n{}'.format(self.ga_pop[parent1]))
                    # print('parent 2:\n{}\n'.format(self.ga_pop[parent2]))
                    # print('the pop\n{}'.format(self.ga_pop))
                    if cycle_cnt > self.pop * 2:
                        if self.convergence_check():
                            print('--------------------convergence ---------------------')
                            consanguinity = False
                        if n1 > n2:
                            parent2 = (parent2 + np.random.choice(range(self.pop), 1)[0]) % self.pop
                        else:
                            parent1 = (parent1 + np.random.choice(range(self.pop), 1)[0]) % self.pop
                    else:
                        n1, n2 = np.random.uniform(0, 1, 2)
                        parent1 = self.get_upper(n1)
                        parent2 = self.get_upper(n2)
                        cycle_cnt += 1

            if self.verbose:
                p('parent 1: {}'.format(parent1))
                p('parent 2: {}'.format(parent2))
                print('parent 1 genes:\n{}'.format(self.ga_pop[parent1]))
                print('parent 2 genes:\n{}\n'.format(self.ga_pop[parent2]))
            # store the paring
            pairings.append([parent1, parent2])

            # once we get a good breeding pair perform crossover creating two children
            # get 0 or 1 using crossover probability to see if we cross over or not
            crossover = np.random.choice([False, True], 1, p=[1 - self.pc, self.pc])[0]
            # if we need to do crossover do so
            crossover_point = None
            # print('cross over? {}'.format(crossover))
            if crossover:
                # get random cross over point
                # set range from 1, to len-1 so dont just reincarnate
                crossover_point = int(np.random.choice(list(range(1, self.gene_length - 1)), 1))
            self.procreate(parent1, parent2, crossover_point)
        # once the required number of mating trials have been run store the kids as a numpy array and copy
        # replace the old with the next generation
        self.kid_array = np.array(self.kid_array)
        # now if in learning mode do random guessing on the even bits
        if self.learning_mode:
            print('\n\n-----------------learning mode on--------------------------\n\n')
            self.nurture()
        # print('\nthe children')
        # print(self.ga_pop2)
        # print(self.kid_array)
        # print('the their parents')
        # print(self.ga_pop)
        # print('the parings that brought this generation')
        # print(pairings)
        # print()
        # move to next generation, to boldly go
        # where no one has gone before
        self.ga_pop = self.kid_array
        # print('\nthe children')
        # print(self.kid_array)
        # print('the their parents')
        # print(self.ga_pop)

        # greater than it and chose that number, making sure the numbers are unique

    def get_upper(self, num):
        idx = 0
        parent = -np.inf
        #print('tally?')
        #print(self.gene_prob_tally)
        # print('gene pop size {}'.format(self.pop))
        while parent <= num and idx < self.pop:
            parent = self.gene_prob_tally[idx]
            idx += 1
        if idx == self.pop:
            idx -= 1
        return idx




class GA_Model_Tester:
    def __init__(self, pop_size=30, gene_length=20, pm=None, pc=.06, seed=1911,
                 num_gen=10, learning_mode=False, new_challenge=False):
        # create the matrix that will represents the population
        self.new_challenge=new_challenge        # used to test the ability of GA to handle change of objective (fitness) function
        np.random.seed(seed)
        self.gen_limit = 100000
        self.current_gen = 0
        # TODO: rename this
        self.learning_mode = learning_mode
        self.ga_pop = np.array([np.random.choice([0, 1], gene_length, replace=True) for i in range(pop_size)])
        self.ga_pop2 = np.array([np.random.choice([0, 1], gene_length, replace=True) for i in range(pop_size)])
        self.pop = pop_size  # store the population of the genes
        self.gene_length = gene_length  # store the length
        self.pm = pm  # mutation probability
        if self.pm is None:
            self.pm = np.around(1 / self.pop, 3)
        self.pc = pc  # crossover probabilty
        self.seed = seed
        self.num_gen = num_gen
        self.breeding_runs = int(np.ceil(self.pop / 2, ))  # get how many iterations of breeding to perform
        self.gene_prob = []
        self.gene_prob_tally = []
        self.total_fitness = 0
        self.fitness_array = np.zeros(pop_size)  # stores the fitness of each gene of the current population
        self.generation = 0  # used to keep track of what generation the population is on
        self.generationX = list([])  # just a list of the number of the generation used for plotting
        self.avg_fitness_gen = list([])  # used to track average fitness of current generation
        self.avg_correct_gen = list([])  # used to track the average number of ones per generation
        self.mostfit_of_gen = list([])
        self.kid_array = list([])
        self.prob_wrong_dict = dict()
        self.prob_wrng_t_cnt = 0
        #np.random.seed(None)

    def run_generations(self):
        for g in range(self.num_gen):
            self.current_gen = g + 1
            print('')
            print('-----------   Generation {}   ------------------'.format(self.generation + 1))
            print('')
            self.calculate_fitness()
            self.select_pairs()
            self.generation += 1
            print('')
            print('')

        if self.new_challenge:
            print('--------->Here Comes A New Challenger')
            print('-------------->Here Comes A New Challenger')
            print('------------------------->Here Comes A New Challenger')
            for g in range(int(20)):
                print('')
                print('-----------   Generation {}   ------------------'.format(self.generation + 1))
                print('')
                self.calculate_fitness()
                self.select_pairs()
                self.generation += 1
        # calculate the probability of incorrect bits in each positon
        for k in self.prob_wrong_dict:
            self.prob_wrong_dict[k] = (self.prob_wrong_dict[k]/self.prob_wrng_t_cnt)/self.pop
        self.prob_wrong_dict = sort_dict(self.prob_wrong_dict, reverse=True)

    def calculate_fitness(self):
        # go through the population
        # calculating the individual fitness of each
        for gdi in range(self.pop):

            num = self.get_num_from_bin(self.ga_pop[gdi])
            # print('number is {}'.format(num))
            # print('for {}'.format(self.ga_pop[gdi]))
            # self.fitness_array[gdi] = (num/(2**self.gene_length))**10
            """ if the generation is lower than the limit"""
            if self.generation < self.num_gen:
                self.fitness_array[gdi] = self.fit_func1(num)
            else:
                print('using fitness function 2')
                self.fitness_array[gdi] = self.fit_func2(num)
        # print('the fitness array\n{}'.format(np.array2string(self.fitness_array)))
        # calculate the total fitness
        self.total_fitness = self.fitness_array.sum()
        self.calculate_gene_prob()
        self.calculate_gene_prob_tally()
        # print('inside calculate fitness the tally?')
        # print(self.gene_prob_tally)
        # print('the probs of the genes')
        # print(self.gene_prob)
        # log the generation
        self.generationX.append(self.generation)  # used for plotting later
        # log the average fitness
        # print('fitness array')
        # print(self.fitness_array)
        avg = self.fitness_array.mean()
        # print('average: {}'.format(avg))
        self.avg_fitness_gen.append(avg)
        print('average for generation {} is {}'.format(self.generation, avg))
        # log the fitness of the most fit individual of population
        #self.mostfit_of_gen.append(self.gene_prob.max())
        self.mostfit_of_gen.append(self.fitness_array.max())
        print('The most fit value for gen: {} is {}'.format(self.generation,self.mostfit_of_gen[-1]))
        # log the average number of correct (1's)
        # print('there are {} 1\'s total'.format(np.count_nonzero(self.ga_pop)))
        if self.generation <= len(self.ga_pop):
            self.avg_correct_gen.append(np.count_nonzero(self.ga_pop) / self.pop)

        else:
            self.avg_correct_gen.append(np.count_nonzero(self.ga_pop==0) / self.pop)
            self.calculate_prob_wrong(1)
        if self.num_gen - self.generation < 5:      # if there are only 4 gen left, start looking at wrong vals
                self.calculate_prob_wrong(0)
        # print(self.avg_correct_gen)

    def fit_func1(self, num):
        return (num / (2 ** self.gene_length)) ** 10

    def fit_func2(self, num):
        return (1 - (num / (2 ** self.gene_length))) ** 10

    def calculate_prob_wrong(self, wrng):
        for row in self.ga_pop:
            for bit in range(len(row)):
                if row[bit] == wrng:
                    if bit not in self.prob_wrong_dict:
                        self.prob_wrong_dict[bit] = 0
                    self.prob_wrong_dict[bit] += 1
        self.prob_wrng_t_cnt += 1

    def get_num_from_bin(self, bin_array):
        """
            will calculate the numerical value of a binary array
        :param bin_array: a binary array of ones and zeros
        :return: the integer value
        """
        rnum = 0
        # print('')
        # print('the array {}'.format(np.array2string(bin_array)))
        # print('')
        for i in range(-1, -len(bin_array) - 1, -1):
            # print('i:{}'.format(i))
            if bin_array[i] == 1:
                rnum += 2 ** (abs(i + 1))
                # print('rumn is now {} at i of {}'.format(rnum, abs(i+1)))
        return rnum

    def calculate_gene_prob(self):
        self.gene_prob = (self.fitness_array / self.total_fitness)
        return

    def calculate_gene_prob_tally(self):
        self.gene_prob_tally = np.cumsum(self.gene_prob)
        return

    def select_pairs(self):
        pairings = list([])
        self.kid_array = list([])
        #print('running {} pairings total'.format(self.breeding_runs))
        # breeding loop
        for i in range(self.breeding_runs):
            # calculate current generations fitness for later plotting
            # get two random number between 0(inclusive) and 1(exclusive)
            # these will be used to as a range to look between
            consanguinity = True  # assume inbreeding, and keep trying until not
            parent1, parent2 = 0, 0
            # avoid the inbreeding
            #np.random.seed(None)
            #n1 = np.random.uniform(0, 1, 1)[0]
            #n2 = np.random.uniform(0, 1, 1)[0]
            n1, n2 = np.random.uniform(0, 1, 2)
            parent1 = self.get_upper(n1)
            parent2 = self.get_upper(n2)
            cycle_cnt = 0
            # check for inbreeding
            while consanguinity:
                #np.random.seed(None)
                #n1, n2 = np.random.uniform(0, 1, 2)
                # print('night {} ;)'.format(i+1))
                # print('the first num {}'.format(n1))
                # print('the second num {}'.format(n2))
                # print('-------------------------------------')
                # for each of the two number find the minimum value
                #n1 = np.random.uniform(0, 1, 1)[0]
                #n2 = np.random.uniform(0, 1, 1)[0]
                #parent1 = self.get_upper(n1)
                #parent2 = self.get_upper(n2)
                # p('parent 1: {}'.format(parent1))
                # p('parent 2: {}'.format(parent2))
                #if not np.array_equal(self.ga_pop[parent1], self.ga_pop[parent2]):
                if parent1 != parent2:      # make sure we get unique parent indices
                    #parent1 = self.get_upper(n1)
                    #parent2 = self.get_upper(n2)
                    # print('\nThey are not related!!!!')
                    #print('parent 1 ({}):\n{}'.format(parent1, self.ga_pop[parent1]))
                    #print('parent 2 ({}):\n{}\n'.format(parent2, self.ga_pop[parent2]))
                    consanguinity = False
                else:
                    #print('\nUh Oh!, kissing cousins, lets try again')
                    #print('num 1: {}'.format(n2))
                    #print('num 2: {}'.format(n1))
                    #print('parent1: {}'.format(parent1))
                    #print('parent2: {}'.format(parent2))
                    #print('parent 1 genes:\n{}'.format(self.ga_pop[parent1]))
                    #print('parent 2:\n{}\n'.format(self.ga_pop[parent2]))
                    #print('the pop\n{}'.format(self.ga_pop))
                    if cycle_cnt > self.pop*2:
                        if self.convergence_check():
                            print('--------------------convergence ---------------------')
                            consanguinity = False
                        if n1 > n2:
                            parent2 = (parent2 + np.random.choice(range(self.pop), 1)[0])%self.pop
                        else:
                            parent1 = (parent1 + np.random.choice(range(self.pop), 1)[0])%self.pop
                    else:
                        n1, n2 = np.random.uniform(0, 1, 2)
                        parent1 = self.get_upper(n1)
                        parent2 = self.get_upper(n2)
                        cycle_cnt += 1

            p('parent 1: {}'.format(parent1))
            p('parent 2: {}'.format(parent2))
            print('parent 1 genes:\n{}'.format(self.ga_pop[parent1]))
            print('parent 2 genes:\n{}\n'.format(self.ga_pop[parent2]))
            # store the paring
            pairings.append([parent1, parent2])
            # once we get a good breeding pair perform crossover creating two children
            # get 0 or 1 using crossover probability to see if we cross over or not
            crossover = np.random.choice([False, True], 1, p=[1 - self.pc, self.pc])[0]
            # if we need to do crossover do so
            crossover_point = None
            # print('cross over? {}'.format(crossover))
            if crossover:
                # get random cross over point
                # set range from 1, to len-1 so dont just reincarnate
                crossover_point = int(np.random.choice(list(range(1, self.gene_length - 1)), 1))
            self.procreate(parent1, parent2, crossover_point)
        # once the required number of mating trials have been run store the kids as a numpy array and copy
        # replace the old with the next generation
        self.kid_array = np.array(self.kid_array)
        # now if in learning mode do random guessing on the even bits
        if self.learning_mode:
            print('\n\n-----------------learning mode on--------------------------\n\n')
            self.nurture()
        # print('\nthe children')
        # print(self.ga_pop2)
        # print(self.kid_array)
        # print('the their parents')
        # print(self.ga_pop)
        # print('the parings that brought this generation')
        # print(pairings)
        # print()
        # move to next generation, to boldly go
        # where no one has gone before
        self.ga_pop = self.kid_array
        # print('\nthe children')
        # print(self.kid_array)
        # print('the their parents')
        # print(self.ga_pop)

        # greater than it and chose that number, making sure the numbers are unique

    def get_upper(self, num):
        idx = 0
        parent = -np.inf
        #print('tally?')
        #print(self.gene_prob_tally)
        # print('gene pop size {}'.format(self.pop))
        while parent <= num and idx < self.pop:
            parent = self.gene_prob_tally[idx]
            idx += 1
        if idx == self.pop:
            idx -= 1
        return idx

    def procreate(self, p1, p2, cp):
        """
                Performs the crossover function
        :param p1: index of parent 1
        :param p2: index of parent 2
        :param cp: crossover point
        :return:
        """
        if cp is not None:
            # print('kids are created at the cp point of {}'.format(cp))
            # print('the parents are indices {} and {} shown below'.format(p1, p2))
            # print('parent 1: {}'.format(self.ga_pop[p1]))
            # print('parent 2: {}'.format(self.ga_pop[p2]))
            # replace parent 1, with kida
            kida = self.ga_pop[p1].tolist()[:cp] + self.ga_pop[p2].tolist()[cp:]
            kidb = self.ga_pop[p2].tolist()[:cp] + self.ga_pop[p1].tolist()[cp:]
            # print('\n--------------------------------------------')
            # print('kida before mutation\n{}'.format(kida))
            # print('kidb before mutation\n{}'.format(kidb))
            # print('--------------------------------------------')
            kida = self.mutate(kida)
            kidb = self.mutate(kidb)
            # print('--------------------------------------------')
            # print('kida after mutation\n{}'.format(kida))
            # print('kidb after mutation\n{}'.format(kidb))
        else:
            # print('parents will possibly mutate and reincarnate')
            # print('parent 1: {}'.format(self.ga_pop[p1]))
            # print('parent 2: {}'.format(self.ga_pop[p2]))
            kida = self.ga_pop[p1].tolist()
            kidb = self.ga_pop[p2].tolist()
            # print('\n--------------------------------------------')
            # print('kida before mutation\n{}'.format(kida))
            # print('kidb before mutation\n{}'.format(kidb))
            # print('--------------------------------------------')
            kida = self.mutate(kida)
            kidb = self.mutate(kidb)
            # print('--------------------------------------------')
            # print('kida after mutation\n{}'.format(kida))
            # print('kidb after mutation\n{}'.format(kidb))

        self.ga_pop2[p1] = np.array(kida.copy())
        self.ga_pop2[p2] = np.array(kidb.copy())

        self.kid_array.append(np.array(kida.copy()))
        self.kid_array.append(np.array(kidb.copy()))
        # print('--------------------------------------------\n')

    def mutate(self, genome):
        for bit in range(len(genome)):
            # print('the probability of mutation and not {}/{}'.format(self.pm, 1-self.pm))
            flip = np.random.choice([False, True], 1, p=[1 - self.pm, self.pm])[0]
            # print(flip)
            if flip:
                if genome[bit] == 1:
                    genome[bit] = 0
                else:
                    genome[bit] = 1
        return genome

    def nurture(self):
        # go through all individuals in
        # population
        for person in range(self.pop):
            current_best_fit = 0
            current_best_config = None
            # make twenty guesses at the best
            # even bits saving the best as you go and using that as new child
            attempt = None
            for i in range(20):
                # move through even bits randomly
                # make copy of current and use to test
                attempt = self.kid_array[person].copy()
                for bit in range(0, self.gene_length, 2):
                    new_bit = np.random.choice([0, 1], 1)[0]
                    attempt[bit] = new_bit
                # check the fitness of current configuration
                x = self.get_num_from_bin(attempt)
                #if self.current_gen > self.gen_limit:
                if self.generation > self.num_gen:
                        fit_scr = self.fit_func2(x)
                else:
                    fit_scr = self.fit_func1(x)
                if fit_scr > current_best_fit:
                    # store config if it is the current best
                    # and its score
                    current_best_config = attempt.copy()
                    current_best_fit = fit_scr
            # store the best result
            self.kid_array[person] = current_best_config.copy()

    def convergence_check(self):
        converge = True
        for gdi in range(len(self.ga_pop-1)):
            for gdi2 in range(gdi + 1, len(self.ga_pop)):
                if not np.array_equal(self.ga_pop[gdi], self.ga_pop[gdi2]):
                    return False
        return True

# a purely random intilizer function where the rate of 1 and 0
# are randomly random when filling in the population of strlen strings
def pure_rng_init(popsize, strlen, **kwargs):
    pop = list()
    for i in range(popsize):
        pop.append(np.random.default_rng().choice([0, 1], strlen) )
    return pop

