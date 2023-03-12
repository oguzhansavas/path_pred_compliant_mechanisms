# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:05:49 2022

@author: ogsa991b
"""

import numpy as np
import pandas as pd
import sklearn
import pickle
from tensorflow.keras.models import load_model
from numpy.random import randint
from numpy.random import rand
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import time

import sys
sys.path.insert(0, r'C:\Users\ogsa991b\Documents\von_Lars\Path_prediction')
from _220302_data_creation_for_NN_ADDITIONAL_FUNCTIONS import *


# Load model
model_filename = "best_model.h5"
#best_model.save(model_filename)
model = load_model(model_filename)


# Define path (function line and limits)
#function_of_curve_as_str = "math.sin(2*np.pi*x*0.02)*10+500"
#function_of_curve_as_str = "6400-26.5*x+0.03*x**2"
#function_of_curve_as_str = "0.01*(x-100)**2-400"
#function_of_curve_as_str = "0.1*(x-500)**2+500"
function_of_curve_as_str = "0.01*(x-500)**2+600"
#function_of_curve_as_str = "0.005*(x-300)**2+250"
xmin = 425
xmax = 500
func_line = target_line(function_of_curve_as_str,xmin,xmax)


# objective function for ML
def objective(x): # pool_chrom
    preds = model.predict([x]).tolist()[0]
    x_ML_cand = []
    y_ML_cand = []

    for p in preds:
        x_ml_ind, y_ml_ind = get_x_y_from_coords(p)
        x_ML_cand.append(x_ml_ind)
        y_ML_cand.append(y_ml_ind)
        
    pop_fitness = []

    for index in range(len(x_ML_cand)):
        target_coords, fea_coords, fea_coords_transformed, jk_D_min, phi, D, comment_tmp = determine_fitness_ind(x_ML_cand[index],y_ML_cand[index], False, func_line=func_line)
        pop_fitness.append(jk_D_min)
    
    return pop_fitness


# objective function for FEA
num_mapdl = 10
nproc = 1
#single_mapdl_or_pool = "mapdl" #or pool
single_mapdl_or_pool = "pool" #or mapdl
pop = []
def objective_mapdl(x): # pool_chrom
    
    global pop
    
    for ind in x:
        ind.extend([np.pi,0,0,0,0,0])

    #pop = loop_chrom(x)
    pop = pool_loop_chrom(x)
    #sorted_pop = sort_pop_chrom(x, pop)
    pop_fitness = []
    
    for e, ind in enumerate(pop):
        pop_fitness.append(ind.fitness)
        #print(ind.chromosome, x[e])
        
    return pop_fitness
    


# selection (compare k random candidates, select the one with the lowest fitness)
def selection(pop, scores, k=3):
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


# crossover two parents to create two children
def crossover(parent_1, parent_2, r_cross):
	# children are copies of parents at first
	c1, c2 = parent_1.copy(), parent_2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point (-2 at the end; not to select the last bit)
		pt = randint(1, len(parent_1)-2)
		# perform crossover
		c1 = parent_1[:pt] + parent_2[pt:]
		c2 = parent_2[:pt] + parent_1[pt:]
	return [c1, c2]


# mutation
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]


# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of characters
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

#------------------------------------------------------------------------------

# GA
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    start = time.time()
    # initial random pop
    pop = [None] * n_pop
    for i in range(n_pop):
        pop[i] = randint(0, 2, n_bits*len(bounds)).tolist()
    
    # keep track of best solutions
    best = 0
    best_eval = 500 #objective(decode(bounds, n_bits, pop[0]))
    
    for gen in range(n_iter):
		# decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]

        # evaluate all candidates
        scores = objective(decoded)
        
        # check new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best = pop[i]
                best_eval = scores[i]
                print(">%d, f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        
        # list of parents
        parents = [None] * n_pop
        for i in range(n_pop):
            parents[i] = selection(pop, scores)
        
        #create next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents
            parent_1 = parents[i]
            parent_2 = parents[i+1]
            #crossover and mutation
            for c in crossover(parent_1, parent_2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        
        # replace pop with the new set of offsprings
        pop = children
    end = time.time()
    print("Execution time:", (end-start)/60)
    return [best, best_eval, pop, decoded, scores]



# GA to run over the final population
def genetic_algorithm_over_pop(objective, pop, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    start = time.time()
    
    """
    # initial random pop
    pop = [None] * n_pop
    for i in range(n_pop):
        pop[i] = randint(0, 2, n_bits*len(bounds)).tolist()
    """
    
    # keep track of best solutions
    best = 0
    best_eval = 500 #objective(decode(bounds, n_bits, pop[0]))
    
    for gen in range(n_iter):
		# decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]

        # evaluate all candidates
        scores = objective(decoded)
        
        # check new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best = pop[i]
                best_eval = scores[i]
                print(">%d, f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        
        # list of parents
        parents = [None] * n_pop
        for i in range(n_pop):
            parents[i] = selection(pop, scores)
        
        #create next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents
            parent_1 = parents[i]
            parent_2 = parents[i+1]
            #crossover and mutation
            for c in crossover(parent_1, parent_2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        
        # replace pop with the new set of offsprings
        pop = children
    end = time.time()
    print("Execution time:", (end-start)/60)
    return [best, best_eval, pop, decoded, scores]