# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:05:20 2022

@author: Oguzhan Savas
"""
import numpy as np
import pandas as pd
import sklearn
import joblib
from numpy.random import randint
from numpy.random import rand
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# objective function
def objective(x):
    return model.predict([x])


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
    # initial random pop
    pop = [None] * n_pop
    for i in range(n_pop):
        pop[i] = randint(0, 2, n_bits*len(bounds)).tolist()
    
    # keep track of best solutions
    best = 0
    best_eval = objective(decode(bounds, n_bits, pop[0]))
    
    for gen in range(n_iter):
		# decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]

        # evaluate all candidates
        scores = []
        for i in decoded:
            score = objective(i)
            scores.append(score)
        
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
    return [best, best_eval, pop, decoded, scores]

#------------------------------------------------------------------------------

# range for input
bounds = [[0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0]]
# define the total iterations
n_iter = 20
# bits per variable
n_bits = 8
# population size
n_pop = 100
# crossover rate
r_cross = 0.8
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))

best, score, pop, decoded, scores = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)

best_decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (best_decoded, score))


coord = np.array(decoded)

data = {'x1': coord[:,0],
        'y1': coord[:,1],
        'x2': coord[:,2],
        'y2': coord[:,3],
        'x3': coord[:,4],
        'y3': coord[:,5],
        'x4': coord[:,6],
        'y4': coord[:,7],
        'Fitness': scores[:]
        }

df = pd.DataFrame(data)
