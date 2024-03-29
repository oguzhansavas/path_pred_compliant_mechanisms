# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:15:01 2022

@author: ogsa991b
"""

# Importing of packages and functions used in this script which are saved in an additional file
import sys
sys.path.insert(0, r'C:\Users\ogsa991b\Documents\von_Lars\Path_prediction')
from _220302_data_creation_for_NN_ADDITIONAL_FUNCTIONS import *



########################### GETTING THE DATA ##################################

# Read the data
filename='220424_RAW_data_population_WITHOUT_COMMENT'
path_data=rf'C:\Users\ogsa991b\Documents\von_Lars\Path_prediction\{filename}'
list_chromosome, list_fitness, coords_RAW = read_RAW_data(path=path_data)


# Create the preprocessed curve points (coords_prepro)
#create the distributed points
number_of_points = 50 #too little and you loose information about the fea path
coords_prepro = get_distributed_points_of_desired_amount(coords_RAW=coords_RAW,
                                                                number_of_points=number_of_points)

#delete elements that contain erroneously values of nan (due to division by zero)
list_chromosome, list_fitness, coords_RAW, coords_prepro = delete_results_with_nan(list_chromosome=list_chromosome,
                                                                                   list_fitness=list_fitness,
                                                                                   coords_RAW=coords_RAW,
                                                                                   coords_prepro=coords_prepro)
#delete elements that erroneously don't have the desired number of points (don't know why this sometimes doesn't work)
list_chromosome, list_fitness, coords_RAW, coords_prepro = delete_results_of_false_length(list_chromosome=list_chromosome,
                                                                                   list_fitness=list_fitness,
                                                                                   coords_RAW=coords_RAW,
                                                                                   coords_prepro=coords_prepro,
                                                                                   number_of_points=number_of_points)
#reduced the chromosome to the first 8 entries (the coords of the mechnisms edge nodes)
list_chromosome = simplify_chromosome(list_chromosome=list_chromosome)


# Add new modified data
# Read modified data
filename_mod = "220602_RAW_data_population_WITHOUT_COMMENT"
path_data_mod = rf'C:\Users\ogsa991b\Documents\von_Lars\Path_prediction\{filename_mod}'
list_chromosome_mod, list_fitness_mod, coords_RAW_mod = read_RAW_data(path=path_data_mod)

# Create the preprocessed curve points (coords_prepro_mod)
#create the distributed points
coords_prepro_mod = get_distributed_points_of_desired_amount(coords_RAW=coords_RAW_mod,
                                                                number_of_points=number_of_points)

#delete elements that contain erroneously values of nan (due to division by zero)
list_chromosome_mod, list_fitness_mod, coords_RAW_mod, coords_prepro_mod = delete_results_with_nan(list_chromosome=list_chromosome_mod,
                                                                                   list_fitness=list_fitness_mod,
                                                                                   coords_RAW=coords_RAW_mod,
                                                                                   coords_prepro=coords_prepro_mod)
#delete elements that erroneously don't have the desired number of points (don't know why this sometimes doesn't work)
list_chromosome_mod, list_fitness_mod, coords_RAW_mod, coords_prepro_mod = delete_results_of_false_length(list_chromosome=list_chromosome_mod,
                                                                                   list_fitness=list_fitness_mod,
                                                                                   coords_RAW=coords_RAW_mod,
                                                                                   coords_prepro=coords_prepro_mod,
                                                                                   number_of_points=number_of_points)
#reduced the chromosome to the first 8 entries (the coords of the mechnisms edge nodes)
list_chromosome_mod = simplify_chromosome(list_chromosome=list_chromosome_mod)


# Extend the original data with new data
list_chrom_new = []
list_chrom_new.extend(list_chromosome)
list_chrom_new.extend(list_chromosome_mod)

coords_prepro_new = []
coords_prepro_new.extend(coords_prepro)
coords_prepro_new.extend(coords_prepro_mod)


# Visualize the RAW and preprocessed coordinates.
path_pics=r"C:\Users\ogsa991b\Documents\220303_pics_coords"
index = 1
compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                        coords_RAW_single    = coords_RAW[index],
                                        coords_prepro_single = coords_prepro[index],
                                        title                = str(index))



############################### BENCHMARKING ##################################

# What is the benchmark to be better than pure random predictions?
import random
coords_prepro_shuffled = coords_prepro_new.copy()
random.shuffle(coords_prepro_shuffled)

from sklearn.metrics import mean_squared_error
mse_benchmark = mean_squared_error(coords_prepro_new, coords_prepro_shuffled)
#me = mse_benchmark**0.5
print(f"Benchmark of MSE to beat: {mse_benchmark}")



############################## DATA PREP ######################################

# Get train-test data
x_train_test = list_chrom_new[100:]
x_validation = list_chrom_new[:100]
y_train_test = coords_prepro_new[100:]
y_validation = coords_prepro_new[:100] 
x_train, x_test, y_train, y_test = tts(x_train_test,y_train_test,test_size=0.2)



######################### BAYESIAN OPTIMIZATION ###############################

# Get the hypermodel for optimization
from bayes_opt_v2 import *

# Define tuner
tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_loss",
    max_trials=200,
    seed = 42,
    overwrite=True)

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=4)

# Start the search
tuner.search(x_train, y_train, epochs=30, validation_data=(x_test, y_test), callbacks=[early_stop])

# Inspect the search summmary (optional)
tuner.results_summary()

# Get the best model
best_model = tuner.get_best_models()[0]

# Build the model.
best_model.build()
best_model.summary()

# Save/Load the model
model_filename = "best_model.h5"
best_model.save(model_filename)

model = load_model(model_filename)



############################# PLOT PREDICTIONS ################################

# After getting the model, plot predicted, preprocessed and RAW paths
for i in range(10):
    index = random.randint(0,len(x_validation)) #random sample from the validation data
    #index = 2
#for index in [1,2,3,4,5]:
    path_pics=r"C:\Users\ogsa991b\Documents\220303_pics_coords"
    
    coords_prepro_single = y_validation[index]
    coords_ML_single     = model.predict([x_validation[index]]).tolist()[0]
    coords_RAW_single    = coords_RAW[coords_prepro.index(coords_prepro_single)]
    
    compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                            coords_RAW_single    = coords_RAW_single,
                                            coords_prepro_single = coords_prepro_single,
                                            coords_ML_single     = coords_ML_single,
                                            title                = str(index))



############################ GENETIC ALGORITHM ################################

# Get GA functions
from GA_mechanism_v2 import *
from run_mapdl import run_mapdl

# Define GA parameters
# range for input
bounds = [[0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0], [0.0, 1000.0]]
# define the total iterations
n_iter = 20
# bits per variable
n_bits = 10
# population size
n_pop = 100
# crossover rate
r_cross = 0.8
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))



############################### ML based GA ###################################
best_ml, score_ml, pop_bit, decoded_ml, scores_ml = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)

# Get the best result
best_decoded = decode(bounds, n_bits, best_ml)
print('f(%s) = %f' % (best_decoded, score_ml))

dict_best = {"coords"  : [best_decoded],
             "Fitness" : score_ml}

df_best = pd.DataFrame(dict_best)

# Create a dataframe from the last population and corresponding fitness values
data_dict = {"coords"  : decoded_ml,
             "Fitness" : scores_ml}

df_pop = pd.DataFrame(data_dict)

df_pop = df_pop.append(df_best)

final_pop_ml = df_pop.sort_values("Fitness", ascending=True).reset_index().drop("index", axis=1)
final_pop_ml.drop(axis=0, index=100, inplace=True)

# save/load
with open("final_pop_ml", "wb") as f:
    pickle.dump(final_pop_ml, f)

with open("final_pop_ml", "rb") as f:
    final_pop_ml = pickle.load(f)

with open("pop_bit", "wb") as f:
    pickle.dump(pop_bit, f)
    
with open("pop_bit", "rb") as f:
    pop_bit = pickle.load(f)
    
# Run ansys over the resulting pop to get true fitness values
mapdl_df, chromosomes = run_mapdl(final_pop_ml, mapdl)

# save/load
with open("mapdl_df", "wb") as f:
    pickle.dump(mapdl_df, f)

with open("chromosomes", "wb") as f:
    pickle.dump(chromosomes, f)

with open("mapdl_df", "rb") as f:
    mapdl_df = pickle.load(f)

with open("chromosomes", "rb") as f:
    chromosomes = pickle.load(f)



########################## FEA based GA #################################
best_fea, score_fea, pop_fea, decoded_fea, scores_fea = genetic_algorithm(objective_mapdl, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)

best_decoded_fea = decode(bounds, n_bits, best_fea)
print('f(%s) = %f' % (best_decoded_fea, score_fea))

dict_best_fea = {"coords"  : [best_decoded_fea],
                 "Fitness" : score_fea}

df_best_fea = pd.DataFrame(dict_best_fea)

# Create a dataframe from the last population and corresponding fitness values
data_dict_fea = {"coords"  : decoded_fea,
                   "Fitness" : scores_fea}

df_pop_fea = pd.DataFrame(data_dict_fea)

df_pop_fea = df_pop_fea.append(df_best_fea)

final_pop_fea = df_pop_fea.sort_values("Fitness", ascending=True).reset_index().drop("index", axis=1)
final_pop_fea.drop(axis=0, index=100, inplace=True)

# save
with open("final_pop_fea", "wb") as f:
    pickle.dump(final_pop_fea, f)

with open("pop_fea", "wb") as f:
    pickle.dump(pop_fea, f)



### Get the results, create df for comparison ###
from get_visual import sort_based_on_error

with open("final_pop_ml", "rb") as f:
    final_pop_ml = pickle.load(f)
    
with open("mapdl_df", "rb") as f:
    mapdl_df = pickle.load(f)


sorted_df_1 = sort_based_on_error(final_pop_ml, mapdl_df)

with open("sorted_df", "wb") as f:
    pickle.dump(sorted_df, f)

with open("sorted_df", "rb") as f:
    sorted_df = pickle.load(f)


#### Create visuals ####
from get_visual import get_ml_visual
from get_visual import get_mapdl_visual

# change the index to see the desired chromosome
get_ml_visual(sorted_df, 27, mapdl)

get_mapdl_visual(sorted_df, 79, mapdl)




##################### GA over the outcome pop (optional) #########################
# optional part to see if running the FEA-based GA over the final population of the ML-based GA makes a difference in terms of runtime.
# no difference, approximately same runtime

best_mapdl, score_mapdl, pop_bit_mapdl, decoded_mapdl, scores_mapdl = genetic_algorithm_over_pop(objective_mapdl, pop_bit, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)

best_decoded_mapdl = decode(bounds, n_bits, best_mapdl)
print('f(%s) = %f' % (best_decoded_mapdl, score_mapdl))

dict_best_mapdl = {"coords"  : [best_decoded_mapdl],
                   "Fitness" : score_mapdl}

df_best_mapdl = pd.DataFrame(dict_best_mapdl)

# Create a dataframe from the last population and corresponding fitness values
data_dict_mapdl = {"coords"  : decoded_mapdl,
                   "Fitness" : scores_mapdl}

df_pop_mapdl = pd.DataFrame(data_dict_mapdl)

df_pop_mapdl = df_pop_mapdl.append(df_best_mapdl)

final_pop_mapdl = df_pop_mapdl.sort_values("Fitness", ascending=True).reset_index().drop("index", axis=1)
final_pop_mapdl.drop(axis=0, index=100, inplace=True)

with open("final_pop_mapdl", "wb") as f:
    pickle.dump(final_pop_mapdl, f)

with open("final_pop_mapdl", "rb") as f:
    final_pop_mapdl = pickle.load(f)

with open("pop_bit", "wb") as f:
    pickle.dump(pop_bit, f)

