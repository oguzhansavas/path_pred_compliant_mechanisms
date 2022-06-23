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
index = 2
compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                        coords_RAW_single    = coords_RAW[index],
                                        coords_prepro_single = coords_prepro[index],
                                        title                = str(index))



############################### BENCHMARKING ##################################

# What is the benchmark to be better than pure random predictions?
import random
coords_prepro_shuffled = coords_prepro.copy()
random.shuffle(coords_prepro_shuffled)

from sklearn.metrics import mean_squared_error
mse_benchmark = mean_squared_error(coords_prepro, coords_prepro_shuffled)
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
    max_trials=2,
    seed = 42,
    overwrite=True)

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=5)

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

# Define path (function line)
function_of_curve_as_str = "math.sin(2*np.pi*x*0.02)*10+500"
#function_of_curve_as_str = "0.1*(x-500)**2+500"
#function_of_curve_as_str = "0.01*(x-500)**2+600"
xmin = 400
xmax = 500
func_line = target_line(function_of_curve_as_str,xmin,xmax)


# Get GA functions
from GA_mechanism_v2 import *

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


# Start GA
best, score, pop, decoded, scores = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)

# Get the best result
best_decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (best_decoded, score))

# Create a dataframe from the last population and corresponding fitness values
data_dict = {"coords"  : decoded,
             "Fitness" : scores}

df_pop = pd.DataFrame(data_dict)
final_pop = df_pop.sort_values("Fitness", ascending=True).reset_index().drop("index", axis=1)

# Chromosome for ansys
chromosome = best_decoded.copy()



########################### RUN ANSYS HERE ####################################

# After running ansys:
    
# best candidate
chromosome.extend([np.pi,0,0,0,0,0])
ind = Individual(mapdl, chromosome)
ind.make_plot()
ind.make_gif()
ind.fitness

# second best
chrom2 = final_pop["coords"][0].copy()
chrom2.extend([np.pi,0,0,0,0,0])
ind2 = Individual(mapdl, chrom2)
ind2.make_plot()
ind2.make_gif()
ind2.fitness

# third best
chrom3 = final_pop["coords"][1].copy()
chrom3.extend([np.pi,0,0,0,0,0])
ind3 = Individual(mapdl, chrom3)
ind3.make_plot()
ind3.make_gif()
ind3.fitness

# 4th best
chrom4 = final_pop["coords"][2].copy()
chrom4.extend([np.pi,0,0,0,0,0])
ind4 = Individual(mapdl, chrom4)
ind4.make_plot()
ind4.make_gif()
ind4.fitness

# 5th best
chrom5 = final_pop["coords"][3].copy()
chrom5.extend([np.pi,0,0,0,0,0])
ind5 = Individual(mapdl, chrom5)
ind5.make_plot()
ind5.make_gif()
ind5.fitness

pop_from_GA = decoded.copy()

with open("pop_from_GA", "wb") as f:
    pickle.dump(pop_from_GA, f)
    
with open("pop_from_GA", "rb") as f:
    pop_from_GA = pickle.load(f)