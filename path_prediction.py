# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:49:13 2022

@author: Lars Muschalski, Oguzhan Savas
"""

# Importing of packages and functions used in this script which are saved in an additional file
import sys
sys.path.insert(0, r'C:\Users\ogsa991b\Documents\von_Lars\Path_prediction')
from _220302_data_creation_for_NN_ADDITIONAL_FUNCTIONS import *


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


# Visualize the RAW and preprocessed coordinates.
path_pics=r"C:\Users\ogsa991b\Documents\220303_pics_coords"
index = 2
compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                        coords_RAW_single    = coords_RAW[index],
                                        coords_prepro_single = coords_prepro[index],
                                        title                = str(index))


# Benchmarking - What is the benchmark for our training to be better than pure random predictions?
import random
coords_prepro_shuffled = coords_prepro.copy()
random.shuffle(coords_prepro_shuffled)

from sklearn.metrics import mean_squared_error
mse_benchmark = mean_squared_error(coords_prepro, coords_prepro_shuffled)
#me = mse_benchmark**0.5
print(f"Benchmark of MSE to beat: {mse_benchmark}")



#-------------------------Deep Learning Part-----------------------------------

sys.path.insert(0, r'C:\Users\ogsa991b\Documents\Python_Scripts\von_Oguzhan\Deep_learning\new_version')
from bayes_opt import *
from additional_functions import *

# Get train-test data
x_train_test = list_chromosome[100:]
x_validation = list_chromosome[:100]
y_train_test = coords_prepro[100:]
y_validation = coords_prepro[:100] 
x_train, x_test, y_train, y_test = tts(x_train_test,y_train_test,test_size=0.2)

# Get hypermodel for Bayesian optimization 
#model = build_model(hp)

# Conduct Bayesian search
trials = 20
best_model = bayes_search(build_model, trials, x_train, y_train, x_test, y_test)

best_model.build()
best_model.summary()

# Save model
save_model(best_model)

# Fit the best model to the data
batch_size = 32
history = model_fit(best_model, batch_size, x_train, y_train, x_test, y_test)

print(np.average(history.history["val_loss"][-5:]))
print(np.average(history.history["loss"][-5:]))

plt.plot(history.history["loss"], "b", label="Train")
plt.plot(history.history["val_loss"], "r", label="Test")
plt.legend()

test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f"test_acc: {test_acc}")

test_loss, test_acc = best_model.evaluate(x_validation, y_validation)
print(f"test_acc: {test_acc}")



#-------------------------Batch Size Inspection--------------------------------

batch_results = batch_size(best_model, x_train, y_train, x_test, y_test, x_validation, y_validation)



#-------------Plotting the predicted, preprocessed and RAW paths---------------

for i in range(5):
    index = random.randint(0,len(x_validation)) #random sample from the validation data

    path_pics=r"C:\Users\ogsa991b\Documents\220303_pics_coords"
    
    coords_prepro_single = y_validation[index]
    coords_ML_single     = best_model.predict([x_validation[index]]).tolist()[0]
    coords_RAW_single    = coords_RAW[coords_prepro.index(coords_prepro_single)]
    
    compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                            coords_RAW_single    = coords_RAW_single,
                                            coords_prepro_single = coords_prepro_single,
                                            coords_ML_single     = coords_ML_single,
                                            title                = str(index))



#-----------------Fitness Calculation - for a single instance------------------

function_of_curve_as_str = "0.01*(x-500)**2+600"
xmin = 425
xmax = 500
func_line = target_line(function_of_curve_as_str,xmin,xmax)

# Data creation for plotting - change index for different curves
index = 201
x_RAW    , y_RAW       = get_x_y_from_coords(coords_RAW[index])
x_prepro , y_prepro    = get_x_y_from_coords(coords_prepro[index])
x_ML     , y_ML        = get_x_y_from_coords(model.predict([list_chromosome[index]]).tolist()[0])

#target_coords, fea_coords, fea_coords_transformed, jk_D_min, phi, D, comment_tmp = determine_fitness_ind(x_RAW,y_RAW, False, func_line=func_line)
target_coords, fea_coords, fea_coords_transformed, jk_D_min, phi, D, comment_tmp = determine_fitness_ind(x_prepro,y_prepro, False, func_line=func_line)
#target_coords, fea_coords, fea_coords_transformed, jk_D_min, phi, D, comment_tmp = determine_fitness_ind(x_ML,y_ML, False, func_line=func_line)
print(f"Fitness = {jk_D_min}")

# actual plotting
compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                        coords_RAW_single    = [_ for sublist in [x_RAW,y_RAW] for _ in sublist],
                                        coords_prepro_single = [_ for sublist in [x_prepro,y_prepro] for _ in sublist],
                                        coords_ML_single     = [_ for sublist in [x_ML,y_ML] for _ in sublist],
                                        target_coords        = target_coords,
                                        fea_coords           = fea_coords,
                                        title                = str(index))



#----------Getting ML coordinates, and corresponding fitness values------------

# Predicted path coordinates - takes some time
x_ml_total, y_ml_total, ml_data = get_ml_coords(best_model, list_chromosome, get_x_y_from_coords)

# after the first run, ml_data will be pickled, load the pickled ml_data instead of running it again
#ml_data = pd.read_pickle('ml_data_new.pkl')
#x_ml_total = ml_data["x_total"].tolist()
#y_ml_total = ml_data["y_total"].tolist()


# Fitness calculation for the whole ML data - also takes some time
ml_fitness, ml_data = get_fitness_values(x_ml_total, y_ml_total, ml_data, determine_fitness_ind)

# again, load pickle after the first run
#ml_data = pd.read_pickle('ml_data_with_fitness.pkl')
#ml_fitness = ml_data["ml_fitness"].tolist()



#---------------Error inspection - predicted path vs. real path----------------

# compare x_ml_total, y_ml_total to coords_prepro, mse is the criteria
mse_df, x_prepro, y_prepro, mse_x, mse_y = get_path_prediction_error(coords_prepro, x_ml_total, y_ml_total)



#----------------Getting the worst 5000 data for improvement-------------------
# sort the data according to their mse values, get the last 5000.
# create a df by getting the corresponding fitness values also (do it later!!)

# get the average of mse_x and mse_y for each row, sort them.
mse_df_sorted, mse_worst_5000 = get_highest_error(mse_df)

# get worst ml coordinates
x_ml_worst , y_ml_worst, worst_ml_coords = get_worst_ml_coords(mse_worst_5000)

# modify worst chromosomes
worst_chrom_mod, worst_chromosome_mod_df = modify(worst_chrom_mod)


#-------------------------Plotting the worst data------------------------------

# Plotting of the worst predicted paths, preprocessed and RAW paths
for i in range(5):
    index = random.randint(0,len(x_validation)) #random sample from the validation data
    #index = 2
#for index in [1,2,3,4,5]:
    path_pics=r"C:\Users\ogsa991b\Documents\220303_pics_coords"
    
    coords_prepro_single = y_validation[index]
    coords_ML_single     = worst_ml_coords[index] #x_ml_worst[index] + y_ml_worst[index]
    coords_RAW_single    = coords_RAW[coords_prepro.index(coords_prepro_single)]
    
    compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
                                            coords_RAW_single    = coords_RAW_single,
                                            coords_prepro_single = coords_prepro_single,
                                            coords_ML_single     = coords_ML_single,
                                            title                = str(index))
