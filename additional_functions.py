# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:55:22 2022

@author: Oguzhan Savas
"""

from numpy.random import seed
seed(42)

# batch size inspection
def batch_size(model, x_train, y_train, x_test, y_test, x_validation, y_validation):
    best_acc_test = []
    best_acc_val = []
    batch_size = [32, 64, 96, 128] 
    
    for batch in batch_size:
        history = model.fit(x_train, y_train, epochs=25, batch_size=batch, verbose=0, validation_data=(x_test, y_test))
        test_loss, test_acc = model.evaluate(x_test, y_test)
        best_acc_test.append(test_acc)
        val_loss, val_acc = model.evaluate(x_validation, y_validation)
        best_acc_val.append(val_acc)
        
    batch_dict = {'batch_size':    batch_size,
                  'test_acc'  :    best_acc_test,
                  'val_acc'   :    best_acc_val}
    
    batch_results = pd.DataFrame(batch_dict)
    return batch_results


# Getting ML prediction coordinates
def get_ml_coords(model, list_chromosome, get_x_y_from_coords):
    x_ml_total = []
    y_ml_total = []
    
    for index in range(len(list_chromosome)):
        x_ml_ind, y_ml_ind = get_x_y_from_coords(model.predict([list_chromosome[index]]).tolist()[0])
        x_ml_total.append(x_ml_ind)
        y_ml_total.append(y_ml_ind)
    
    # define dict to create dataframe, pickle it to save
    ml_data_dict = {'x_total'  :    x_ml_total,
                    'y_total'  :    y_ml_total}

    ml_data = pd.DataFrame(ml_data_dict)
    ml_data.to_pickle("ml_data.pkl")
    
    return x_ml_total, y_ml_total, ml_data


# Getting corresponding fitness values to ml predictions
def get_fitness_values(x_ml_total, y_ml_total, ml_data, determine_fitness_ind):
    ml_fitness = []

    for index in range(len(x_ml_total)):
        target_coords, fea_coords, fea_coords_transformed, jk_D_min, phi, D, comment_tmp = determine_fitness_ind(x_ml_total[index],y_ml_total[index], False, func_line=func_line)
        ml_fitness.append(jk_D_min)
        
    # add ml_fitness to ml_data dataframe, pickle it
    ml_data["ml_fitness"] = ml_fitness
    ml_data.to_pickle("ml_data_with_fitness.pkl")
    
    return ml_fitness, ml_data


# Error inspection of predicted path coordinates - criteria:mse
def get_path_prediction_error(coords_prepro, x_ml_total, y_ml_total):
    mse_x = []
    mse_y = []
    x_prepro = []
    y_prepro = []
    
    for index in range(len(coords_prepro)):
        x_prepro_ind = coords_prepro[index][0:50]
        y_prepro_ind = coords_prepro[index][50:100]
        x_prepro.append(x_prepro_ind)
        y_prepro.append(y_prepro_ind)
        
    for index in range(len(x_prepro)):
        ind_mse_x = mean_squared_error(x_prepro[index], x_ml_total[index])
        mse_x.append(ind_mse_x)

    for index in range(len(y_prepro)):
        ind_mse_y = mean_squared_error(y_prepro[index], y_ml_total[index])
        mse_y.append(ind_mse_y)
        
    mse_dict = {"x_prepro"  :   x_prepro,
                "x_ml"      :   x_ml_total,
                "mse_x"     :   mse_x,
                "y_prepro"  :   y_prepro,
                "y_ml"      :   y_ml_total,
                "mse_y"     :   mse_y,
                "list_chromosome" : list_chromosome}

    mse_df = pd.DataFrame(mse_dict)
    
    return mse_df, x_prepro, y_prepro, mse_x, mse_y


# Get worst 5000 data
# compare the average of mse_x and mse_y for each row, sort them based on highest average error.
def get_highest_error(mse_df):
    mse_df["avg_error"] = (mse_df["mse_x"] + mse_df["mse_y"])/2

    mse_df_sorted = mse_df.sort_values("avg_error")
    mse_worst_5000 = mse_df_sorted[-5000:]

    mse_worst_5000.to_pickle("mse_worst_5000.pkl")
    
    return mse_df_sorted, mse_worst_5000

# get worst coordinates
def get_worst_ml_coords(mse_worst_5000):
    x_ml_worst = mse_worst_5000["x_ml"].tolist()
    y_ml_worst = mse_worst_5000["y_ml"].tolist()
    
    worst_ml_coords = []
    for index in range(len(x_ml_worst)):
        worst_ind = x_ml_worst[index] + y_ml_worst[index]
        worst_ml_coords.append(worst_ind)
    
    return x_ml_worst , y_ml_worst, worst_ml_coords

# get worst chromosomes to modify
def get_worst_chromosomes(mse_worst_5000):
    worst_chromosome = mse_worst_5000["list_chromosome"].tolist()
    worst_chrom_mod = worst_chromosome
    return worst_chrom_mod

# modify worst coords, correct this, it is not right!!!!
def modify(worst_chrom_mod):
    for i in range(len(worst_chrom_mod)):
        for j in range(8):
            coord = worst_chrom_mod[i][j]
            if coord < 10:
                coord = coord + random.randint(0, 10)
                worst_chrom_mod[i][j] = coord
            elif coord > 990:
                coord = coord + random.randint(-10, 0)
                worst_chrom_mod[i][j] = coord
            else:
                coord = coord + random.randint(-10, 10)
                worst_chrom_mod[i][j] = coord
    
    worst_chrom_dict = {"worst_chromosome_mod" : worst_chrom_mod}
    worst_chromosome_mod_df = pd.DataFrame(worst_chrom_dict)
    
    worst_chromosome_mod_df.to_pickle("worst_chromosomes_modified.pkl")
    
    return worst_chrom_mod, worst_chromosome_mod_df
