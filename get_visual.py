# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:29:51 2022

@author: ogsa991b
"""

import numpy as np

def get_ml_visual(df_sorted, index, mapdl):
    
    global Individual
    
    chrom = df_sorted["ML Coords"][index].copy()
    chrom.extend([np.pi,0,0,0,0,0])
    ind = Individual(mapdl, chrom)
    ind.make_plot()
    ind.make_gif()


def get_mapdl_visual(df_sorted, index, mapdl):
    
    global Individual
    
    chrom = df_sorted["MAPDL Coords"][index].copy()
    chrom.extend([np.pi,0,0,0,0,0])
    ind = Individual(mapdl, chrom)
    ind.make_plot()
    ind.make_gif()


def sort_based_on_error(df_ml, df_mapdl):
    
    fitness_ml = df_ml["Fitness"]
    coords_ml = df_ml["coords"]
    
    fitness_mapdl = df_mapdl["Fitness"]
    coords_mapdl = df_mapdl["Chromosomes"]
    error = np.absolute(fitness_mapdl - fitness_ml)
    
    error_dict = {"ML Coords"     : coords_ml,
                  "ML Fitness"    : fitness_ml,
                  "MAPDL Coords"  : coords_mapdl,
                  "MAPDL Fitness" : fitness_mapdl,
                  "error"         : error}
    
    sorted_df = pd.DataFrame(error_dict).sort_values("MAPDL Fitness", ascending=True)
    
    return sorted_df
