# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:29:35 2022

@author: s8913814
"""


# In[37]:
#Libs


from tensorflow.keras import Input, Model,layers
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import concatenate # functional API

from sklearn.model_selection import train_test_split as tts
from sklearn import metrics

from skopt import Optimizer
from skopt import gp_minimize
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image   

import pickle

import os

import datetime

from itertools import compress

import math




# In[37]:
#saving, reading, preprocessing, plotting...


def reduce_pop_to_those_with_max_rotation(pop):
    #written on 28.02.22
    pop_reduced=[]
    for i in pop:
        if not any(["converged" in _ for _ in i.comment]) and hasattr(i, 'array_joint2_ux_RAW'):
            pop_reduced.append(i)
    return pop_reduced

def save_RAW_data_of_given_population(pop_reduced:list, path:str):
    list_chromosome = []
    list_fitness = []
    coords_RAW = []
    for ind in pop_reduced:
        list_chromosome.append(ind.chromosome)
        list_fitness.append(ind.fitness)
        _=[]
        x_tmp = ind.array_joint2_ux_RAW.copy()
        x_tmp = np.append(x_tmp,0)
        x_tmp += ind.joint2_start_x
        _.extend(x_tmp)
        y_tmp = ind.array_joint2_uy_RAW.copy()
        y_tmp = np.append(y_tmp,0)
        y_tmp += ind.joint2_start_y
        _.extend(y_tmp)
        coords_RAW.append(_)
    #path=r'C:\LokaleDaten\s8913814\Documents\Python_experimental\Data\TEST_DELETE_ME'):
    with open(rf'{path}',"wb") as f:
        pickle.dump(list_chromosome, f)
        pickle.dump(list_fitness, f)
        pickle.dump(coords_RAW, f)

"""
def read_RAW_data(path:str):
    with open(rf'{path}',"rb") as f:
        list_chromosome = pickle.load(f)
        list_fitness    = pickle.load(f)
        coords_RAW      = pickle.load(f)
    return list_chromosome, list_fitness, coords_RAW
"""

def read_RAW_data(path:str, with_comment=False):
    list_comment = []
    with open(rf'{path}',"rb") as f:
        list_chromosome = pickle.load(f)
        list_fitness = pickle.load(f)
        coords_RAW = pickle.load(f)
        if with_comment:
            return list_chromosome, list_fitness, coords_RAW, list_comment
        else:
            return list_chromosome, list_fitness, coords_RAW

def merge_saved_RAW_data(*args,
                         target_path=str):
    list_chromosome = []
    list_fitness    = []
    coords_RAW      = []
    for arg in args:
        _0,_1,_2=read_RAW_data(path=arg)
        list_chromosome.extend(_0)
        list_fitness.extend(_1)
        coords_RAW.extend(_2)
    with open(rf'{target_path}',"wb") as f:
        pickle.dump(list_chromosome, f)
        pickle.dump(list_fitness, f)
        pickle.dump(coords_RAW, f)
            



############### PreProcessing of DATA ###############

def get_length_curve(list_fea_x, list_fea_y):
    length_curve = 0
    for i in range(len(list_fea_x)-1):
        length_curve+=((list_fea_x[i]-list_fea_x[i+1])**2+(list_fea_y[i]-list_fea_y[i+1])**2)**0.5
    return length_curve

def get_distributed_points_of_desired_amount(coords_RAW:list,
                                             number_of_points=50):
    list_even_fea_coords = []
    for coords_RAW_single in coords_RAW:
        x = coords_RAW_single[:int(len(coords_RAW_single)/2)]
        y = coords_RAW_single[int(len(coords_RAW_single)/2):]
        length_curve = get_length_curve(x,y)
        _ = evenly_distribute_fea_coords(x,y,length_curve,number_of_points=number_of_points) #do the distribution
        _ = [item for sublist in list(zip(*_)) for item in sublist]     #flatten the list
        list_even_fea_coords.append(_)
    return list_even_fea_coords

def delete_results_with_nan(list_chromosome:list,
                            list_fitness:list,
                            coords_RAW:list,
                            coords_prepro:list):
    list_bool = [False if np.isnan(_).any() else True for _ in coords_prepro]
    list_chromosome = list(compress(list_chromosome,list_bool))
    list_fitness = list(compress(list_fitness,list_bool))
    coords_RAW = list(compress(coords_RAW,list_bool))
    coords_prepro = list(compress(coords_prepro,list_bool))
    return list_chromosome, list_fitness, coords_RAW, coords_prepro

def delete_results_of_false_length(list_chromosome:list,
                                   list_fitness:list,
                                   coords_RAW:list,
                                   coords_prepro:list,
                                   number_of_points:int):
    list_bool = [True if len(_)==number_of_points*2 else False for _ in coords_prepro]
    list_chromosome = list(compress(list_chromosome,list_bool))
    list_fitness = list(compress(list_fitness,list_bool))
    coords_RAW = list(compress(coords_RAW,list_bool))
    coords_prepro = list(compress(coords_prepro,list_bool))
    return list_chromosome, list_fitness, coords_RAW, coords_prepro
    

def simplify_chromosome(list_chromosome:list):
    return [_[:8] for _ in list_chromosome]



def get_x_y_from_coords(coords:list):
    l = int(len(coords)/2)
    return coords[:l], coords[l:]


############### Plotting of results ###############


def compare_RAW_preprocessed_coords_in_plot(path:str,**kwargs):
    font = {'weight' : 'bold',
            'size'   : 12}
    matplotlib.rc('font', **font)
    
    fig = Figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)
  
    if "coords_RAW_single" in kwargs:
        l=int(len(kwargs["coords_RAW_single"])/2)
        x_RAW,y_RAW = kwargs["coords_RAW_single"][:l],kwargs["coords_RAW_single"][l:]  #list(zip(*list_preprocessed_fea_coords[index]))
        ax.plot(x_RAW,y_RAW,color='red',linestyle='None',marker='x',markersize=10,linewidth=3,alpha=0.5,label="RAW")        

    if "coords_prepro_single" in kwargs:
        l=int(len(kwargs["coords_prepro_single"])/2)
        x_RAW,y_RAW = kwargs["coords_prepro_single"][:l],kwargs["coords_prepro_single"][l:]  #list(zip(*list_preprocessed_fea_coords[index]))
        ax.plot(x_RAW,y_RAW,color='green',linestyle='None',marker='o',markersize=5,linewidth=3,alpha=0.5,label="prepro")        

    if "coords_ML_single" in kwargs:
        l=int(len(kwargs["coords_ML_single"])/2)
        x_RAW,y_RAW = kwargs["coords_ML_single"][:l],kwargs["coords_ML_single"][l:]  #list(zip(*list_preprocessed_fea_coords[index]))
        ax.plot(x_RAW,y_RAW,color='blue',linestyle='None',marker='o',markersize=5,linewidth=3,alpha=0.5,label="ML")        

    if "target_coords" in kwargs:
        ax.plot(kwargs["target_coords"][0],kwargs["target_coords"][1],color='black',linestyle='None',marker='.',markersize=5,linewidth=3,alpha=0.5,label="target coords")        

    if "fea_coords" in kwargs:
        ax.plot(kwargs["fea_coords"][0],kwargs["fea_coords"][1],color='black',linestyle='None',marker='.',markersize=5,linewidth=3,alpha=0.5,label="fea coords")        

    if "title" in kwargs:
        ax.set_title("Index: "+kwargs["title"])

    ax.axis('equal')      
    ax.legend()
    #fig.patch.set_facecolor('white')
    path = rf"{path}\{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{kwargs['title']}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    #plt.pyplot.savefig(rf"C:\LokaleDaten\s8913814\Documents\GA_mechanism\{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{index}.png")                                                                            
    img = Image.open(path)
    img.show()



# In[37]:
#ML functions


def build_model(neurons_to_skip: list = None, neurons_all: list = None, dropout: float = 0., skip_con: bool = False, number_of_points: int = 100) -> keras.Model:
    """Build up neuronal network.

    Args:
        neurons_to_skip (list, optional): List with number of neurons per layer, which are bypassed if ``skip_con`` is set to ``True``. Defaults to [10].
        neurons_all (list, optional): List with number of neurons per layer, which will not be bypassed. Defaults to [].
        dropout (float, optional): Ratio of inputs to be set to 0. in dropout-layers. Has to be in the open interval [0., 1.). If ``dropout`` is not 0., dropout layers will be added after ``len(neurons_to_skip)`` layers and right before output layer. Defaults to 0..
        skip_con (bool, optional): Add skip connection, bypassing ``neurons_to_skip`` and concatenating afterwards. Defaults to False.

    Returns:
        keras.Model: Neuronal network according to above specifications.
    """

        
    initializer = keras.initializers.GlorotNormal()

    neurons_to_skip = neurons_to_skip if neurons_to_skip is not None else [10]
    neurons_all = neurons_all if neurons_all is not None else []
    
    # print(neurons_to_skip, neurons_all)

    input = Input(shape=(8,), name="design_parameters")
    for i, neurons in enumerate(neurons_to_skip):
        if i == 0:
            x = layers.Dense(neurons,activation="relu", kernel_initializer=initializer)(input)
        else:
            x = layers.Dense(neurons,activation="relu", kernel_initializer=initializer)(x)
    
    if dropout:
        x = layers.Dropout(dropout)(x)

    if skip_con:
        x = concatenate([input, x])
    
    for neurons in neurons_all:
        x = layers.Dense(neurons,activation="relu", kernel_initializer=initializer)(x)
    
    if dropout:
        x = layers.Dropout(dropout)(x)
    
    output = layers.Dense(number_of_points*2, name="Coordinates_fea_path", activation="linear", kernel_initializer=initializer)(x)
    model = Model(input, output, name="mechanism_prediction_of_fea_path")

    return model

def train_model(model: keras.Model, epochs: int, x_train, y_train, x_test, y_test) -> keras.callbacks.History:
    """Train passed in model for ``epochs`` epochs.

    Args:
        model (keras.Model): Uncomplied model to be trainedj
        epochs (int): Number of epochs to train for

    Returns:
        keras.callbacks.History: ``history`` object containing details of training.
    """
    model.compile(loss="mse", optimizer='adam',metrics=["mse"])
    """
    global x_train
    global y_train
    global x_test
    global y_test
    """
    #logdir = os.path.join("C:\LokaleDaten\s8913814\Documents\Python_experimental\logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test))
        #callbacks=[tensorboard_callback]    )
    return history



# In[37]:
### functions from "useful_functions_mix_v15.py"



def dffilter(df_input,COLUMNNAME,CHAR,VALUE):			#filtert df nach Werten in einer Spalte
	df_output=[]
	try:
		if CHAR=='<':df_output=df_input[df_input[COLUMNNAME] < VALUE]
		if CHAR=='<=':df_output=df_input[df_input[COLUMNNAME] <= VALUE]
		if CHAR=='>':df_output=df_input[df_input[COLUMNNAME] > VALUE]
		if CHAR=='>=':df_output=df_input[df_input[COLUMNNAME] >= VALUE]
		if CHAR=='==':df_output=df_input[df_input[COLUMNNAME] == VALUE]
		if CHAR=='!=':df_output=df_input[df_input[COLUMNNAME] != VALUE]
		return df_output
	except:
		print('Eingabe unbekannt oder nicht im df hinterlegt')
        

def evenly_distribute_fea_coords(list_fea_x,list_fea_y,length_fea,number_of_points="Default",plot=False):
    """
    Distribute the coords evenly, so there are accumulation of points on the fea line which may give a poor fea path a good fitness
    """
    #raise Exception("evenly_distribute_fea_coords() not yet finished???")

    #list_fea_x = [0,1,3,3.5,3.7,5,8,8.1,10] + [9 , 6 , 4]
    #list_fea_y = [y**2/2 for y in list_fea_x] 
    #list_fea_y = [0.0, 0.5, 4.5, 6.125, 6.845000000000001, 12.5, 32.0, 32.805, 50.0, 50, 45, 40]
    list_fea_coords= list(zip(list_fea_x,list_fea_y))
    
    #list_fea_coords=[(x, y**2) for x, y in list_fea_coords]
    """
    length_fea = 0
    for i in range(len(list_fea_coords)-1):
        length_fea += ((list_fea_coords[i+1][0]-list_fea_coords[i][0])**2+(list_fea_coords[i+1][1]-list_fea_coords[i][1])**2)**0.5
    """
    if number_of_points == "Default":
        number_of_points=len(list_fea_x)
    #number_of_divisions = 10
    
    length_step_ideal = length_fea/(number_of_points-1) 
    
    list_fea_coords_even=[]
    list_fea_coords_even.append(list_fea_coords[0])
    
    greatest_index = 1
    #loop start
    #for i in range(1,len(list_fea_coords)):
    for i in range(1,number_of_points):
        distance_to_point=[]
        #if i ==4: break
        for ii in range(greatest_index,len(list_fea_coords)):
            distance_to_point.append(((((list_fea_coords[ii][0]-list_fea_coords_even[-1][0])**2+(list_fea_coords[ii][1]-list_fea_coords_even[-1][1])**2)**0.5-length_step_ideal),ii))
            if distance_to_point[-1][0]>=length_step_ideal:
                break
        
        #delete negative elements in list
        #distance_to_point = [entry for entry in distance_to_point if entry[0]>=0]        
        delta_distance_to_point_sorted=sorted(distance_to_point, key=lambda tup: abs(tup[0]))
        greatest_index = delta_distance_to_point_sorted[0][1]
        #https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
        x1 = list_fea_coords_even[-1][0]
        y1 = list_fea_coords_even[-1][1]
        x2 = list_fea_coords[delta_distance_to_point_sorted[0][1]][0]
        y2 = list_fea_coords[delta_distance_to_point_sorted[0][1]][1]   
        v1=x2-x1
        v2=y2-y1
        xn=x1+v1/(v1**2+v2**2)**0.5*length_step_ideal
        yn=y1+v2/(v1**2+v2**2)**0.5*length_step_ideal
        list_fea_coords_even.append((xn,yn))
        #wenn der fea punkt mit dem greatest_index zwischen dem letzen und dem vorletzten liegt, dann muss der greatest_index um einen erhöht werden
        if ((list_fea_coords[greatest_index][0]-list_fea_coords_even[-2][0])**2+(list_fea_coords[greatest_index][1]-list_fea_coords_even[-2][1])**2)**0.5 < length_step_ideal:
            greatest_index += 1
            if greatest_index >= len(list_fea_coords):
                break
        #print(xn,yn)
        #list_fea_x_even[i+1]=xn
        #list_fea_y_even[i+1]=yn    
    
    #list_fea_coords_even.append(list_fea_coords[-1])
    
    
    if plot:
        fig = Figure(figsize=(20, 20), dpi=300)
        ax = fig.add_subplot(111)
        #ax.set_ylim([0 , 100])
        ax.axis('equal')
        ax.plot(list(zip(*list_fea_coords))[0],list(zip(*list_fea_coords))[1],color='red',linestyle='None',marker='x',markersize=5,linewidth=3,label="FEA")
        ax.plot(list(zip(*list_fea_coords_even))[0],list(zip(*list_fea_coords_even))[1],color='green',linestyle='None',marker='o',markersize=5,linewidth=3,alpha=0.5,label="even")
        #ax.plot(list_fea_x_even, list_fea_y_even,color='green',linestyle='None',marker='x',markersize=5,linewidth=3)
        ax.legend()
        fig.savefig(r"C:\LokaleDaten\s8913814\Documents\GA_mechanism\test.png")
    
    return list_fea_coords_even



def determine_fitness_ind(list_fea_x, list_fea_y, bool_plot, func_line):
    """
    list_target_x sind die x-Koordinaten des zu erreichenden Pfades
    list_fea_x sind die x-Koordinaten die von der fem angefahren werden
    """
    #print("determine_fitness_ind(...) START")
    """
    if hasattr(kwargs["func_line"]):
        func_line = kwargs["func_line"]
    else:
        try:
            global func_line
        except:
            pass
    """
    length_target = func_line.length
    list_fea_x = list_fea_x[::-1]
    list_fea_y = list_fea_y[::-1]
    """
    bool_plot = True
    list_fea_x = test_ind2.array_joint2_ux+test_ind2.chromosome[2]
    list_fea_y = test_ind2.array_joint2_uy+test_ind2.chromosome[3]
    """
    """
    list_fea_x = [100,200,210,300,500,550,700,800] #Testwerte
    list_fea_y = [100,100,100,100,100,100,100,100] #Testwerte
    """
    """
    #Sinus Testwerte
    anz_messwerte=100
    size_x_max = 500
    list_fea_x = []
    #sinus funktion "a*sin(2*pi/(size_x))
    for i in range(anz_messwerte):
        list_fea_x.append(size_x_max*random.random()+300)
    list_fea_x.sort(key=lambda x:x, reverse = False)
    list_fea_y = []
    for x in list_fea_x:    
        list_fea_y.append(math.sin(1.1*2*np.pi/max(list_fea_x)*x)*200+500)
    #list_fea_y = [x * 0.5 for x in list_fea_y]
    """
    
    """
    list_target_x = [1,2,3,4,5] #Testwerte
    list_target_y = [4,5,6,7,8] #Testwerte    
    length_target = 0
    for i in range(len(list_target_x)-1):
        length_segment = ((list_target_x[i+1]-list_target_x[i])**2+(list_target_y[i+1]-list_target_y[i])**2)**0.5
        length_target += length_segment
    """


    #zuerst die richtige Anzahl an Punkten der FEM bestimmen, die zusammen die gleiche Länge wie length_target haben
    length_fea_array = [] # first value is the index of list_fea_x and the second the last index of the line, that forms a line equal to the target length
    df=pd.DataFrame([], columns=['case','target_coords','fea_coords','fea_length','fea_coords_transformed','distance_target_fea','jk_D_min','phi','D'])
    
    for i in range(len(list_fea_x)-1):
        length_segment = 0
        for ii in np.linspace(i,len(list_fea_x)-2,len(list_fea_x)-i-1):
            ii=int(ii)
            #print(ii)
            length_segment += ((list_fea_x[ii+1]-list_fea_x[ii])**2+(list_fea_y[ii+1]-list_fea_y[ii])**2)**0.5
            if ii == len(list_fea_x)-2:
                length_fea_array.append([i,ii+1,length_segment])#,abs(length_segment-length_target)])
            elif abs(length_target - length_segment) <= abs(length_target - (length_segment + ((list_fea_x[ii+2]-list_fea_x[ii+1])**2+(list_fea_y[ii+2]-list_fea_y[ii+1])**2)**0.5)):
                #dann ist das jetzige segment besser, als das nächste
                length_fea_array.append([i,ii+1,length_segment])#,abs(length_segment-length_target)])
                break
    
    #reduce to all rows of length_fea_array with a length which is between 0.9-1.1*length_target
    length_fea_array = [a for a in length_fea_array if a[2] >= 0.9*length_target and a[2] <= 1.1*length_target]
    if length_fea_array == []:
        return 999999, 999999, 999999, 999999, 999999, 999999,'WARNING: determine_fitness_ind() did not work, there is no segment with similar length to func_line'
        #raise ValueError("WARNING: determine_fitness_ind() did not work, there is no segment with similar length to func_line")    
        
    #reduce to all rows of length_fea_array with at least 5 points
    length_fea_array = [a for a in length_fea_array if a[1]-a[0] >= 4]
    if length_fea_array == []:
        return 999999, 999999, 999999, 999999, 999999, 999999,'WARNING: determine_fitness_ind() did not work, there is no segment with at least 5 points and similar length to func_line'        
        #raise ValueError("WARNING: determine_fitness_ind() did not work, there is no segment with at least 5 points and similar length to func_line")
        
    length_fea_array=sorted(length_fea_array,key=lambda x:(abs(x[2]-length_target)),reverse=False)
    

    for case in range(len(length_fea_array)):
        #print(case)
        #die laenge der jeweiligen segmente der fea linie berechnen
        length_segments_fea = []
        for i in np.linspace(length_fea_array[case][0],length_fea_array[case][1],length_fea_array[case][1]-length_fea_array[case][0]+1):
            i=int(i)
            if i == length_fea_array[case][1]:
                break
            length_segments_fea.append(((list_fea_x[i]-list_fea_x[i+1])**2+(list_fea_y[i]-list_fea_y[i+1])**2)**0.5)
        
        #Punkte auf der Target Linie finden, die die gleiche Laenge haben, wie die segmente der fea linie
        
        
        
        #unterteilen der Target linie in die gleiche Anzahl wie die Linie der FEM
        fea_coords = []
        fea_coords.append(list_fea_x[length_fea_array[case][0]:length_fea_array[case][1]+1])
        fea_coords.append(list_fea_y[length_fea_array[case][0]:length_fea_array[case][1]+1])
        #n_div = len(fea_coords[0])-1
        #func_line.divide_into_n_parts(n_div)
        func_line.divide_into_n_parts_of_variable_length(length_segments_fea)
        
        #hier unterteilen der Target-line in n gleiche Teile
        target_coords = []
        target_coords.append([i[0] for i in func_line.array_coord])
        target_coords.append([i[1] for i in func_line.array_coord])
        """
        #ALTERNATIV Z.B. IN 100 TEILE UNTERTEILEN UND den nähesten punkt suchen
        #ist deutlich besser, wenn die Punkte der fea unregelmäßig verteilt sind
        x_array = np.linspace(func_line.x_min,func_line.x_max,100)
        y_array = []
        for i in x_array:
            y_array.append(func_line.calc_y(i))
        target_coords_all = []
        target_coords_all.append(x_array)
        target_coords_all.append(y_array)
        target_coords = []
        arr_t = np.array(target_coords_test).T  #Transponieren 
        for i in range(len(fea_coords[0])):
            target_coords.append(min(arr_t, key=lambda x:((x[0]-fea_coords[0][i])**2+(x[1]-fea_coords[1][i])**2)**0.5))
        """
        #besser nicht in gleiche Teile unterteilen, sondern in Teile die die gleiche Laenge haben wie die jweiligen Abschnitte der fea segmente haben
        #target_coords = []
        
        
        
        #suchen nach dem optimalen winkel und optimalem offset für die transformation von fea_coords
        #nach zhao Gl.8-9
        jk_x_delta_F = sum(fea_coords[0])
        jk_y_delta_F = sum(fea_coords[1])
        jk_x_delta_T = sum(target_coords[0])
        jk_y_delta_T = sum(target_coords[1])
        k = len(target_coords[0])
        j = 1
        sigma1 = 0
        for i in range(len(fea_coords[0])):
            sigma1 += target_coords[0][i]*fea_coords[1][i] - fea_coords[0][i]*target_coords[1][i]
        sigma2 = 0
        for i in range(len(fea_coords[0])):
            sigma2 += fea_coords[0][i]*target_coords[0][i] + fea_coords[1][i]*target_coords[1][i]
        """    
        phi = math.atan(
        ((jk_x_delta_T*jk_y_delta_F - jk_x_delta_F*jk_y_delta_T)/(k-j+1) - sigma1)
        /
        (sigma2-(jk_x_delta_F*jk_x_delta_T+jk_y_delta_F*jk_y_delta_T)/(k-j+1)))
        """
        phi = np.arctan2(((jk_x_delta_T*jk_y_delta_F - jk_x_delta_F*jk_y_delta_T)/(k-j+1) - sigma1), (sigma2-(jk_x_delta_F*jk_x_delta_T+jk_y_delta_F*jk_y_delta_T)/(k-j+1)))
        
        #atan liefert zwei Loesungen, deshalb fuer beide das jk_D_min berechnen und den Winkel phi ausgeben, für den jk_D_min kleiner ist

        target_coords_reverse = [target_coords[0][::-1],target_coords[1][::-1]]
        
        df_sub=pd.DataFrame([], columns=['case','target_coords','fea_coords_transformed','distance_target_fea','jk_D_min','phi','D'])
        iiii = 1
        for i_phi in [0, phi, phi+np.pi]:
            #i_phi = 0 dient dazu, die fitness des ind ohne transformation zu bestimmen
            fea_coords_transformed = [[],[]]
            if i_phi == 0:
                dx_tmp = 0
                dy_tmp = 0
            else:
                dx_tmp = ((jk_x_delta_T-(math.cos(i_phi)*jk_x_delta_F-math.sin(i_phi)*jk_y_delta_F))/(k-j+1))
                dy_tmp = ((jk_y_delta_T-(math.sin(i_phi)*jk_x_delta_F+math.cos(i_phi)*jk_y_delta_F))/(k-j+1))                        
            
            for iii in range(len(fea_coords[0])):
                x_neu = math.cos(i_phi) * fea_coords[0][iii]    -  math.sin(i_phi) * fea_coords[1][iii] + dx_tmp
                y_neu = math.sin(i_phi) * fea_coords[0][iii]    +  math.cos(i_phi) * fea_coords[1][iii] + dy_tmp
                #fea_coords_transformed[0][i] = x_neu
                #fea_coords_transformed[1][i] = y_neu
                fea_coords_transformed[0].append(x_neu)
                fea_coords_transformed[1].append(y_neu)            

            for i_target_coords in [target_coords, target_coords_reverse]:
                df_sub.loc[iiii,'case'] = case
                df_sub.loc[iiii,'phi'] = i_phi                                
                df_sub.loc[iiii,'D'] = [dx_tmp, dy_tmp]                
                df_sub.loc[iiii,'fea_coords_transformed'] = fea_coords_transformed                

                #abweichung von target zu fea, quasi fitness, nach Zhao Gl.5
                df_sub.loc[iiii,'target_coords'] = i_target_coords
                tmp_distance_target_fea = 0
                for ii in range(len(target_coords[0])):
                    tmp_distance_target_fea += (fea_coords_transformed[0][ii]-i_target_coords[0][ii])**2+(fea_coords_transformed[1][ii]-i_target_coords[1][ii])**2
                df_sub.loc[iiii,'distance_target_fea'] = tmp_distance_target_fea
            
           
                #average error jk_D_min nach Zhao Gl.9
                sigma3 = 0
                for iii in range(len(fea_coords[0])):
                    sigma3 += ((fea_coords_transformed[0][iii]-i_target_coords[0][iii])**2+(fea_coords_transformed[1][iii]-i_target_coords[1][iii])**2)**0.5
                jk_D_min_tmp = (1/(k-j+1)*sigma3)
                df_sub.loc[iiii,'jk_D_min'] = jk_D_min_tmp              
                iiii += 1

        df_sub = df_sub.sort_values(by=['distance_target_fea'])
        df_sub = df_sub.reset_index(drop=True)    
        
        df.loc[case,'case'] = df_sub.loc[0,'case']
        df.loc[case,'target_coords'] = df_sub.loc[0,'target_coords']
        df.loc[case,'fea_coords'] = fea_coords
        df.loc[case,'fea_length'] = length_fea_array[case][2]
        df.loc[case,'fea_coords_transformed'] = df_sub.loc[0,'fea_coords_transformed']
        #df.loc[case,'distance_target_fea'] = df_sub.loc[0,'distance_target_fea']
        #df.loc[case,'jk_D_min'] = df_sub.loc[0,'jk_D_min']
        df.loc[case,'distance_target_fea'] = dffilter(df_sub,'phi','==',0).sort_values(by=['distance_target_fea']).reset_index(drop=True).loc[0,'distance_target_fea']
        df.loc[case,'jk_D_min'] = dffilter(df_sub,'phi','==',0).sort_values(by=['distance_target_fea']).reset_index(drop=True).loc[0,'jk_D_min']        
        df.loc[case,'phi'] = df_sub.loc[0,'phi']
        df.loc[case,'D'] = df_sub.loc[0,'D']
        
    df = df.sort_values(by=['distance_target_fea'])
    df = df.reset_index(drop=True)    
    #make_control_plot(target_coords, list_fea_x, list_fea_y, fea_coords, fea_coords_transformed)
    if bool_plot == True:
        make_control_plot(df.loc[0,'target_coords'], list_fea_x, list_fea_y, df.loc[0,'fea_coords'], df.loc[0,'fea_coords_transformed'])
    
    #print("determine_fitness_ind(...) FINISH")
    return df.loc[0,'target_coords'], df.loc[0,'fea_coords'], df.loc[0,'fea_coords_transformed'], df.loc[0,'jk_D_min'], df.loc[0,'phi'], df.loc[0,'D'],''







# In[37]:
### func_line class from "class_line_v15.py"


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:12:16 2021

@author: Lars
"""



class target_line(object):
    '''
    Class representing the function of the target curve
    str_of_function = "x**2"
    '''
    def __init__(self, str_of_function,x_min,x_max):
        #str_of_function = "x**2"
        self.str_of_function = str_of_function
        self.x_min = x_min
        self.x_max = x_max
        #print(str_of_function)
        self.calc_length()
        self.divide_into_n_parts(1000)
        self.all_coord_x = [i[0] for i in self.array_coord]
        self.all_coord_y = [i[1] for i in self.array_coord]
        
    #@classmethod
    def calc_y(self,x_value):
        '''
        calculate the y value of a given x value
        '''
        x=x_value
        y_value = eval(self.str_of_function)
        return y_value        
        
    #@classmethod
    def calc_length(self):
        n_div = 1000
        dx = (self.x_max-self.x_min)/n_div
        self.length = 0
        for i in range(n_div-1):
            self.length += ((dx*i-dx*(i+1))**2+(self.calc_y(self.x_min+i*dx)-self.calc_y(self.x_min+(i+1)*dx))**2)**0.5
    
    def line2string(self):
        #gibt die func_line als str wieder um diese bei save_para2file() zu nutzen
        return '"'+self.str_of_function+'",'+str(self.x_min)+','+str(self.x_max)        
    
    def divide_into_n_parts(self,n_div):
        """
        #Teilt die Funktion in n gleiche Teile und gibt die Koordinaten wieder
        """
        target_segment_length = self.length / n_div
        dx = target_segment_length / 100
        array_coord = [[self.x_min,self.calc_y(self.x_min)]]
        for i in range(n_div-1):
            length_tmp = 0
            x_tmp = array_coord[-1][0]
            while length_tmp < target_segment_length:
                x_tmp += dx
                length_tmp +=  ((dx)**2+(self.calc_y(x_tmp)-self.calc_y(x_tmp-dx))**2)**0.5
            array_coord.append([x_tmp,self.calc_y(x_tmp)])
        array_coord.append([self.x_max,self.calc_y(self.x_max)])
        self.array_coord = array_coord

    def divide_into_n_parts_of_variable_length(self,length_segments_fea):
        """
        #Teilt die Funktion in n gleiche Teile und gibt die Koordinaten wieder
        """
        array_coord = [[self.x_min,self.calc_y(self.x_min)]]
        for i in length_segments_fea:
            #print(i)
            dx = i / 100
            length_tmp = 0
            x_tmp = array_coord[-1][0]
            while length_tmp < i:
                x_tmp += dx
                length_tmp +=  ((dx)**2+(self.calc_y(x_tmp)-self.calc_y(x_tmp-dx))**2)**0.5
            array_coord.append([x_tmp,self.calc_y(x_tmp)])
        #array_coord.append([self.x_max,self.calc_y(self.x_max)])
        self.array_coord = array_coord

    
    
    def make_plot(self):
       """
       ...
       """
       from matplotlib.figure import Figure
       import numpy as np
       fig = Figure(figsize=(4, 4), dpi=150)
       ax = fig.add_subplot(111)
       xlim_lower =    0#- 500
       xlim_upper =    1000# 1500
       ylim_lower =    0#- 500
       ylim_upper =    1000# 1500 

       ax.set_aspect('equal', adjustable='box')
       ax.set_xlim([xlim_lower, xlim_upper])
       ax.set_xticks(np.arange(xlim_lower,xlim_upper+1,100))
       ax.set_ylim([ylim_lower, ylim_upper])
       ax.set_yticks(np.arange(ylim_lower,ylim_upper+1,100))
       
       ax.plot([self.x_min,self.x_max],[self.calc_y(self.x_min),self.calc_y(self.x_max)], color='black',linestyle='None',marker="o",markersize=2)
       x_array = np.linspace(self.x_min,self.x_max,100)
       y_array = []
       for i in x_array:
           y_array.append(self.calc_y(i))
       ax.plot(x_array,y_array, color='black',linestyle='-',linewidth=1)
       
       self.divide_into_n_parts(10)
       ax.plot([i[0] for i in self.array_coord],[i[1] for i in self.array_coord], color='red',linestyle='-',marker = "x", linewidth=1,alpha=0.2)
       
       #ax.set_title(plot_title[:12])
       ax.grid(which='major', color='#666666', linestyle='-',alpha=0.2)
       #ax.minorticks_on()
       #ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
       global main_folder
       global node_computer
       global saving_dir
       #fig.savefig("%sGA_mechanism\\%s\\Pics\\Plot_of_target_func"%(main_folder,node_computer),transparent=True)
       fig.savefig("%sGA_mechanism\\%s\\%s\\Pics\\Plot_of_target_func"%(main_folder,node_computer,saving_dir),transparent=True)
       
















# In[37]:
### dump...


