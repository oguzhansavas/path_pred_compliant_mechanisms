# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:29:36 2022

@author: Oguzhan Savas
"""
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import keras_tuner as kt
import joblib


# Create the hypermodel for optimization
initializer = keras.initializers.GlorotNormal()

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(8,)))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 4)):
        reg_rate = hp.Float("reg", min_value=1e-3, max_value=1e-1, sampling="log")
        model.add(
            Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation="relu",
                kernel_initializer = initializer,
                kernel_regularizer=regularizers.l2(reg_rate)
            )
        )
    if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.2))
    model.add(Dense(100, activation="linear"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"])
    return model


# Conduct bayesian search
def bayes_search(hypermodel, trials, x_train, y_train, x_test, y_test):
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,#hypermodel,
        objective="val_loss",
        max_trials=trials,
        seed = 42,
        overwrite=True)

    # Start the search
    tuner.search(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

    # Get the best model
    #best_model = tuner.get_best_models()[0]
    
    # Build the model.
    #best_model.build()
    
    return tuner.get_best_models()[0]#, best_model.summary()

def get_model(best_model):
    best_model = best_model.build()
    return best_model

# save best model
def save_model(best_model):
    model_filename = "best_model.sav"
    joblib.dump(best_model, model_filename)
    
    
# Fit the best model to the data
def model_fit(best_model, batch_size, x_train, y_train, x_test, y_test):
    history = best_model.fit(x_train, y_train, verbose=0, validation_data=(x_test, y_test))
    return history