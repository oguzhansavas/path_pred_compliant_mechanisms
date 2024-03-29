# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:08:10 2022

@author: ogsa991b
"""

import tensorflow
from tensorflow import keras
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class MyHyperModel(kt.HyperModel):
    
    def build(self, hp):
        model = tensorflow.keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(8,)))
        # Tune number of layers
        for i in range(hp.Int("num_layers", 1, 4)):
            reg_rate = hp.Float("reg", min_value=1e-3, max_value=1e-1, sampling="log")
            model.add(layers.Dense(
                # Tune number of units
                units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation = "relu",
                kernel_initializer = keras.initializers.GlorotNormal(),
                kernel_regularizer = regularizers.l2(reg_rate)))
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(100, activation="linear"))
        learn_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = learn_rate),
                      loss = "mean_squared_error",
                      metrics = ["mean_squared_error"])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
                         batch_size = hp.Choice("batch_size", [16, 32, 64]),
                         **kwargs)
