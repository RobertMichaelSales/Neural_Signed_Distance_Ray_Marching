""" Created: 29.01.2024  \\  Updated: 15.03.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf

#==============================================================================
# Define the positional encoding layer (from the Neural Radiance Fields paper)

def PositionalEncoding(inputs,frequencies):
    
    # Define the positional encoding frequency bands
    frequency_bands = 2.0**tf.linspace(0.0,frequencies-1,frequencies)
    
    # Define the positional encoding periodic functions
    periodic_functions = [tf.math.sin,tf.math.cos]
    
    # Create an empty list to fill with encoding functions
    encoding_functions = []
    
    # Iterate through each of the frequency bands
    for fb in frequency_bands:
        
        # Iterate through each of the periodic functions
        for pf in periodic_functions:
            
            # Append encoding lambda functions with arguments
            encoding_functions.append(lambda x, pf=pf, fb=fb: pf(x*np.pi*fb))
       
        ##
        
    ##
    
    # Evaluate the encoding function on each input and concatenate
    x = tf.concat([ef(inputs) for ef in encoding_functions],axis=-1)
    
    return x

##

#==============================================================================

# Define a function that constructs the 'basic' network 

def ConstructNetwork(layer_dimensions,frequencies):

    # Set python, numpy and tensorflow random seeds for the same initialisation
    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
 
    # Compute the number of total network layers
    total_layers = len(layer_dimensions)

    # Iterate through network layers
    for layer in range(total_layers):
        
        # Add the input layer
        if (layer == 0):
            
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="L{}_input".format(layer))
            
            # Add positional encoding if 'frequencies' > 0
            if (frequencies > 0):
                x = PositionalEncoding(inputs=input_layer,frequencies=frequencies)               
            else:
                x = input_layer
            ##
                       
        # Add the final output layer
        elif (layer == (total_layers - 1)):
            
            output_layer = tf.keras.layers.Dense(units=layer_dimensions[layer],activation="tanh",name="L{}_output".format(layer))(x)
         
        # Add the intermediate dense layers
        else:        
        
            x = tf.keras.layers.Dense(units=layer_dimensions[layer],activation="relu",name="L{}_dense".format(layer))(x)
    
        ##
        
    ##
    
    # Create the network model
    SquashSDF = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return SquashSDF

##

#==============================================================================