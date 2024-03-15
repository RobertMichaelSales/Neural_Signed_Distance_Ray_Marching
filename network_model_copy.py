""" Created: 29.01.2024  \\  Updated: 19.02.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Sine Layer' 

def SineLayer(inputs,units,name):
    
    # Mathematically: x1 = sin(W1*x0 + b1)
    
    x = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense")(inputs))
             
    return x

#==============================================================================
# Define a 'Sine Block'

def SineBlock(inputs,units,name):
    
    # Mathematically: x1 = (1/2) * (x0 + sin(w12*sin(w11*x0 + b11) + b12))
            
    sine_1 = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense_a")(inputs))
    sine_2 = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense_b")(sine_1))
    
    x = tf.math.add(inputs,sine_2)
    
    return x

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
# Define the Gaussian mapping layer (from the Fourier Features Let ... paper)

# This will need modifying if it works since B must be saved with the network

# https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb#scrollTo=BwbEtsn0gB2h

def RFGaussianMapping(inputs,frequencies,stddev):
    
    # Generate a matrix of Gaussian features
    gaussian_kernel = tf.random.normal(shape=[frequencies,inputs.shape[-1]],mean=0.0,stddev=stddev,dtype=tf.dtypes.float32)    
    
    # Compute the sin components    
    x_sin_gaussian = tf.math.sin(tf.linalg.matmul((2.0*np.pi*inputs),tf.transpose(gaussian_kernel)))
    
    # Compute the cos components
    x_cos_gaussian = tf.math.cos(tf.linalg.matmul((2.0*np.pi*inputs),tf.transpose(gaussian_kernel)))
    
    # Evaluate the encoding function on each input and concatenate
    x = tf.concat([x_sin_gaussian,x_cos_gaussian],axis=-1)   
    
    return x,gaussian_kernel

##

#==============================================================================

# Define a function that constructs the 'basic' network 

def ConstructNetworkBASIC(layer_dimensions,frequencies,activation):

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
        
            x = tf.keras.layers.Dense(units=layer_dimensions[layer],activation=activation,name="L{}_dense".format(layer))(x)
    
        ##
        
    ##
    
    # Create the network model
    SquashSDF = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return SquashSDF

##

#==============================================================================
# Define a function that constructs the 'siren' network 

def ConstructNetworkSIREN(layer_dimensions,frequencies,activation):

    # Set python, numpy and tensorflow random seeds for the same initialisation
    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
 
    # Compute the number of total network layers
    total_layers = len(layer_dimensions)

    # Iterate through network layers
    for layer in range(total_layers):
        
        # Add the input layer
        if (layer == 0):
            
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            
            # Add positional encoding if 'frequencies' > 0
            if (frequencies > 0):
                x = PositionalEncoding(inputs=input_layer,frequencies=frequencies)               
                x = SineLayer(inputs=x          ,units=layer_dimensions[layer+1],name="l{}_sinelayer".format(layer))
            else:
                x = SineLayer(inputs=input_layer,units=layer_dimensions[layer+1],name="l{}_sinelayer".format(layer))
            ##
          
        # Add the final output layer
        elif (layer == (total_layers - 1)):
          
            output_layer =  tf.keras.layers.Dense(units=layer_dimensions[layer],name="l{}_output".format(layer))(x)
          
        # Add the intermediate sine blocks
        else:
            
            x = SineBlock(inputs=x,units=layer_dimensions[layer],name="l{}_sineblock".format(layer))
    
        ##
        
    ##
    
    # Create the network model
    SquashSDF = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return SquashSDF

##

#==============================================================================
# Define a function that constructs the 'basic' network 

def ConstructNetworkGAUSS(layer_dimensions,frequencies,stddev,activation):

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
                x,gaussian_kernel = RFGaussianMapping(inputs=input_layer,frequencies=frequencies,stddev=stddev)               
            else:
                x,gaussian_kernel = input_layer,None
            ##
                       
        # Add the final output layer
        elif (layer == (total_layers - 1)):
            
            output_layer = tf.keras.layers.Dense(units=layer_dimensions[layer],activation="tanh",name="L{}_output".format(layer))(x)
         
        # Add the intermediate dense layers
        else:        
        
            x = tf.keras.layers.Dense(units=layer_dimensions[layer],activation=activation,name="L{}_dense".format(layer))(x)
    
        ##
        
    ##
    
    # Create the network model
    SquashSDF = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    # Assign the gaussian kernel
    SquashSDF.gaussian_kernel = gaussian_kernel
    
    return SquashSDF

##

#==============================================================================

#==============================================================================