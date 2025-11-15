# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))
# Calculate Pseudo-inverse
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))

def load_data(path_x='X_train.csv', path_y='Y_train.csv'):
    """
    Carga el dataset desde archivos CSV.

    Args:
        path_x (str): ruta del archivo con características.
        path_y (str): ruta del archivo con etiquetas.

    Returns:
        X (numpy.ndarray): matriz de características normalizadas o crudas.
        Y (numpy.ndarray): vector/matriz de etiquetas.
    """
    X = pd.read_csv(path_x, header=None).to_numpy(dtype=float)
    Y = pd.read_csv(path_y, header=None).to_numpy(dtype=float)
    return X.T, Y.T  # según tu formato, usualmente transpuesto

