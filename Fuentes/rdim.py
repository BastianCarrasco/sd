# Dimensionality Reducting 


import pandas     as pd
import numpy      as np
from   utility      import *


def pc_svd():
    ...
    return
    
def updW_adam():
    ...
    return
#
def gradW1():
    ...
    return
#
def forward():
    ...
    return
#
def w2_pinv():
    ...
    return

#
def train_minibatch():
    ...
    return

#SLS's Training 
def train_sae():
    ...    
    return(W1) 

# Beginning ...
def main():    
    lod_data()
    load_param()
    zscores_data()
    V  = train_sae()    
    V  = pc_svd()    
    save_new_data()
    
       
if __name__ == '__main__':   
	 main()

