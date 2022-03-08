#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:42:07 2022

@author: paolo
"""

from MainSim import Sim
import numpy as np
from time import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime

n_points = 5 # Number of points along each direction

if __name__ == '__main__':
    now = datetime.now().strftime("%H:%M:%S")
    print(f"Starting simulation at {now}")
    SPair_1 = ("A","C")
    SPair_2 = ("B","D")
    
    DemRates1 = np.linspace(50000,250000,n_points)
    DemRates2 = np.linspace(50000,250000,n_points)
    
    Output_RAW = []
    
    nprocs = 10 #Number of workers in the pool
    
    InputList = []
    
    pool = mp.Pool(processes=nprocs)
    
    t1 = time()
    for r1 in DemRates1:
        for r2 in DemRates2:
            SimInput = {frozenset(SPair_1) : r1,
                        frozenset(SPair_2) : r2}
            InputList.append(SimInput)
    
    Output_RAW = pool.map(Sim,InputList)       
    
    t2 = time()-t1
    now = datetime.now().strftime("%H:%M:%S")
    print(f"Ending at {now}. Elapsed time: {np.floor(t2/60)} minutes and {(t2/60-np.floor(t2/60))*60:.2f} seconds")
    Output = np.array(Output_RAW).reshape((n_points,n_points),order="F")
    Output = np.flipud(Output)

    plt.figure(1)
    plt.imshow(Output*100,vmin=0,vmax=100)
    plt.colorbar()
    xlabels = ['{:.2f}'.format(i) for i in DemRates1/1000]
    ylabels = ['{:.2f}'.format(i) for i in np.flip(DemRates2)/1000]
    plt.xticks(range(len(DemRates1)),xlabels,rotation=70)
    plt.yticks(range(len(DemRates2)),ylabels)
    plt.xlabel(f"Average demand rate across pair {SPair_1[0]}-{SPair_1[1]}, kHz")  
    plt.ylabel(f"Average demand rate across pair {SPair_2[0]}-{SPair_2[1]}, kHz")
    plt.title("% Unserved demands, Partial Knowledge Quadratic")   
    plt.show()
