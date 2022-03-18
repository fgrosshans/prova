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
from math import exp

with open("inputs.in") as f: # Importing variables from the inputs file
    exec(f.read())

LossParam = 1 - exp(-kappa*t_step) # Done here too, just so that the user can check
print("###############Recap:###############")
print(f"- {topologyname} topology, {n_points}x{n_points} pixels")
print(f"- Losses (1-eta): {1 - LossParam:.2f}, Beta: {beta}")
print(f"- Service pairs: - {SPair_1}, {DemRates1[0]} - {DemRates1[-1]} Hz,")
print(f"                 - {SPair_2}, {DemRates2[0]} - {DemRates2[-1]}")
print(f"Parallel Run: {ParallelRun}")

if __name__ == '__main__':
    
    now = datetime.now().strftime("%H:%M:%S")
    print(f"Starting simulation at {now}")
    
    Output_RAW = [] # tuples of unservedpairs, Qstate, Dstate

    unserved_RAW = []
    Q_final_RAW = []
    D_final_RAW = []    

    InputList = []
    
    t1 = time()
    for r1 in DemRates1:
        for r2 in DemRates2:
            SimInput = {frozenset(SPair_1) : r1,
                        frozenset(SPair_2) : r2}
            InputList.append(SimInput)
    

    nprocs = mp.cpu_count() #Number of workers in the pool
    
    if ParallelRun:
        with mp.Manager() as manager:
            memo = manager.dict()
            memoList = [memo for i in InputList]
            with manager.Pool(processes=nprocs) as p:
                Output_RAW = p.starmap(Sim,zip(InputList,memoList))
                p.close()
                p.join()
    else:
        from functools import partial
        memo = dict()
        MemoizedSim = partial(Sim,memoDict = memo)
        Output_RAW = list(map(MemoizedSim,InputList))
   
        
    
    t2 = time()-t1
    now = datetime.now().strftime("%H:%M:%S")
    print(f"Ending at {now}. Elapsed time: {np.floor(t2/60)} minutes and {(t2/60-np.floor(t2/60))*60:.2f} seconds")
    
    unserved_RAW, Q_final_RAW, D_final_RAW = zip(*Output_RAW)
    
    unserved = np.array(unserved_RAW).reshape((n_points,n_points),order="F")
    unserved = np.flipud(unserved)

    qnumber = len(Q_final_RAW[0])   
    Q_final = np.array(Q_final_RAW).reshape((n_points,n_points,qnumber),order="F") 
    D_final = np.array(D_final_RAW).reshape((n_points,n_points,qnumber),order="F")  
    
    Q_final = np.flipud(Q_final)
    D_final = np.flipud(D_final)
    
    
    
    plt.figure(1)
    plt.imshow(unserved*100,vmin=0,vmax=10)
    plt.colorbar()
    
    n_labels = 25
    xlabels = ['{:.2f}'.format(i) for i in np.linspace(DemRates1[0],DemRates1[-1],n_labels)/1000]
    ylabels = ['{:.2f}'.format(i) for i in np.linspace(DemRates2[0],DemRates2[-1],n_labels)/1000]
    ylabels = np.flip(ylabels)

    try:
        xintersect = np.where(DemRates1 == np.atleast_1d(200000))[0]
        yintersect = np.where(np.flip(DemRates2) == np.atleast_1d(200000))[0]
        xline = [0, xintersect]
        yline = [yintersect, len(unserved)-1]
        # plt.plot(xline,yline)
    except ValueError:
        print("200.00 is not a tick in the plot, can't plot the optimal diagonal")
    
    plt.xticks(np.linspace(0,n_points-1,n_labels),xlabels,rotation=70)
    plt.yticks(np.linspace(0,n_points-1,n_labels),ylabels)
    plt.xlabel(f"Average demand rate across pair {SPair_1[0]}-{SPair_1[1]}, kHz")  
    plt.ylabel(f"Average demand rate across pair {SPair_2[0]}-{SPair_2[1]}, kHz")
    schedulername = "FK Quadratic"
    plt.title(f"% Unserved demands,{schedulername},{topologyname}")
    plt.show()
    plt.savefig(f"{n_points}x{n_points}_{schedulername}_{topologyname}_{now}_{nprocs}t")
    np.savez(f"{n_points}x{n_points}_{schedulername}_{topologyname}_{now}_{nprocs}t",unserved = unserved, Q_final=Q_final, D_final=D_final, pair1=SPair_1,pair2=SPair_2,rates1=DemRates1,rates2=DemRates2,threads=nprocs,n_points=n_points,schedulername=schedulername,allow_pickle=True)
