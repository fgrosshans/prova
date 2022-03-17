#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:30:44 2021

@author: paolo
"""

with open("inputs.in") as f:
    exec(f.read())


import GlobalFunctions as AllQueues
import MWsolve_gurobi as mw
import numpy as np
from Q_class import Queue
import Fred as fg

def Sim(BatchInput,memoDict):   
    flatInput = tuple(zip(*BatchInput.items())) # List of tuples
    # memoDict = dict() # Uncomment to DISABLE memoization
    for i in memoDict.keys():
        flatMemo = i
        if flatInput[1][0] >= flatMemo[1][0] and flatInput[1][1] >= flatMemo[1][1]:
            output = memoDict[i]
            return output
    # Deriving the scheduling matrix and the lists of queues and scheduling rates
    # from FG's code, see fg.smalltest() for more information    
    qnet = fg.eswapnet()
    for rt in routes:
        qnet.addpath(rt)

    M, QLabels, R_components = qnet.QC.matrix(with_sinks=True)
    
    ### Building the model 
    Q = [Queue(tq[0],tq[1],tran_prob=1) for tq in QLabels]
    
    [q.SetPhysical(ArrRates[q.nodes],t_step) for q in Q if q.nodes in ArrRates]
    [q.SetService(BatchInput[q.nodes],t_step) for q in Q if q.nodes in BatchInput]
    dem_arr_rates = np.array([getattr(q,"PoissParam",0) for q in Q]) # Already converted to timesteps^-1
    alpha = [getattr(q,"GenPParam",0) for q in Q] # Already converted to timesteps^-1
    
    # Defining the building blocks of the optimization problem.
    # From now on, every variable with an s in front is to be read as \tilde{x}
    
    r_matrix = -np.identity(len(Q)) #Matrix for the demand part
    Ms = np.concatenate((M,r_matrix),1) # Full "Big M" matrix
    Ns = np.concatenate((np.zeros((len(M),len(M[0]))),r_matrix),1) # Auxiliary matrix analogous to big M but for demands
    qp_G = np.vstack((Ms,Ns)) # Full constraints matrix
    qp_A = -Ns # This matrix is used to force to zero the consumption along every non-service queue(see next lines)
    
    to_relax = [] # List of the service queues'indices: their r_ij=0 constraints will be removed
    for i in range(len(Q)):
        if Q[i].serv == "service":
            to_relax.append(i)
    
    qp_A = np.delete(qp_A, to_relax, 0)
    qp_b = np.zeros(len(qp_A))   
    
    memo = dict() # Initializing the memory
    ProbDim = len(Ms[1]) # Dimensionality of the problem
    R = np.zeros((ProbDim,time_steps)) # Initializing the R array, that will contain the R vector at each time step
    violations=0

    for Maintimestep in range(time_steps):
        Qt = np.array([q.Qdpairs for q in Q])
        Dt = np.array([q.demands for q in Q]) # Those are NOT the demand arrivals: they are the total demands pending across each queue.
        A = AllQueues.Arrivals(Q)
        L = AllQueues.Losses(Q,Qt,LossParam)
        B = AllQueues.Demand(Q)
        qp_q, qp_h = mw.UpdateConstraints(Q,beta, Dt, dem_arr_rates, Ns, Qt, LossParam, alpha, Ms)
        R[:,Maintimestep], memo = mw.Schedule(qp_q, qp_G, qp_h, qp_A, qp_b,Dt ,memo,memo_len)
        violations += AllQueues.CheckActualFeasibility(Ms,Ns,R[:,Maintimestep],Qt,Dt,L,A,B)
        AllQueues.Evolve(Q,Ms,Ns,R[:,Maintimestep],Qt,L,A,Dt,B) # This method disobeys to impossible orders. 
    ## OUTPUT
    # print(f"There were {violations} steps in which the scheduler asked for something impossible, {violations/time_steps*100}% of the times")
    D_final = [q.demands for q in Q]
    Q_final = [q.Qdpairs for q in Q]
    Tot_dem_rate = sum(BatchInput.values())
    
    
    unserved = sum(D_final)/(t_step*time_steps*Tot_dem_rate) #Unserved demands at the end divided by an approximation of the total incoming demand
    if unserved >= 0.15:
        to_store = tuple(zip(*BatchInput.items()))
        memoDict[to_store] = (unserved, Q_final, D_final) 
    return unserved, D_final, Q_final #, violations
