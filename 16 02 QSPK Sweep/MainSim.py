#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:30:44 2021

@author: paolo
"""

import GlobalFunctions as AllQueues
import Quadsolve_gurobi as qp
import numpy as np
from Q_class import Queue
import Fred as fg


rng = np.random.default_rng()

def BreakConflicts(Ms,q,R):
    scheduled = -Ms@R
    conflictIndices = np.flatnonzero(scheduled > q)
    for i in conflictIndices: # Which constraints are broken  
        to_reassign = q[i] + np.dot(Ms[i] == 1,R) # How many pairs are actually available, i.e. constraints + POSITIVE incoming scheduling
        concurrents = np.flatnonzero(Ms[i] == -1) #Those are the indices of the -1 terms, the ones generating conflict
        demandIndex = concurrents[-1] 
        R[demandIndex] = min(R[demandIndex],to_reassign)
        to_reassign-=R[demandIndex]
        concurrents = concurrents[:-1]
        rng.shuffle(concurrents)
        for j in concurrents:
            if to_reassign < 0:
                to_reassign = 0
            R[j] = min(R[j],to_reassign)
            to_reassign-=R[j]
                
                
        
def Sim(BatchInput):   
    ############ INPUT
    
    qnet = fg.eswapnet()
    qnet.addpath('ABC')
    qnet.addpath('BCD')
    
    M, qs, ts = qnet.QC.matrix(with_sinks=True)
    LossParam = .9; # This is the eta from backpressure
    Sources_rate = 200000; # Generation rate of the source, Hz.
    t_step = 1e-5; # Length of the time step, s
    time_steps = int(1e4); # Number of steps to simulate
    memo = dict()
    memo_len=int(time_steps/10) # How many configurations should be memoized
    
    skipped = 0
    
    beta = 1    # Demand weight in the scheduling calculation     
    
    ############ PARAMETERS
    
    alpha = Sources_rate*t_step
    
    Q = [Queue(tq[0],tq[1],tran_prob=1) for tq in qs]
    violations = np.zeros(2*len(Q))
    nodeset = set()
    for tq in qs:
        nodeset = nodeset.union(set(tq))# Set of the nodes. May not be necessary now but will be useful going forward
    
    
    for i in range(len(M)):
        if 1 not in M[i]:
            Q[i].SetPhysical(alpha)
    
    
    [q.SetService(BatchInput[q.nodes],t_step) for q in Q if q.nodes in BatchInput]
    
    r_matrix = -np.identity(len(Q))
    # From now on, every variable number with an s in front is to be read as x_tilde
    Ms = np.concatenate((M,r_matrix),1)
    
    Ns = np.concatenate((np.zeros((len(M),len(M[0]))),r_matrix),1)
    
    
    ProbDim = len(Ms[1]) # Dimensionality of the problem, i.e. per-node rates+ #queues
    R = np.zeros((ProbDim,time_steps)) # Initializing the R array, that will contain the R vector at each time step
    
    #Instantiate the problem here
    
    qp_P = (Ms.T)@Ms + 2*beta*(Ns.T)@Ns #These are computed here because they don't vary.
    qp_G = np.vstack((Ms,Ns))
    
    
    
    ProbDim = len(Ms[1]) # Dimensionality of the problem
    R = np.zeros((ProbDim,time_steps)) # Initializing the R array, that will contain the R vector at each time step
    
    qp_A = -Ns
    
    to_relax = [] # List of the service queues'indices: their r_ij constraints will be removed
    for i in range(len(Q)):
        if Q[i].serv == "service":
            to_relax.append(i)
    
    qp_A = np.delete(qp_A, to_relax, 0)
    qp_b = np.zeros(len(qp_A))
            
    dem_arr_rates = [getattr(q,"PoissParam",0) for q in Q] # Already converted to timesteps^-1
    
    for Maintimestep in range(time_steps):
        # memo = dict() #Uncomment here to DISABLE memoization 
        Qt = np.array([q.Qdpairs for q in Q])
        Dt = np.array([q.demands for q in Q]) 
        A = AllQueues.Arrivals(Q)
        L = AllQueues.Losses(Q,Qt,LossParam)
        B = AllQueues.Demand(Q) 
        for nd in nodeset:
            qp_q, qp_h = qp.UpdateConstraints(Q,nd,qs,beta,Dt,dem_arr_rates,B,Ns,Qt,LossParam,L,alpha,A,Ms)
            partialsol, memo                 = qp.QuadLyap(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b,Dt,memo,memo_len,nd,ts,qs)
            #                        , debug_mixedsol
            #debug_memo[nd] = debug_mixedsol
            R[:len(ts),Maintimestep] = R[:len(ts),Maintimestep] + partialsol[:len(ts)]
            for i in range(len(qs)):
                if nd in qs[i]:
                    flag = int((partialsol[len(ts) + i] !=0)) + int((R[len(ts) + i,Maintimestep] != 0))
                    if flag == 1: # Either this node was the only one to request consumption, or it was already requested by another node BUT NOT THIS ONE
                            R[len(ts) + i,Maintimestep] += partialsol[len(ts) + i]
                    elif flag == 2: # This node requested consumption, but the other node had already requested it -> break the tie
                        R[len(ts) + i,Maintimestep] = min(R[len(ts) + i,Maintimestep],partialsol[len(ts) + i]) 
        actualQ =Qt+A-L
        BreakConflicts(Ms, actualQ,R[:,Maintimestep])
        violations += AllQueues.CheckActualFeasibility(Ms,Ns,R[:,Maintimestep],Qt,Dt,L,A,B)
        debug_violation = AllQueues.CheckActualFeasibility(Ms,Ns,R[:,Maintimestep],Qt,Dt,L,A,B)
        #if debug_violation:
            #breakpoint()
            #pass
        AllQueues.Evolve(Q,Ms,R[:,Maintimestep])
    
    
    print(f"There were {violations} steps in which the scheduler asked for something impossible, {violations/time_steps*100}% of the times")
    D_final = [q.demands for q in Q]
    Tot_dem_rate = sum(BatchInput.values())
    unserved = sum(D_final)/(t_step*time_steps*Tot_dem_rate) #Unserved demands at the end divided by an approximation of the total incoming demand
    return unserved #, violations
