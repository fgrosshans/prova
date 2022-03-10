#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:29:20 2021

@author: paolo
"""


import GlobalFunctions as AllQueues
from itertools import combinations
from Q_class import Queue
import Fred as fg

ArrRates = {
            frozenset(('A','B')) : 200000,
            frozenset(('C','B')) : 200000,
            frozenset(('D','C')) : 200000
            }

def Sim(BatchInput,memoDict):
    flatInput = tuple(zip(*BatchInput.items())) # List of tuples
    # memoDict = dict() # Uncomment to DISABLE memoization
    for i in memoDict.keys():
        flatMemo = i
        if flatInput[1][0] >= flatMemo[1][0] and flatInput[1][1] >= flatMemo[1][1]:
            output = memoDict[i]
            return output
    ######################################## INPUTS 
    
    LossParam = .9 # This  is the eta from BP paper
    t_step = 1e-6; # Length of the time step, s
    time_steps = int(1e4); # Number of steps to simulate
    BSM_Success_Probability = 1;

    
    ######################################## READING INPUT
    
    # Deriving the scheduling matrix and the lists of queues and scheduling rates
    # from FG's code, see fg.smalltest() for more information    
    qnet = fg.eswapnet()
    qnet.addpath('ABC')
    qnet.addpath('BCD')
    M, qs, ts = qnet.QC.matrix(with_sinks=True)
    
    ### Building the model 
    
    
    nodeset = set()
    for tq in qs:
        nodeset = nodeset.union(set(tq))# Set of the nodes. May not be necessary now but will be useful going forward
    
    Q = [Queue(tq[0],tq[1]) for tq in combinations(nodeset,2)]
   
    [q.SetPhysical(ArrRates[q.nodes],t_step) for q in Q if q.nodes in ArrRates]
    [q.SetService(BatchInput[q.nodes],t_step) for q in Q if q.nodes in BatchInput]
    
    
    # This next block builds a dictionary of neighbors for each queue, so that AC - BD is never checked as a potential entanglement swapping operation
    ConnectedTo = {}
    for label in nodeset:
        ConnectedTo[label] = [q for q in Q if (label in q.nodes)]
    
       
    
    ######################################## MAIN LOOP
    for Maintimestep in range(time_steps):
        AllQueues.Demand(Q)
        AllQueues.Generate(Q)
        AllQueues.Loss(Q, LossParam)
        AllQueues.Consume(Q)
        AllQueues.Schedule(Q,ConnectedTo)
        AllQueues.Update(Q,BSM_Success_Probability)
    D_final = [q.demands for q in Q]
    Q_final = [q.Qdpairs for q in Q]
    Tot_dem_rate = sum(BatchInput.values())
    unserved = sum(D_final)/(t_step*time_steps*Tot_dem_rate) #Unserved demands at the end divided by an approximation of the total incoming demand
    if unserved >= 0.15:
        to_store = tuple(zip(*BatchInput.items()))
        memoDict[to_store] = (unserved, Q_final, D_final) 
    return unserved, Q_final, D_final
    
