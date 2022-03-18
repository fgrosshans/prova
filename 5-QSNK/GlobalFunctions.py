#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:13:27 2021

@author: paolo

This class contains functions that interface to all queues in the system, both of pairs and of demands.
"""
import numpy as np

def Arrivals(Q):
    A = np.zeros(len(Q))
    for i in range(len(Q)):
        A[i] = Q[i].Generate();
    return A

def Losses(Q,Qt,LossParam):
    L = np.zeros(len(Q))
    for i in range(len(Q)):
        L[i] = Q[i].Loss(LossParam);
    return L

def Demand(Q):
    B = np.zeros(len(Q))
    for i in range(len(Q)):
        B[i] = Q[i].Demand();
    return B

def Evolve(Q,Ms,Ns,R,Qt,L,A,Dt,B):
    rng = np.random.default_rng()
    actualQ = Qt - L + A
    actualD = np.array(Dt) + np.array(B)
    actual_qp_q = np.hstack((actualQ,actualD))
    G = np.vstack((Ms,Ns))
    scheduled = -G@R 

    #conflictIndices = np.flatnonzero(scheduled > actual_qp_q) # Find which constraints are broken  
    #doubleQ = np.hstack((actualQ,actualQ)) # I'm checking that nothing violates Q constraints: This vector allows to check both Q and D lines against Q constraints
    #for i in conflictIndices: 
    #    to_reassign = doubleQ[i] + np.dot(G[i] == 1,R) # How many pairs are actually available, i.e. constraints + POSITIVE incoming scheduling
    #    concurrents = np.flatnonzero(G[i] == -1) # Those are the indices of the -1 terms, the ones generating conflict
    #    demandIndex = concurrents[-1] # Demand is the priority in breaking these conflicts
    #    R[demandIndex] = min(R[demandIndex],to_reassign)
    #    to_reassign-=R[demandIndex]
    #    concurrents = concurrents[:-1]
    #    rng.shuffle(concurrents) # After serving demand, the rest of the resources are assigned randomly.
    #    for j in concurrents:
    #        if to_reassign < 0:
    #            to_reassign = 0
    #        R[j] = min(R[j],to_reassign)
    #        to_reassign-=R[j]
    D_t1 = actualD - R[-len(Qt):]
    Q_t1 = actualQ + Ms@R
    for i in range(len(Qt)):
        Q[i].Qdpairs = int(max(Q_t1[i],0))
        Q[i].demands = D_t1[i]

def CheckActualFeasibility(M,N,R,Q,D,L,A,B): # This method checks if the scheduler's order are actually feasible
                                             # and counts the instances in which they are not.
    G = np.vstack((M,N))
    scheduled = -G@R 
    error = False
    actualQ = Q - L + A
    actualD = np.array(D) + np.array(B)
    actual_qp_q = np.hstack((actualQ,actualD))
    check = scheduled > actual_qp_q
    error = check.any()
    return error