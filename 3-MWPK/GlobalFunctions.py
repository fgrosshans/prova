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

def Evolve(Q,M,R):
    Q_t = np.array([q.Qdpairs for q in Q])
    D_t = np.array([q.demands for q in Q])
    D_t1 = D_t - R[-len(Q):]
    Q_t1 = Q_t + M@R
    for i in range(len(Q)):
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