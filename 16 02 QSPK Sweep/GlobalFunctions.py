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
    Q_t1 = np.maximum(0,Q_t + M@R) 
    for i in range(len(Q)):
        Q[i].Qdpairs = Q_t1[i]
        Q[i].demands = max(D_t1[i],0)

#This happened:

"""
[A[B]C B[C]D]['BC', 'AB', 'CD', 'AC', 'BD']

D_t
[0. 0. 0. 5. 0.]

D_t1
[0. 0. 0. 3. 0.]

R
[1. 3. 0. 0. 0. 2. 0.]

Q_t
[6. 1. 4. 0. 0.] Notice how queue 3 didn't have enough pairs to serve 2 demands, even after swapping

Q_t1
[2. 0. 1. 0. 3.]

Q_t+M@R
[ 2.  0.  1. -1.  3.] The queue went negative and was corrected. However, the demand counter was lowered by two. This is a problem

R
[1. 3. 0. 0. 0. 2. 0.]

Q_t
[6. 1. 4. 0. 0.]

Q_t1-Q_t
[-4. -1. -3.  0.  3.]

D_t1-D_t
[ 0.  0.  0. -2.  0.]

M@R
[-4. -1. -3. -1.  3.]
The problem could be corrected by changing R into the actual scheduling vector.
This should NOT be necessary, since all decision are based on perfect local information.
"""


def CheckActualFeasibility(M,N,R,Q,D,L,A,B):
    G = np.vstack((M,N))
    scheduled = -G@R 
    error = False
    actualQ = Q - L + A
    actualD = D + B
    actual_qp_q = np.hstack((actualQ,actualD))
    check = scheduled > actual_qp_q
    #error = bool(check[1])
    return np.array(check,dtype=bool)
            


