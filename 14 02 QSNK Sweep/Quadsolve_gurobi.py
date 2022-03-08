#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:35:44 2022

@author: paolo
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def QuadLyap(P, q, G, h, A, b,#Arguments for the solver
             Dt,memo,memo_len):  #Arguments to manage the memoization process

#This function wraps the quadratic solver 

    cstate = str(Dt) + str(h) # Keeping track of the whole system state. Notice how Qt is implicit 
                              # in ub: this makes the system more likely to skip optimization. 
                              # It's cheaper in memory to store cstate as a string.
    
    if cstate in memo: # If this configuration is already known, skip optimization
        memo[cstate][1]+=1 # Increase frequency of this configuration in the memo dictionary
    else:
        sol = solve_qp(P, q, G, h, A, b)
        if len(memo) > memo_len: # This prevents the memo dictionary from becoming too large
            to_delete = min(memo.keys(), key= lambda x : memo[x][1]) # Find the state with minimal frequency
            memo.pop(to_delete)
        memo[cstate] = [sol,1] # Store config and start frequency count
    return memo[cstate][0],memo # Retrieve scheduling vector

def UpdateConstraints(Q,beta,Dt,d_arr_rates,N,
                      Qt,q_loss_rate,q_arr_rate,M):
    
    ExpDt1 = Dt + np.array(d_arr_rates) # constraining term = etaQt...
    ExpQt1 = np.array(q_loss_rate*Qt,dtype=float)
    for i in range(len(ExpQt1)):
        if Q[i].type == "physical":
            ExpQt1 += q_arr_rate # ... + alpha if the queue is physical
    
    h = np.hstack((ExpQt1,ExpDt1)) # In the calculations this looks like a vstack. However, numpy has no concept of row/column vectors 
                                   # and will try to create a matrix if asked for a vstack between two vectors
    
    q = 2*beta*ExpDt1@N + ExpQt1@M #Linear term of the OF
    return q, h

def solve_qp(P, q, G=None, h=None, A=None, b=None):
    prob = gp.Model()
    prob.Params.OutputFlag = 0
    prob.Params.threads = 1
    R = prob.addMVar(len(P[0]),vtype=GRB.INTEGER)
    aux = R@P
    expr = 0.5*aux@R + q@R
    prob.setObjective(expr,GRB.MINIMIZE)
    constrleq = -G@R <= h
    # constreq = A@R == 0
    # prob.addConstr(constreq)
    prob.addConstr(constrleq)
    prob.optimize()
    if prob.Status == GRB.OPTIMAL:
        sol = np.array([v.x for v in prob.getVars()])
    else:
        sol = np.zeros(len(P))    
    return sol
        