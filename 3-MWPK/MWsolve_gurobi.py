#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:35:44 2022

@author: paolo
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def Schedule(q, G, h, A, b,#Arguments for the wrapped solver
             Dt,memo,memo_len,node,tran_labels,q_labels):  #Arguments to manage the memoization process

    cstate = str(h) + node # Keeping track of the whole system state. 
                           # Notice how Qt is implicit in ub: this makes the system more likely to sk optimization.
                           # It is cheaper to store cstate as a string.
    
    if cstate in memo: # If this configuration is already known, skip optimization
        memo[cstate][1]+=1 # Increase frequency
        sol = memo[cstate][0] # Retrieve solution
    else:
        sol = solve_lp(q, G, h, A, b)

        if len(memo) > memo_len: # If the memory dict is too long
            to_delete = min(memo.keys(), key= lambda x : memo[x][1]) 
            memo.pop(to_delete)
        memo[cstate] = [sol,1] # Store config and start frequency count
    
    for i in range(len(tran_labels)): # Once the solution is established, it must be cleaned
                                      # i.e. the estimate swapping rates for disconnected nodes
                                      # must be removed
        active = tran_labels[i]
        if active[2] is not node:
            sol[i] = 0 # Dumping anything that doesn't come from connected nodes
    return sol,memo

def UpdateConstraints(Q,node,qlabels,beta,Dt,d_arr_rates,Bt,N,
                      Qt,q_loss_param,Lt,q_arr_rate,At,M):
    
    ExpQt1 = q_loss_param*Qt + q_arr_rate
    ExpDt1 = Dt + np.array(d_arr_rates)    
    for i in range(len(ExpQt1)):
        if node in qlabels[i]:
            ExpQt1[i] = Qt[i] + At[i] - Lt[i] # However, if the queue is connected, the upper bounds are exact. 
            ExpDt1[i] = Dt[i] + Bt[i]
    h = np.hstack((ExpQt1,ExpDt1)) # In the calculations this looks like a vstack. However, numpy has no concept of row/column vectors and will try to create a matrix if asked for a vstack between two vectors
    q = 2*beta*ExpDt1@N + ExpQt1@M # Linear term for the OF
    return q, h

def solve_lp(q, G=None, h=None, A=None, b=None):
    prob = gp.Model()
    prob.Params.OutputFlag = 0
    prob.Params.threads = 1 # Restricting Gurobi to one thread, to have multiple instances running in parallel. The problem is very small, so this is not detrimental
    R = prob.addMVar(len(q),vtype=GRB.INTEGER)
    expr = q@R
    prob.setObjective(expr,GRB.MINIMIZE)
    constrleq = -G@R <= h
    # constreq = A@R == 0
    # prob.addConstr(constreq)
    prob.addConstr(constrleq)
    prob.optimize()
    if prob.Status == GRB.OPTIMAL:
        sol = np.array([v.x for v in prob.getVars()])
    else:
        sol = np.zeros(len(q))
    return sol
    