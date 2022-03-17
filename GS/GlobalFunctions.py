#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:13:27 2021

@author: paolo

This class contains functions that interface to all queues in the system, both of pairs and of demands.
"""
import random as rd
import numpy as np

def Generate(Q):
    for queue in Q:
        queue.Generate();
        
def Loss(Q,LossParam):
    for q in Q:
        q.Loss(LossParam)
        
def Consume(Q):
    for queue in Q:
        if (queue.serv == "service"):
            queue.Consume();

def Update(Q,BSM):
    for q in Q:
        q.Update(BSM)
               
def Demand(Q):
    D = np.zeros(len(Q))
    for i in range(len(Q)):
        D[i] = Q[i].Demand();
            
def Schedule(Q,connectedto):
    nodes = connectedto.keys()
    nodes = rd.sample(nodes,k=len(nodes))
    for ActiveNode in nodes:
        Q_Ready = [q for q in connectedto[ActiveNode] if q.Qdpairs> 0]
        while len(Q_Ready) >= 2:
            parent1, parent2 = rd.sample(Q_Ready,2)
            childnode1 = next(i for i in parent1.nodes if i != ActiveNode); 
            childnode2 = next(i for i in parent2.nodes if i != ActiveNode); 
            Childnodes = frozenset((childnode1,childnode2)) 
            child = next((q for q in Q if q.nodes == Childnodes));
            parent1.ScheduleOUT() 
            parent2.ScheduleOUT()
            child.ScheduleIN()
            Q_Ready = [q for q in connectedto[ActiveNode] if q.Qdpairs> 0]
 
    
    