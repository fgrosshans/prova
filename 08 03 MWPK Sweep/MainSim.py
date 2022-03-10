### INPUT
LossParam = .9; # This is the eta from backpressure
Sources_rate = 200000; # Generation rate of the source, Hz.
t_step = 1e-6; # Length of the time step, s
time_steps = int(1e4); # Number of steps to simulate
memo_len=int(time_steps/5) # How many configurations should be memoized
beta = 1      # Demand weight in the scheduling calculation     


import GlobalFunctions as AllQueues
import MWsolve_gurobi as mw
import numpy as np
from Q_class import Queue
import Fred as fg

rng = np.random.default_rng()
def BreakConflicts(Ms,q,R): # This function solves the inter-node conflicts.
    scheduled = -Ms@R
    conflictIndices = np.flatnonzero(scheduled > q) # Find which constraints are broken  
    for i in conflictIndices: 
        to_reassign = q[i] + np.dot(Ms[i] == 1,R) # How many pairs are actually available, i.e. constraints + POSITIVE incoming scheduling
        concurrents = np.flatnonzero(Ms[i] == -1) # Those are the indices of the -1 terms, the ones generating conflict
        demandIndex = concurrents[-1] # Demand is the priority in breaking these conflicts
        R[demandIndex] = min(R[demandIndex],to_reassign)
        to_reassign-=R[demandIndex]
        concurrents = concurrents[:-1]
        rng.shuffle(concurrents) # After serving demand, the rest of the resources are assigned randomly.
        for j in concurrents:
            if to_reassign < 0:
                to_reassign = 0
            R[j] = min(R[j],to_reassign)
            to_reassign-=R[j]


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
    qnet.addpath('ABC')
    qnet.addpath('BCD')
    M, qs, ts = qnet.QC.matrix(with_sinks=True)
    
    ### Building the model 
    Q = [Queue(tq[0],tq[1],tran_prob=1) for tq in qs]
    
    nodeset = set()
    for tq in qs:
        nodeset = nodeset.union(set(tq))# Set of the nodes. May not be necessary now but will be useful going forward
    
    
    for i in range(len(M)):
        if 1 not in M[i]: # ASSUMPTION: physical queues do not receive swaps.
            Q[i].SetPhysical(Sources_rate,t_step)
    
    [q.SetService(BatchInput[q.nodes],t_step) for q in Q if q.nodes in BatchInput]
    
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
    
    ProbDim = len(Ms[1]) # Dimensionality of the problem
    R = np.zeros((ProbDim,time_steps)) # Initializing the R array, that will contain the R vector at each time step
    
    memo = dict() # Initializing the memory
    alpha = Sources_rate*t_step
    dem_arr_rates = [getattr(q,"PoissParam",0) for q in Q] # Already converted to timesteps^-1
    
    
    for Maintimestep in range(time_steps):
        Qt = np.array([q.Qdpairs for q in Q])
        Dt = np.array([q.demands for q in Q]) 
        A = AllQueues.Arrivals(Q)
        L = AllQueues.Losses(Q,Qt,LossParam)
        B = AllQueues.Demand(Q) 
        for nd in nodeset:
            qp_q, qp_h = mw.UpdateConstraints(Q,nd,qs,beta,Dt,dem_arr_rates,B,Ns,Qt,LossParam,L,alpha,A,Ms)
            partialsol, memo = mw.Schedule(qp_q, qp_G, qp_h, qp_A, qp_b,Dt,memo,memo_len,nd,ts,qs)
            R[:len(ts),Maintimestep] = R[:len(ts),Maintimestep] + partialsol[:len(ts)]
            for i in range(len(qs)):
                if nd in qs[i]:
                    flag = int((partialsol[len(ts) + i] !=0)) + int((R[len(ts) + i,Maintimestep] != 0))
                    if flag == 1: # Either this node was the only one to request consumption, or it was already requested by another node BUT NOT THIS ONE
                            R[len(ts) + i,Maintimestep] += partialsol[len(ts) + i]
                    elif flag == 2: # This node requested consumption, but the other node had already requested it -> break the tie
                        R[len(ts) + i,Maintimestep] = min(R[len(ts) + i,Maintimestep],partialsol[len(ts) + i]) 
        actualQ = Qt+A-L
        BreakConflicts(Ms, actualQ,R[:,Maintimestep])
        AllQueues.Evolve(Q,Ms,R[:,Maintimestep])
    
    
    D_final = [q.demands for q in Q]
    Q_final = [q.Qdpairs for q in Q]
    Tot_dem_rate = sum(BatchInput.values())
    unserved = sum(D_final)/(t_step*time_steps*Tot_dem_rate) #Unserved demands at the end divided by an approximation of the total incoming demand
    if unserved >= 0.15:
        to_store = tuple(zip(*BatchInput.items()))
        memoDict[to_store] = (unserved, Q_final, D_final) 
    return unserved, Q_final, D_final #, violations

