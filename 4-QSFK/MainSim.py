import GlobalFunctions as AllQueues
import Quadsolve_gurobi as qp
import numpy as np
from Q_class import Queue
import Fred as fg


with open("inputs.in") as f: # This imports the simulation parameters from the inputs.in file
    exec(f.read())


def Sim(BatchInput,memoDict):   
    flatInput = tuple(zip(*BatchInput.items())) # List of tuples
    #memoDict = dict() # Uncomment to DISABLE memoization
    for i in memoDict.keys():
        flatMemo = i
        if flatInput[1][0] >= flatMemo[1][0] and flatInput[1][1] >= flatMemo[1][1]:
            output = memoDict[i]
            #return output
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
    # Defining the building blocks of the optimization problem.
    # From now on, every variable with an s in front is to be read as \tilde{x}
    
    r_matrix = -np.identity(len(Q)) #Matrix for the demand part
    Ms = np.concatenate((M,r_matrix),1) # Full "Big M" matrix
    Ns = np.concatenate((np.zeros((len(M),len(M[0]))),r_matrix),1) # Auxiliary matrix analogous to big M but for demands
    qp_P = (Ms.T)@Ms + 2*beta*(Ns.T)@Ns # Matrix for the quadratic term
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
    
    debug = np.zeros((time_steps,len(Q)))
    for Maintimestep in range(time_steps):
        Qt = np.array([q.Qdpairs for q in Q])
        Dt = np.array([q.demands for q in Q]) # Snapping a picture of the system at time t
        A = AllQueues.Arrivals(Q)
        L = AllQueues.Losses(Q,Qt,LossParam)
        Bt = AllQueues.Demand(Q)
        qp_q, qp_h = qp.UpdateConstraints(beta,Dt,Bt,Ns,Qt,L,A,Ms)
        debug[Maintimestep,:] = [q.Qdpairs for q in Q]
        R[:,Maintimestep], memo = qp.QuadLyap(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b,Dt ,memo,memo_len)
        AllQueues.Evolve(Q,Ms,R[:,Maintimestep]) # Note to me: this method DOES evolve demands, it just does it directly on the q objects
        
    ### OUTPUT
    D_final = [q.demands for q in Q]
    Q_final = [q.Qdpairs for q in Q]
    Tot_dem_rate = sum(BatchInput.values())
    unserved = sum(D_final)/(t_step*time_steps*Tot_dem_rate) #Unserved demands at the end divided by an approximation of the total incoming demand
    
    
    if unserved >= 0.18:
        to_store = tuple(zip(*BatchInput.items()))
        memoDict[to_store] = (unserved, Q_final, D_final) 
    return (unserved, Q_final, D_final) #, violations
