import GlobalFunctions as AllQueues
import Quadsolve_gurobi as qp
import numpy as np
from Q_class import Queue
import Fred as fg
from ImpossibleOrders import BreakConflicts

with open("inputs.in") as f:
    exec(f.read())
      
def Sim(BatchInput,memoDict):   
    flatInput = tuple(zip(*BatchInput.items())) # List of tuples
    # memoDict = dict() # Uncomment to DISABLE memoization
    for i in memoDict.keys():
        flatMemo = i
        if flatInput[1][0] >= flatMemo[1][0] and flatInput[1][1] >= flatMemo[1][1]:
            output = memoDict[i]
            return output
    ############ INPUT
    
    qnet = fg.eswapnet()
    for rt in routes:
        qnet.addpath(rt)

    M, qs, ts = qnet.QC.matrix(with_sinks=True)
    
    ############ PARAMETERS
    
    memo = dict()
    
    
    Q = [Queue(tq[0],tq[1],tran_prob=1) for tq in qs]
    violations = 0
    violationsPre = 0
    nodeset = set()
    for tq in qs:
        nodeset = nodeset.union(set(tq))# Set of the nodes. May not be necessary now but will be useful going forward
    
    
    
    [q.SetPhysical(ArrRates[q.nodes],t_step) for q in Q if q.nodes in ArrRates]
    [q.SetService(BatchInput[q.nodes],t_step) for q in Q if q.nodes in BatchInput]
    
    r_matrix = -np.identity(len(Q))
    # From now on, every variable number with an s in front is to be read as x_tilde
    Ms = np.concatenate((M,r_matrix),1)
    
    Ns = np.concatenate((np.zeros((len(M),len(M[0]))),r_matrix),1)
    
    
    ProbDim = len(Ms[1]) # Dimensionality of the problem, i.e. per-node rates+ #queues
    R = np.zeros((ProbDim,time_steps)) # Initializing the R array, that will contain the R vector at each time step
    
    #Instantiate the problem here
    ###Ranking the queues: this is useful for conflict management
    to_rank = qnet.QC.transitions
    rank = {i:0 for i in qs}
    
    for i in to_rank:
        rank[i.outputs[0]] = max(rank[i.inputs[0]]+1,rank[i.inputs[1]]+1,rank[i.outputs[0]])
    for i in to_rank: # THIS IS NOT AN ERROR! The code needs to comb through the list twice in order to assign correct ranks
        rank[i.outputs[0]] = max(rank[i.inputs[0]]+1,rank[i.inputs[1]]+1,rank[i.outputs[0]])
    
    
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
    alpha = [getattr(q,"GenPParam",0) for q in Q] # Already converted to timesteps^-1
    if PhotonLifeTime == "Inf":
        LossParam = 1
    else:
        LossParam = 1 - t_step/PhotonLifeTime
    
    for Maintimestep in range(time_steps):
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
        if AllQueues.CheckActualFeasibility(Ms,Ns,R[:,Maintimestep],Qt,Dt,L,A,B):
            violationsPre+=1
            R[:,Maintimestep] = BreakConflicts(R[:,Maintimestep],qp_G,Q,rank,qs)
        
        if AllQueues.CheckActualFeasibility(Ms,Ns,R[:,Maintimestep],Qt,Dt,L,A,B):
            violations+=1
        AllQueues.Evolve(Q,Ms,R[:,Maintimestep])
    
    if quiet == False:
        print(f"Impossible orders: {violationsPre}/{time_steps}. After correction: {violations}/{time_steps}")
    D_final = [q.demands for q in Q]
    Q_final = [q.Qdpairs for q in Q]
    Tot_dem_rate = sum(BatchInput.values())
    unserved = sum(D_final)/(t_step*time_steps*Tot_dem_rate) #Unserved demands at the end divided by an approximation of the total incoming demand
    if unserved >= 0.17:
        to_store = tuple(zip(*BatchInput.items()))
        memoDict[to_store] = (unserved, Q_final, D_final) 
    return unserved, Q_final, D_final
