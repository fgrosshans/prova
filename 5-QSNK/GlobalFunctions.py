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
    Q_t = Qt - L + A
    D_t = np.array(Dt) + np.array(B)
    
    D_t1 = D_t - R[-len(Qt):]
    Q_t1 = Q_t + Ms@R
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