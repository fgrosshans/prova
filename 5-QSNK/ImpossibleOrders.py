import numpy as np

def BreakConflicts(R,G,Q,rank,qs):
    rng = np.random.default_rng()
    QandD = zip(*[(q.Qdpairs,q.demands) for q in Q]) # creating this generator and then iterating over it to use only one list comprehension
    actualQ = next(QandD)
    actualD = next(QandD)
    scheduled = -G@R
    actual_qp_q = np.hstack((actualQ,actualD))
    conflictIndices = list(np.flatnonzero(scheduled > actual_qp_q)) # Find which constraints are broken  
    if len(conflictIndices) == 2 and (rank[qs[conflictIndices[0]]] != rank[qs[conflictIndices[1]]]):
        breakpoint()
    doubleQ = np.hstack((actualQ,actualQ)) # This vector is useful (see later) in the offchance that more demands are scheduled to be served than there are available pairs 
    rng.shuffle(conflictIndices) # Tackle conflicts in a random order...
    conflictIndices.sort(key=lambda x: rank[qs[x]]) # Inside their rank.
    for i in conflictIndices: 
        to_reassign = doubleQ[i] + np.dot(G[i] == 1,R) # How many pairs are actually available, i.e. constraints + POSITIVE incoming scheduling
        concurrents = np.flatnonzero(G[i] == -1) # Those are the indices of the -1 terms, the ones generating conflict
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
    return R