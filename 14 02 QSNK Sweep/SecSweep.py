#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from MainSim import Sim
import numpy as np
from time import time
import matplotlib.pyplot as plt

SPair_1 = ("A","C")
SPair_2 = ("B","D")

npoints = 21

DemRates1 = np.linspace(1,200000,npoints)
DemRates2 = np.linspace(1,200000,npoints)

Output = np.zeros((len(DemRates1),len(DemRates2)))


t1 = time()
for r1 in DemRates1:
    for r2 in DemRates2:
        SimInput = {frozenset(SPair_1) : r1,
                    frozenset(SPair_2) : r2}
        Output[np.where(DemRates1 == r1),np.where(DemRates2 == r2)] = Sim(SimInput)
        t2 = time()-t1
        print(f"Iteration {r1/1000:.2f}-{r2/1000:.2f} complete. Elapsed time: {np.floor(t2/60)} minutes and {(t2/60-np.floor(t2/60))*60:.2f} seconds")


Output = np.flipud(Output)

plt.figure(1)
plt.imshow(Output*100,vmin=0,vmax=100)
plt.colorbar()
xlabels = ['{:.2f}'.format(i) for i in DemRates1/1000]
ylabels = ['{:.2f}'.format(i) for i in np.flip(DemRates2)/1000]
plt.xticks(range(len(DemRates1)),xlabels,rotation=70)
plt.yticks(range(len(DemRates2)),ylabels)
plt.xlabel(f"Average demand rate across pair {SPair_1[0]}-{SPair_1[1]}, kHz")  
plt.ylabel(f"Average demand rate across pair {SPair_2[0]}-{SPair_2[1]}, kHz")
plt.title("% Unserved demands, QSPK")   
plt.show()
