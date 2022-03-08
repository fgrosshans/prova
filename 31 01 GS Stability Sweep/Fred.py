#! /usr/bin/python

# See smalltest() for usage

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

inf=float("inf")

class queueevent():
    def __init__(self, inputs, outputs, name=''):
        self.inputs  = inputs
        self.outputs = outputs
        self.name = name
    def __str__(self):
        return "queueevent '" + self.name +"' {" + ", ".join(x for x in self.inputs) \
                + "} -> {" \
                + ", ".join(y for y in self.outputs) + "}"  
                
class queueconstraints():
    """This class encodes the possible constraints : a set of queues, 
    a set of sources, a set of sinks, as well as a set of transitions taking 
    decrasing the input queues by one and increasing the output queues by one."""
    def __init__(self, source="source", sink="sink", initialsource=inf, initialsink=0):
        self.queues=dict()
        self.transitions=set()
        self.transitionsfrom=dict()
        self.transitionsto=dict()
        self.sinks=set()
    
    def addqueue(self, label, initval=0):
        "if queue already exists, do nothing"
        if label not in self.queues.keys(): 
            self.queues[label]=initval
            self.transitionsfrom[label]=[]
            self.transitionsto[label]=[]

    
    def addtransition(self, transition, strict=False):
        """transition is a queuevent object"""
        if not strict :
            def qaction(q, dic):
               self.addqueue(q)
               dic[q]=transition
        else :
            qaction = lambda q, dic : dic[q] #raises an error if q is not here
        for q in transition.inputs : qaction(q, self.transitionsfrom)
        for q in transition.outputs: qaction(q, self.transitionsto)
        
        self.transitions.add(transition)
        
    def addsink(self,q):
        self.sinks.add(q)

    def graph(self):
        G = nx.DiGraph()
        
        for t in self.transitions:
            G.add_edges_from(((inp, t.name) for inp in t.inputs))
            G.add_edges_from(((t.name, out) for out in t.outputs))
        return G
    def matrix(self, with_sinks=True):
        lqs = list(self.queues.keys())
        nsinks=len(self.sinks)
        lqs.sort(key=lambda q: q in self.sinks) 
            # lqs[-nsinks:] are the sinks
            # lqs[:-nsinks] are the other queues
        qi = {q:i for i,q in enumerate(lqs)}
        lts=list(self.transitions)
        ltns=[t.name for t in lts]
   #     ti = {t:i for i,t in enumerate(lts)}
        M = np.zeros((len(lqs), len(lts)))
        for j,t in enumerate(lts):
              for q in t.inputs : M[qi[q],j] -=1
              for q in t.outputs: M[qi[q],j] +=1
        if with_sinks : return M, lqs, ltns
        else : return M[:-nsinks,:] , lqs[:-nsinks], ltns
     #TODO a test functions
     #  * the 3 dicts should have the same keys (but source and sink ?)
     # Store/updaute graph


def edgelabel(u, v, sep=''):
     return sep.join(str(x) for x in (u,v))
class eswapnet():
    def __init__(self):
        self.QC=queueconstraints()
        self.G =nx.Graph()
        
    def addvertex(self,label):
        self.G.add_node(label)
        
    def addvertices(self,itr):
        self.G.add_nodes_from(itr)
        
    def addedge(self, x, y):
        x, y = sorted((x,y))
        label=edgelabel(x,y)
        self.G.add_edge(x, y, label=label)
        self.QC.addqueue(label)
                
    def addpath(self, path): #path is acyclic
        lp = len(path)
        self.QC.addsink(edgelabel(*sorted([path[0],path[-1]])))
        self.addedge(*path[-2:])
        for i in range(lp-2):
            self.addedge(*path[i:i+2])
            for j in range(i+1,lp-1):
                for k in range(j+1,lp):
                    a,c = sorted((path[i],path[k]))
                    b = path[j]
                    self.QC.addtransition(
                        queueevent([edgelabel(*sorted(tpl)) for tpl in ((a,b),(b,c))],
                                    [edgelabel(a,c)], 
                                    name=f'{a}[{b}]{c}' ) )
def smalltest():
    qnet = eswapnet()
    qnet.addpath('abc')
    qnet.addpath('bcd')
    qnet.addpath('abcd')
#    qnet.addpath('AbcD')
    
    M, qs, ts = qnet.QC.matrix(with_sinks=True)
    plt.figure()
    nx.draw(qnet.G, with_labels=True)
    
    QG=qnet.QC.graph()
    QG.graph['graph']={'rankdir': 'TB'}
#    pos=nx.planar_layout(QG)
    plt.figure()
    pos = nx.nx_pydot.pydot_layout(QG, prog='dot')
    nx.draw(QG, pos, with_labels=True, node_shape='s') #one of 'so^>v<dph8'
    return

