# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:42:02 2021

@author: kobienkung
"""

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

#_________________________________________________________
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2,3])
G.add_nodes_from(['u', 'v'])
G.nodes()

G.add_edge(1,2)
G.add_edge('u', 'v')
G.add_edge('u', 'w')
G.add_edges_from([(1,3),(1,4),(1,5),(1,6)])
G.edges()

G.remove_node(2)
G.remove_nodes_from([4,5])
G.remove_edge(1,3)
G.remove_edges_from([(1,2),('u','w')])

G.number_of_nodes()
G.number_of_edges()


#_________________________________________________________
G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')

G.degree()
G.degree()[33]
G.degree(33)

G.number_of_nodes()
G.number_of_edges()


#_________________________________________________________
bernoulli.rvs(p=0.2) #return 0=False, 1=True

N = 20
p = 0.2
# Create empty graph
# add all N nodes in the graph
# loop over all pairs of nodes
    # add an edge with prob p

def er_graph(N,p):
    '''Generate an Erdős-Rényi graph.'''
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1, node2)
    return G
nx.draw(er_graph(50, 0.08), node_size=40, node_color='gray')
# plt.savefig('er1.pdf')


nx.erdos_renyi_graph() #Built in Function


#_________________________________________________________
def plot_degree_distribution(G):
    plt.hist(list(d for n,d in G.degree()), histtype='step')
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree Distribution')

G1 = er_graph(500, 0.08)
plot_degree_distribution(G1)
G2 = er_graph(500, 0.08)
plot_degree_distribution(G2)
G3 = er_graph(500, 0.08)
plot_degree_distribution(G3)


#_________________________________________________________
A1 = np.loadtxt('adj_allVillageRelationships_vilno_1.csv', delimiter=',')
A2 = np.loadtxt('adj_allVillageRelationships_vilno_2.csv', delimiter=',')

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print('Number of nodes: %d' % G.number_of_nodes())
    print('Number of edges: %d' % G.number_of_edges())
    print('Average degree: %.2f' % np.mean([d for n,d in G.degree()]))

basic_net_stats(G1)
basic_net_stats(G2)

plot_degree_distribution(G1)
plot_degree_distribution(G2)


#_________________________________________________________
gen = (G1.subgraph(c) for c in nx.connected_components(G1))
g = gen.__next__()
g.number_of_nodes()
len(gen.__next__()) # many times/ return No. of nodes
len(G1) # return No. of nodes
G1.number_of_nodes()

# largest connected component
G1LCC = max((G1.subgraph(c) for c in nx.connected_components(G1)), key = len)
G2LCC = max((G2.subgraph(c) for c in nx.connected_components(G2)), key = len)

len(G1LCC)
G1LCC.number_of_nodes() / G1.number_of_nodes()

len(G2LCC) 
G2LCC.number_of_nodes() / G2.number_of_nodes()


plt.figure()
nx.draw(G1LCC, node_color='red', edge_color='gray', node_size=20)
#plt.savefig('village1.pdf')

plt.figure()
nx.draw(G2LCC, node_color='green', edge_color='gray', node_size=20)
#plt.savefig('village2.pdf')






