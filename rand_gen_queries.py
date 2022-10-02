# https://blog.finxter.com/how-to-generate-random-graphs-with-python/

import random 
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
n = 6
p = 0.5
G = erdos_renyi_graph(n, p)
print(G.nodes)
print(G.edges)
# https://stackoverflow.com/questions/60528279/how-to-create-networkx-graph-with-randomly-assigned-node-labels

labels={}
# Iterate through all nodes
for x in range(len(G.nodes())):
  # Label node as either B or R until half of the nodes are labeled as either
  if(list(labels.values()).count('R') == len(G.nodes())/2):
    labels[x] = 'B'
  if(list(labels.values()).count('B') == len(G.nodes())/2):
    labels[x] = 'R'
  else:
    labels[x] = random.choice(['B', 'R'])
    
# list(newQuery.nodes(data="label"))

#### print each query to a file


