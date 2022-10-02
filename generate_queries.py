import networkx as nx
import random 
import matplotlib.pyplot as plt
import os

# randomly generate a node or an edge, then try adding and checking if query exists

# OR rand gen numNodes and numEdges from nodes. add until done

numQueries = 10000
numLabels = 3
maxNumNodes = 5

output_prefix = 'training queries/'
if not os.path.exists(output_prefix):
    os.makedirs(output_prefix)

list_of_queries = []

for querynum in range(0, numQueries):
    newQuery = nx.DiGraph()
    numNodes = random.randint(3, maxNumNodes)
    for nodeID in range(0, numNodes): 
        nodeLabel = random.randint(1,numLabels)
        newQuery.add_nodes_from([(nodeID, {"label": nodeLabel}) ])
    
    # list(newQuery.nodes(data="label"))
    
    # note that EVERY node has to be connected to at least one other node
    
    while True:
        numE = round ((numNodes * (numNodes - 1) ) / 2 )
        
        undirG = nx.Graph()  #undir b/c is_conn doesn't work for dir G
        for node in newQuery.nodes():
            newQuery.add_node(node)
            undirG.add_nodes_from([(node, {"label": nodeLabel}) ])
        for i in range(0, numE):
            head = 0
            tail = 0
            while head == tail:
                head = random.randint(0, numNodes - 1)
                tail = random.randint(0, numNodes - 1)
            edgeLabel = random.randint(0,1) # 50% chance is DE
            undirG.add_edge(head,tail, label = edgeLabel)
        
        # check if connected. if not, do over.
        if nx.is_connected(undirG):
            # add undir's edges to query
            for edge in list(undirG.edges(data="label")):
                newQuery.add_edge(edge[0], edge[1], label = edge[2])
            
            if newQuery not in list_of_queries: # check for duplicates
                break
        
    list_of_queries.append(newQuery)
    
    # fig = plt.figure()
    # pos = nx.spring_layout(newQuery)
    # nx.draw(newQuery, pos, with_labels=True)

    outFN = 'q' + str(querynum)
    output_path = output_prefix + '/' + outFN
    out_file = open(output_path + ".qvw", "w")
    
    out_file.write('q # 0\n')
    nodes = nx.get_node_attributes(newQuery,'label')
    for nodeID, vertex in enumerate(nodes.keys()):
        out_file.write("v " + str(nodeID) + " " + str(nodes[vertex]) + '\n' )
    edges = nx.get_edge_attributes(newQuery,'label')
    for e in edges:
        head = str(list(nodes.keys()).index(e[0]))
        tail = str(list(nodes.keys()).index(e[1]))
        out_file.write("e " + str(head) + " " + str(tail) + " " + str(edges[e]) + '\n' )
            
    out_file.close()

# node_labels = nx.get_node_attributes(qry, 'label')
# edges = nx.get_edge_attributes(qry,'label')
# colors = [qry[u][v]['color'] for u,v in edges]
# Ncolors = ['skyblue' for n in list(qry.nodes())]
# nx.draw(qry, pos, with_labels=True,node_size=800, font_weight = 'bold',
#         labels = node_labels, node_color = Ncolors, edge_color=colors)
# edge_labels = nx.get_edge_attributes(qry, 'label')
# nx.draw_networkx_edge_labels(qry, pos, edge_labels, font_weight = 'bold')
# plt.savefig(new_output_path+"_view"+ str(q) + ".jpg")









