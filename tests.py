import networkx as nx
import matplotlib.pyplot as plt
from cp_methods import periphery_profile

G = nx.karate_club_graph()

[profile, persistences] = periphery_profile(G)
print("Profile ", str(profile))
print("Persistences", str(persistences))

# Sort by Degrees
N = nx.number_of_nodes(G)
degrees = list(map(lambda x: [x,G.degree(x)], range(N)))
degrees_sorted =  sorted(degrees, key=lambda x: x[1])
nodes_by_degree = list(map(lambda x: x[0], degrees_sorted))
#print(nodes_by_degree)

iters = 200

plt.figure(1)
#pos = nx.spring_layout(G, iterations=iters)
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx_edges(G, pos)
nodes = nx.draw_networkx_nodes(G, pos, node_list=nodes_by_degree, 
	node_color=list(map(lambda x: G.degree(x), range(N))), cmap=plt.cm.jet)
plt.colorbar(nodes)
plt.axis('off')
plt.title("Colormap of Degree Structure")
plt.savefig('cm_degree_karate.rw')


plt.figure(2)
nx.draw_networkx_edges(G, pos)
nodes = nx.draw_networkx_nodes(G, pos, node_list=profile,
	node_color=persistences, cmap=plt.cm.jet)
plt.colorbar(nodes)
plt.axis('off')
plt.title("Colormap of CP Structure using Random Walkers")
plt.savefig('cm_rw_karate.png')


plt.show()


