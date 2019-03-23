import networkx as nx
import matplotlib.pyplot as plt
from cp_methods import *


# Helper Methods for testing 
def generate_figure(G, nodelist, colorlist, title="", saveas=""):
	# Generate graph figure using Kamada Kawai Layout
	pos = nx.kamada_kawai_layout(G)
	nx.draw_networkx_edges(G, pos)
	nodes = nx.draw_networkx_nodes(G, pos, node_list=nodelist,
		node_color=colorlist, cmap=plt.cm.jet)
	plt.colorbar(nodes)
	plt.axis('off')
	plt.title(title)
	plt.savefig('plots/'+saveas)


def test_random_walk(G, name):
	# Test random walkers core-periphery profile method 
	# name is name of graph for saving purposes
	[profile, persistences] = periphery_profile(G)
	N = nx.number_of_nodes(G)
	
	plt.figure(1)
	generate_figure(G, range(N), list(map(lambda x: G.degree(x), range(N))), 
				"Colormap of Degree Structure", 'cm_degree_'+name+'.png')

	plt.figure(2)
	generate_figure(G, range(N), list(map(lambda i: persistences[profile.index(i)], range(N))), 
				"Colormap of CP Structure using Random Walkers", 'cm_rw_'+name+'png')

	plt.show()


def test_sbm(prob_cp, prob_cc, prob_pp, size_c, size_p):
	# test sbm method
	SBM = sbm(prob_cp, prob_cc, prob_pp, size_c, size_p)
	N = size_c + size_p
	A = nx.to_numpy_matrix(SBM)


	plt.figure(1) 
	colors = ['g' for i in range(size_c)] + ['b' for i in range(size_p)]
	nx.draw_networkx(SBM, node_color=colors, node_size=200, with_labels=False)
	plt.title("Graph generated using SBM")
	plt.axis('off')
	plt.savefig('plots/sbm_graph.png')

	plt.matshow(A)
	plt.title("Adjacency matrix of SBM")
	plt.axis('off')
	plt.savefig("plots/SBM_adjacency.png")

	plt.show()


def test_pathcore(G, name):
	# Test path core method
	# name is name of graph for saving purposes
	generate_figure(G, range(nx.number_of_nodes(G)), path_core(G), 
		"Colormap of Path-Core Scores", 'cm_pathcore_'+name+'.png')
	plt.show()



# Run tests


#prob_cp, prob_cc, prob_pp, size_c, size_p = 0.7, 0.8, 0.2, 15, 25
prob_cp, prob_cc, prob_pp, size_c, size_p = 0.5, 0.75, 0.3, 15, 25


G1 = nx.karate_club_graph()
G2 = sbm(prob_cp, prob_cc, prob_pp, size_c, size_p) 

# SBM tests
#test_sbm(prob_cp, prob_cc, prob_pp, size_c, size_p)

# Random Walk tests
#test_random_walk(G1, 'karate')
#test_random_walk(G2, 'strong_sbm')

# Tests for path-core
#test_pathcore(G1, 'karate')
test_pathcore(G2, 'weak_sbm')









