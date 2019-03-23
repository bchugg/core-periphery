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

def sbm_figure(scores, N, title, saveas):
	# Graph of scores versus node index
	plt.bar(range(N), scores)
	plt.xlabel('Node index')
	plt.ylabel('Core score')
	plt.title(title)
	plt.savefig('plots/'+saveas)


def core_score_vs_index(k, p, size_c, size_p):
	# Generate core score vs index plots for all methods using SBM 
	G = sbm(k, p, size_c, size_p)
	N = size_c + size_p

	t_suffix=', N='+str(N)+', k='+str(k)
	f_suffix='_sbm_kappa='+str(k)+'.png'

	plt.figure(1) # Random Walk persistences
	[profile, persistences] = periphery_profile(G) # Random walk
	sbm_figure(list(map(lambda i: persistences[profile.index(i)], range(N))), N, 
		'Persistence'+t_suffix, 'pprofile'+f_suffix)

	plt.figure(2) # Path core
	sbm_figure(path_core(G), N, "Path-Core"+t_suffix, 'pathcores'+f_suffix)

	plt.figure(3) # Degree
	sbm_figure(list(map(lambda x: G.degree(x), range(N))), N, "Degree"+t_suffix, 
		'degrees'+f_suffix)

	plt.figure(4) # Betweenness Centralities
	C = nx.betweenness_centrality(G)
	sbm_figure([C[i] for i in range(N)], N, 'Betweenness'+t_suffix, 'betweenness'+f_suffix)

	plt.show()




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
	N = nx.number_of_nodes(G)

	plt.figure(1)
	generate_figure(G, range(N), path_core(G),
		"Colormap of Path-Core Scores", 'cm_pathcore_'+name+'.png')

	# Test pathcore against betweenness centrality
	C = nx.betweenness_centrality(G)
	plt.figure(2)
	generate_figure(G, range(N), [C[i] for i in range(N)], 
		"Colormap of Betweenness Centrality Scores", 'cm_between_'+name+'.png')

	plt.show()






# Run tests

# Block model parameters
p = 0.25
size_c, size_p = 30, 30

# for k in [1.3, 1.5, 1.8, 2]:
# 	core_score_vs_index(k, p, size_c, size_p)


G1 = nx.karate_club_graph()
G2 = sbm(2, 0.25, 15,15) 

# SBM tests
#test_sbm(prob_cp, prob_cc, prob_pp, size_c, size_p)

# Random Walk tests
#test_random_walk(G1, 'karate')
#test_random_walk(G2, 'strong_sbm')

# Tests for path-core
#test_pathcore(G1, 'karate')
#test_pathcore(G2, 'strong_sbm')

# Test Holme coefficient
coeff = holme_coefficient(G2, 50)
print(coeff)









