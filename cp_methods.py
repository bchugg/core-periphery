# Small library implementing various core-periphery detection algorithms 
# and related functions. 
# Created for Networks Mini-Project, University of Oxford.  

import numpy as np
import networkx as nx
import copy
from functools import reduce

def sbm(k, p, size_c, size_p):
	# Return graph drawn from stochastic block model 
	# with size_c nodes in core, size_p in periphery, 
	# and probabilities 
	# - k*p between core and periphery nodes
	# - k^2*p between core nodes 
	# - p between peripheral nodes
	# Returns an unweighted and undirected matrix

	N = size_c + size_p
	G = nx.Graph()
	G.add_nodes_from(range(N)) # first size_c nodes will be core

	for i in range(N):
		for j in range(i+1,N):
			r = np.random.random()
			if i < size_c and j < size_c: # Both in core
				if r < k*k*p: 
					G.add_edge(i,j)
			elif i < size_c:			# One in core, on in periphery
				if r < k*p:
				 	G.add_edge(i,j)
			else:						# Both in periphery 
				if r < p:
					G.add_edge(i,j)

	return G





def periphery_profile(G):
# Return periphery profile and persistences 
# of undirected and unweighted graph G. Returns as 
# tuple [profile, persistences]

	A = nx.adjacency_matrix(G)
	N = nx.number_of_nodes(G)
	M = nx.number_of_edges(G)
	Profile = []
	Persistences = [0]

	# Compute stationary distribution
	pi = [G.degree(i)/float(2*M) for i in range(N)]

	# Compute transition matrix
	T = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			T[i,j] = A[i,j] / float(G.degree(i))

	# Track current persistence probability
	# Separate numerator and denominator for easy updating
	alpha_num = 0
	alpha_denom = 0

	# Initialize Profile with node with lowest weight
	degrees = [G.degree(i) for i in range(N)]
	Profile.append(degrees.index(min(degrees)))
	
	while len(Profile) < N:
		min_persistence = float("inf")
		min_index = 0
		min_p_num = 0
		for i in range(N):
		# Only check i if it's not in Profile already
			if not (i in Profile):
				# Calculate alpha(Propfile+i), beginning with numerator change
				num = np.sum([pi[i]*T[i,k] + pi[k]*T[k,i] for k in Profile])
				# and denominator
				denom = pi[i]

				persistence = (alpha_num + num) / float(alpha_denom + denom)
				if persistence < min_persistence:
					min_persistence = persistence
					min_p_num = num
					min_index = i
		
		# Update profile
		Profile.append(min_index)
		alpha_num += min_p_num
		alpha_denom += pi[min_index]
		Persistences.append(alpha_num / float(alpha_denom))

	return [Profile, Persistences] 


def sp_coeff(G,i,j,k):
	# Compute shortest paths coefficient for node i and edge (j,k) in G
	# Return 0 if no path between j and k
	coeff = 0
	try:
		paths = [p for p in nx.all_shortest_paths(G,j,k)] # All shortest paths between j and k
		paths_with_i = len(list(filter(lambda x: i in x, paths)))
		coeff = paths_with_i / float(len(paths))
	finally: 
		return coeff



def path_core(G):
	# Compute the path-core for each vertex in the undirected graph G
	# Return array of path cores

	N = nx.number_of_nodes(G)
	path_cores = []
	
	for i in range(N):
		# Compute path score of i
		G_without_i = copy.deepcopy(G)
		G_without_i.remove_node(i)
		score  = 0
		for e in [e for e in G_without_i.edges()]:
			G.remove_edge(*e)
			score += sp_coeff(G, i, e[0], e[1])
			G.add_edge(*e)

		path_cores.append(score)

	return path_cores


def CC(U, closeness):
	# Calculate closeness of the subgraph U
	if nx.is_empty(U): return 0
	
	for i in U:
		if closeness[i] == 0:
			U.remove_node(i)
	
	denom = sum([1/float(closeness[i]) for i in U.nodes()])
	return len(U) / float(denom)

def max_CC(G, max_degree, N):
	# Calculate k core U which maximizes CC(U), and return CC(U) / CC(V) 

	# Calculate closeness for each vertex
	C = nx.closeness_centrality(G)
	closeness = [C[i] for i in range(N)]

	max_score =  max(list(map(lambda k: CC(nx.k_core(G,k), closeness), range(max_degree))))
	return max_score / float(CC(G, closeness))

		
def holme_coefficient(G, T):
	# Calculate holme coefficient for a graph G
	# - T is number of trials taken to be the average for null model

	N = nx.number_of_nodes(G)
	deg_sequence = list(map(lambda i: G.degree(i), range(N)))
	max_degree = max(deg_sequence)

	# Compute average coefficient for graphs with same degree sequence
	avg_coeff = 0
	for t in range(T):
		# Create graph with same degree sequence; remove multiedges and self loops
		H = nx.Graph(nx.configuration_model(deg_sequence))
		H.remove_edges_from(nx.selfloop_edges(H))
		avg_coeff += max_CC(H, max_degree, N)

	avg_coeff /= float(T)

	return max_CC(G, max_degree, N) - avg_coeff

def max_DC(G):
	# Calculate degree coefficient of G

	# sort by degrees, in reverse
	sorted_degs = sorted(G.degree(), key=lambda x: x[1], reverse=True)

	max_score = 0
	for k in range(1,nx.number_of_nodes(G)):
		# average degree of first k nodes
		avg_core = sum(i[1] for i in sorted_degs[:k]) / float(k)
		# Create subgraph with these first k nodes
		H = copy.deepcopy(G)
		H.remove_nodes_from([i[0] for i in sorted_degs[:k]])
		H_degrees = [H.degree(i) for i in H.nodes()]
		avg_per = 1 + reduce(lambda x,y: x + y, H_degrees, 0) / float(len(H_degrees))
		score = avg_core / float(avg_per)
		# Update if maximum seen so far
		if score > max_score: max_score = score

	return max_score



def degree_coefficient(G, T):
	# Calculate average degree coefficient of G
	# - T is number of trials taken to be the average for null model

	deg_sequence = list(map(lambda i: G.degree(i), range(nx.number_of_nodes(G))))
	max_degree = max(deg_sequence)

	# Compute average coefficient for graphs with same degree sequence
	avg_coeff = 0
	for t in range(T):
		# Create graph with same degree sequence; remove multiedges and self loops
		H = nx.Graph(nx.configuration_model(deg_sequence))
		H.remove_edges_from(nx.selfloop_edges(H))
		avg_coeff += max_DC(H)

	if not T==0: avg_coeff /= float(T)

	return max_DC(G) - avg_coeff 







	
	



	











