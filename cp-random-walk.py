import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()



# Return periphery profile of undirected graph G
def periphery_profile(G):

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

	# Initialize P with node with lowest weight
	degrees = [G.degree(i) for i in range(N)]
	Profile.append(degrees.index(min(degrees)))
	
	while len(Profile) < N:
		min_persistence = float("inf")
		min_index = 0
		min_p_num = 0
		for i in range(N):
		# Only check i if it's not in P already
			if not (i in Profile):
				# Calculate alpha(P+i), beginning with numerator change
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


[Profile, Persistences] = periphery_profile(G)
print("Profile ", str(Profile))
print("Persistences", str(Persistences))
