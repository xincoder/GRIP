import numpy as np

class Graph():
	""" The Graph Representation
	Args:
		max_hop (int): the maximal distance between two connected nodes.
			max_hop = distance//length_of_object, 
			e.g., 30//15 means that we consider 30 feet range, each object is 15 feet. 
	How to use:
		1. graph = Graph(max_hop=1)
		2. A = graph.get_adjacency()
		3. A = code to modify A
		4. normalized_A = graph.normalize_adjacency(A)
	"""
	def __init__(self,
				 num_node = 70,
				 max_hop = 1
				 ):
		self.max_hop = max_hop
		self.num_node = num_node 

	def get_adjacency(self, A):
		# compute hop steps
		self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
		transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
		arrive_mat = (np.stack(transfer_mat) > 0)
		for d in range(self.max_hop, -1, -1):
			self.hop_dis[arrive_mat[d]] = d

		# compute adjacency
		valid_hop = range(0, self.max_hop + 1)
		adjacency = np.zeros((self.num_node, self.num_node))
		for hop in valid_hop:
			adjacency[self.hop_dis == hop] = 1
		return adjacency

	def normalize_adjacency(self, A):
		Dl = np.sum(A, 0)
		num_node = A.shape[0]
		Dn = np.zeros((num_node, num_node))
		for i in range(num_node):
			if Dl[i] > 0:
				Dn[i, i] = Dl[i]**(-1)
		AD = np.dot(A, Dn)

		valid_hop = range(0, self.max_hop + 1)
		A = np.zeros((len(valid_hop), self.num_node, self.num_node))
		for i, hop in enumerate(valid_hop):
			A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
		return A


if __name__ == '__main__':
	# np.set_printoptions(precision=2)
	data_path = '../train_data.pkl'
	import pickle
	import numpy as np 
	with open(data_path, 'rb') as reader:
		content = pickle.load(reader)
		print(content[0].shape, content[1].shape)
	test_adj = content[1][0]
	
	for row in test_adj.astype(str):#.astype(str):
		print('b', ' '.join(row))
	print('')

	graph = Graph(max_hop=2)
	A = graph.get_adjacency(test_adj)
	for row in A.astype(str):#.astype(str):
		print('a', ' '.join(row))
	print('')

	# object_loc = [3, 18, 19, 37]
	# remove_ind = set(range(39)) - set(object_loc)
	# # print(remove_ind)
	# for ind in remove_ind:
	# 	A[ind] = 0
	# 	A[:, ind] = 0

	# print('')
	# for row in A.astype(str):
	# 	print(' '.join(row))

	NA = graph.normalize_adjacency(A)
	# print(NA[0])
	# print(NA[1])
	for ind in range(len(NA)):
		for i, row in enumerate(NA[ind].astype(str)):
			print(i, ' '.join(row))
			# print(row)
		print('')

	