# sys
import os
import sys
import numpy as np
import random
import pickle
# import scipy.io as scp	

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from layers.graph import Graph

# visualization
import time

# operation
# from . import tools

class Feeder(torch.utils.data.Dataset):
	""" Feeder for skeleton-based action recognition
	Arguments:
		data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
	"""

	def __init__(self, data_path, graph_args={}, train_val_test='train'):
		'''
		train_val_test: (train, val, test)
		'''
		self.data_path = data_path
		self.load_data()

		total_num = len(self.all_feature)
		# equally choose validation set
		train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
		val_id_list = list(set(list(range(total_num))) - set(train_id_list))

		# # last 20% data as validation set
		# train_id_list = list(range(int(total_num*0.8)))
		# val_id_list = list(range(int(total_num*0.8), total_num))

		if train_val_test.lower() == 'train':
			self.all_feature = self.all_feature[train_id_list]
			self.all_adjacency = self.all_adjacency[train_id_list]
			self.all_mean_xy = self.all_mean_xy[train_id_list]
			# self.all_feature = self.all_feature
			# self.all_adjacency = self.all_adjacency
			# self.all_mean_xy = self.all_mean_xy
		elif train_val_test.lower() == 'val':
			self.all_feature = self.all_feature[val_id_list]
			self.all_adjacency = self.all_adjacency[val_id_list]
			self.all_mean_xy = self.all_mean_xy[val_id_list]

		self.graph = Graph(**graph_args) #num_node = 70,max_hop = 1
		# self.ori_Adjacency = self.graph.get_adjacency()


	def load_data(self):
		with open(self.data_path, 'rb') as reader:
			# Training (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
			[self.all_feature, self.all_adjacency, self.all_mean_xy]= pickle.load(reader)
			

	def __len__(self):
		return len(self.all_feature)

	def __getitem__(self, idx):
		now_feature = self.all_feature[idx]
		now_mean_xy = self.all_mean_xy[idx]

		now_adjacency = self.graph.get_adjacency(self.all_adjacency[idx])
		now_A = self.graph.normalize_adjacency(now_adjacency)
		
		return now_feature, now_A, now_mean_xy


if __name__ == "__main__":
	feeder = Feeder(data_path='/Users/xincoder/Documents/Dataset/ApolloScape/Baidu/120object_all_objects_10meters/train_data.pkl', graph_args={'num_node':120,'max_hop':2})
	# feeder = Feeder(data_path='/data/xincoder/ApolloScape/Baidu/train_data.pkl', graph_args={'num_node':120,'max_hop':2}, train_val_test='test')
	min_x = 99999999
	max_x = -99999999
	min_y = 99999999
	max_y = -9999999999
	for data, now_A,_ in feeder:
		# new_data = data.copy()
		new_mask = (data[3:5, 1:]!=0) * (data[3:5, :-1]!=0) * (data[2,1:]==4)
		data[3:5, 1:] = (data[3:5, 1:] - data[3:5, :-1]) * new_mask
		data[3:5, 0] = 0	

		# print(np.shape(data), np.shape(now_A))
		# print(np.min(data[0]), np.max(data[1]), np.min(data[1]), np.max(data[1]))
		# if np.min(data[3]) == -72.303:
		# 	print(data[3:5])

		# data[3][np.where(data[3]==-72.303)] = 0
		# data[3][np.where(data[3]==34.31999999999999)] = 0
		min_x = min(min_x, np.min(data[3]))
		max_x = max(max_x, np.max(data[3]))
		min_y = min(min_y, np.min(data[4]))
		max_y = max(max_y, np.max(data[4]))
		print(min_x, max_x, min_y, max_y, set(np.abs(data[3:5]).flatten().astype(int)))


	# # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
	# for data, now_A, _ in feeder:
	# 	new_data = data.copy()
	# 	new_mask = (new_data[3:5, 1:]!=0) * (new_data[3:5, :-1]!=0)
	# 	new_data[3:5, 1:] = (new_data[3:5, 1:] - new_data[3:5, :-1]) * new_mask
	# 	new_data[3:5, 0] = 0
	# 	# print(data.shape)
	# 	for i in range(120):
	# 		x = data[3, :, i]
	# 		y = data[4, :, i]
	# 		print(i)
	# 		print('x', ' '.join(x.astype(str)))
	# 		print('y', ' '.join(y.astype(str)))
			
	# 		x = new_data[3, :, i]
	# 		y = new_data[4, :, i]
	# 		print('x', ' '.join(x.astype(str)))
	# 		print('y', ' '.join(y.astype(str)))

	# 		print('')

