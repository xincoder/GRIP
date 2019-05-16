import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq
import numpy as np 

class Model(nn.Module):
	def __init__(self, in_channels, pred_length, graph_args, edge_importance_weighting, **kwargs):
		super().__init__()

		# load graph
		self.graph = Graph(**graph_args)
		# A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
		# print(A.shape)
		# A = self.graph.get_adjacency().copy()
		# A = self.graph.normalize_adjacency(A)
		# A = self.graph.A
		# print(np.shape(A))
		# A = self.graph.normalize_adjacency(A)
		# self.register_buffer('A', A)
		# {'max_hop':1, 'num_node':70}
		A = np.zeros((graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node']))

		# build networks
		spatial_kernel_size = np.shape(A)[0]
		temporal_kernel_size = 9 #5 # 3
		kernel_size = (temporal_kernel_size, spatial_kernel_size)
		self.data_bn = nn.BatchNorm1d(in_channels * np.shape(A)[1])
		# self.st_gcn_networks = nn.ModuleList((
		# 	Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
		# 	Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(64, 128, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(128, 128, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(128, 128, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(128, 256, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(256, 256, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(256, 256, kernel_size, 1, **kwargs),
		# ))

		self.st_gcn_networks = nn.ModuleList((
			Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			Graph_Conv_Block(64, 128, kernel_size, 1, **kwargs),
			Graph_Conv_Block(128, 128, kernel_size, 1, **kwargs),
			Graph_Conv_Block(128, 128, kernel_size, 1, **kwargs),
			Graph_Conv_Block(128, 256, kernel_size, 1, **kwargs),
			Graph_Conv_Block(256, 256, kernel_size, 1, **kwargs),
			Graph_Conv_Block(256, 256, kernel_size, 1, **kwargs),
			Graph_Conv_Block(256, 512, kernel_size, 1, **kwargs),
		))

		# self.st_gcn_networks = nn.ModuleList((
		# 	Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
		# 	Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(64, 128, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(128, 128, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(128, 128, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(128, 256, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(256, 256, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(256, 256, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(256, 512, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(512, 512, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(512, 512, kernel_size, 1, **kwargs),
		# 	Graph_Conv_Block(512, 1024, kernel_size, 1, **kwargs),
		# ))

		# initialize parameters for edge importance weighting
		if edge_importance_weighting:
			self.edge_importance = nn.ParameterList(
				[nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
				)
		else:
			self.edge_importance = [1] * len(self.st_gcn_networks)

		# fcn for prediction
		# self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
		self.pred_length = pred_length
		self.num_node = num_node = self.graph.num_node
		out_dim_per_node = 2 #(x, y) coordinate
		# self.seq2seq = Seq2Seq(input_size=num_node*(256), hidden_size=num_node*out_dim_per_node, pred_length=pred_length, num_layers=2, dropout=0.5, isCuda=True)
		self.seq2seq = Seq2Seq(input_size=num_node*(512), hidden_size=num_node*out_dim_per_node, pred_length=pred_length, num_layers=2, dropout=0.5, isCuda=True)
		# self.seq2seq = Seq2Seq(input_size=num_node*256, hidden_size=num_node*out_dim_per_node, pred_length=pred_length, num_layers=2, dropout=0.5, isCuda=True)
		# self.seq2seq = Seq2Seq(input_size=self.graph.num_node*256, hidden_size=num_node*out_dim_per_node, pred_length=pred_length, num_layers=2, dropout=0.5, isCuda=True)

	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 2, 3, 1).contiguous()
		now_feat = now_feat.view(N, T, V*C) 
		return now_feat

	def reshape_from_lstm(self, predicted):
		# predicted (N, T, V*C)
		N, T, _ = predicted.size()
		now_feat = predicted.view(N, T, self.num_node, -1) # (N, T, V, C) -> (N, C, T, V)
		now_feat = now_feat.permute(0, 3, 1, 2).contiguous()
		return now_feat

	def forward(self, pra_x, pra_A, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		# data normalization
		N, C, T, V = pra_x.size()
		x = pra_x.permute(0, 3, 1, 2).contiguous()
		x = x.view(N, V * C, T)
		x = self.data_bn(x)
		x = x.view(N, V, C, T)
		x = x.permute(0, 2, 3, 1).contiguous()
		x = x.view(N, C, T, V)

		# print(x.data)
		
		# forwad
		for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
			# print(x.shape, pra_A.shape, importance.shape)
			x, _ = gcn(x, pra_A[0] * importance)
			# print('importance', np.sum(importance), self.edge_importance)
		
		# print(x)
		# # forwad
		# for gcn in self.st_gcn_networks:
		# 	# print(x.shape, pra_A.shape, importance.shape)
		# 	x, _ = gcn(x, pra_A[0])
		# print(x.data)

		# print(x.shape, pra_x.shape)
		# x = torch.cat((x, pra_x), dim=1)

		# now x shape: (N, C, T, V) = (N, 256, 4, 39)
		# prepare for seq2seq lstm model
		graph_conv_feature = self.reshape_for_lstm(x)
		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2)

		if pra_teacher_forcing_ratio>0 and type(pra_teacher_location) is not type(None):
			pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

		# now_predict.shape = (N, T, V*C)
		now_predict = self.seq2seq(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# print('data', torch.min(pra_x), torch.max(pra_x[:,:2]), torch.min(now_predict), torch.max(now_predict))
		now_predict = self.reshape_from_lstm(now_predict) # (N, C, T, V)

		# (N, C, T, V) 
		# C: (4) x_mean, y_mean, x_sig, y_sig
		now_predict_loc = now_predict[:, :2] # mean x,y
		

		# print('data', pra_x.shape, now_predict.shape, pra_x.min().item(), pra_x[:,:2].max().item(), now_predict.min().item(), now_predict.max().item())
		# print('')
		# if pra_teacher_forcing_ratio==-1:
		# 	print('pra_x', pra_x[0,:2,:,19].flatten())
		# 	print('predx', now_predict[0,:2,:,19].flatten())
		# 	print('gt', pra_teacher_location[0,:2,:,19].flatten())

		return now_predict 

if __name__ == '__main__':
	model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
	print(model)
