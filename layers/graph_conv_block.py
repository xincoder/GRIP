import torch
import torch.nn as nn
from layers.graph_operation_layer import ConvTemporalGraphical

class Graph_Conv_Block(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size,
				 stride=1,
				 dropout=0,
				 residual=True):
		super().__init__()

		assert len(kernel_size) == 2
		assert kernel_size[0] % 2 == 1
		padding = ((kernel_size[0] - 1) // 2, 0)
		
		self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
		self.tcn = nn.Sequential(
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=False),
			nn.Conv2d(
				out_channels,
				out_channels,
				(kernel_size[0], 1),
				(stride, 1),
				padding,
			),
			nn.BatchNorm2d(out_channels),
			nn.Dropout(dropout, inplace=False),
		)

		if not residual:
			self.residual = lambda x: 0
		elif (in_channels == out_channels) and (stride == 1):
			self.residual = lambda x: x
		else:
			self.residual = nn.Sequential(
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size=1,
					stride=(stride, 1)),
				nn.BatchNorm2d(out_channels),
			)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x, A):
		res = self.residual(x)
		x, A = self.gcn(x, A)
		x = self.tcn(x) + res
		return self.relu(x), A

