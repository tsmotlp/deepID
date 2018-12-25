import torch
import torch.nn as nn
import torch.nn.functional as F

class deepID_1(nn.Module):
	def __init__(self, num_channels):
		super(deepID_1, self).__init__()

		self.block1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=4),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2))
		self.block2 = nn.Sequential (nn.Conv2d (in_channels=20, out_channels=40, kernel_size=3),
									 nn.ReLU (),
									 nn.MaxPool2d (kernel_size=2))
		self.block3 = nn.Sequential (nn.Conv2d (in_channels=40, out_channels=60, kernel_size=3),
									 nn.ReLU (),
									 nn.MaxPool2d (kernel_size=2))
		self.deepID_layer = nn.Conv2d(in_channels=60, out_channels=80, kernel_size=2)

		self.dense_layer = nn.Sequential(nn.Linear(in_features=2160, out_features=145),
										nn.ReLU())

	def forward(self, x):
		x = self.block1(x)
		x = self.block2 (x)
		x1 = self.block3 (x)
		x2 = self.deepID_layer (x1)
		x1 = x1.view(-1, self.num_flat_features(x1))
		x2 = x2.view(-1, self.num_flat_features(x2))
		x = torch.cat((x1, x2), 1)
		out = self.dense_layer(x)
		out = F.log_softmax(out, dim=1)
		return out


	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

