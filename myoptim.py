# Let f(x) = x^2 - x - 6
# Find x = argmin f
# by Neural Networks

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np
import matplotlib.pyplot as plt



class LinearLayer(torch.nn.Module):

	def __init__(self, input_nodes=1, output_nodes=1):
		super(LinearLayer, self).__init__()
		self.linear_output = torch.nn.Linear(input_nodes, output_nodes, bias=False)

	def forward(self, x):
		logits = self.linear_output(x)
		return logits


class MySGD(Optimizer):
	def __init__(self, parameters, lr=required):
		if lr is not required and lr < 0.0:
			raise ValueError(f"Invalid Learning Rate {lr}")
		defaults = dict(lr=lr)
		super(MySGD, self).__init__(parameters, defaults)

	def __setstate__(self, state):
		super(MySGD, self).__setstate__(state)

	@torch.no_grad()
	def step(self):
		loss = None
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				dp = p.grad
				# p = p - lr * dp
				p.add_(dp, alpha=-group['lr'])
		return loss


def loss_fn(x):
	return x ** 2 - x -6



if __name__ == "__main__":
	print("Create Dataset")
	# dataset
	data = torch.tensor([1.0], dtype=torch.float, requires_grad=True)
	

	learning_rate = 0.1
	num_epoch = 100

	# Construct model
	model = LinearLayer()
	# for _ in model.parameters():
	# 	print(_)
	# print(model.linear_output.weight)

	# Optimizer Setting
	torch.manual_seed(0)
	op_name = ['SGD', 'MySGD']
	op_list = [optim.SGD(model.parameters(), lr=learning_rate),
				MySGD(model.parameters(), lr=learning_rate)]


	optimizer = op_list[1]

	# Train model
	x = []
	y = []
	for epoch in range(num_epoch):
		logits = model(data)
		loss = loss_fn(logits)

		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

		w = model.linear_output.weight.detach().numpy().item()
		x.append(w)
		y.append(loss_fn(w))

	plt.plot(x, y, marker='o')
	plt.show()

