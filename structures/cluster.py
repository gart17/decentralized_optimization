import tensorflow as tf
from decentralized_optimization.structures.topology import Topology
from decentralized_optimization.structures.model import Model
from decentralized_optimization.structures.agent import Agent

class Cluster:
	
	def __init__(self, 
				datasets, # list of tf.contrib.learn.Dataset for each agent
				loss_model='logistic', 
				topology=Topology(),
				options = None):
		self.datasets = datasets
		self.loss_model = loss_model
		self.topology = topology
		self.options = options
		self.agents = [self.generate_agent(i) 
						for i in range(topology.n_agents)]

	def generate_agent(self, agent_index):
		# Infer models from datasets and loss_model
		return Agent(cluster=self, 
			agent_index=agent_index, 
			dataset=self.datasets[agent_index], 
			loss_model=self.loss_model, 
			neighbor_weights=self.topology.neighbor_weights,
			options=self.options)

	def set_training(self, 
		parameter_dict={'scheme': 'diffusion', 
						'exact': True, 
						'step_size': 0.01}):
		# despite its name, the function adds operators of decentralized optimization to each agent
		if scheme == 'diffusion':
			from decentralized_optimization.algorithms.diffusion import diffusion_operators
			for agent in self.agents:
				diffusion_operators(agent, parameter_dict)

		else:
			print('Decentralized scheme {0} not supported yet.'.format(self.scheme))
			raise NotImplementedError

	def train(sess):
		pass