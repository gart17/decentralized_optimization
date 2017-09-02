import tensorflow as tf
from decentralized_optimization.structures.topology import Topology
from decentralized_optimization.structures.model import Model
from decentralized_optimization.structures.agent import Agent
from decentralized_optimization import algorithms

class Cluster:
	
	def __init__(self, mode='single_process_simulation'):
		self.mode = mode
		
	def set_topology(self, 
			n_agents=5, 
			density=0.6, 
			policy='metropolis', 
			adjacency_matrix=None):
		self.topology = Topology(n_agents, density, policy, adjacency_matrix)

	def set_model(self, 
			datasets, # list of tf.contrib.learn.Dataset for each agent
			loss_model='logistic'):
		self.datasets = datasets
		self.loss_model = loss_model
		self.agents = [
			Agent(cluster=self, 
			agent_index=agent_index, 
			dataset=self.datasets[agent_index], 
			loss_model=self.loss_model, 
			neighbor_weights=self.topology.neighbor_weights) 
			for agent_index in range(self.topology.n_agents)]

	def set_training(self, **train):
		# despite its name, the function adds operators of decentralized optimization to each agent

		if train['scheme'] == 'diffusion':
			from algorithms.diffusion import add_diffusion_operators
			for agent in self.agents:
				add_diffusion_operators(agent, train)

		else:
			print('Decentralized scheme {0} not supported yet.'.format(self.scheme))
			raise NotImplementedError

	def train(sess):
		pass