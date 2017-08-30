import tensorflow as tf
from decentralized_optimization.structures.agent import Agent

def add_diffusion_operators(agent, train):
	'''
	add diffusion operators to agent
	train: a dict that contains patterns and parameters of decent opt
	'''
	if train['exact']:
		agent.decentralized_scheme = 'exact_diffusion'
	else:
		agent.decentralized_scheme = 'diffusion'

	# Set weights 
	agent.neighbor_weights[agent.agent_index] = agent.neighbor_weights[agent.agent_index] + 1
	agent.neighbor_weights = agent.neighbor_weights / 2

	# Initialization
	