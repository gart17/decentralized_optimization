import tensorflow as tf

def add_diffusion_operators(agent, train):
	'''
	add diffusion operators to agent
	train: a dict that contains patterns and parameters of decent opt
	'''

	# Set weights for diffusion
	agent.neighbor_weights[agent.agent_index] = agent.neighbor_weights[agent.agent_index] + 1
	agent.neighbor_weights = agent.neighbor_weights / 2

	# Add diffusion operators
	with tf.name_scope(agent.scope):
	# Initialization
		params = agent.model.parameters.values()
		data = agent.model.parameters.data()
		if agent. == 'exact_diffusion':
			agent.psi = dict()
			agent.phi = dict()
			with tf.name_scope('Initialization'):
				for p in params:
					agent.initialization.append(p.initializer)	
				for p in params:
					agent.initialization.append()
		else:

def train_diffusion(cluster, sess, train):
	pass