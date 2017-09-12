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

		parameters_to_train = train.get('parameters_to_train', 'all')
		if train['parameters_to_train'] == 'all':
			params = agent.model.parameters.values()
		else:
			params = [agent.model.parameters[name] for name in train['parameters_to_train']]
		data = agent.model.data.values()
		opt = tf.train.GradientDescentOptimizer(learning_rate=train['step_size'])

		if train['scheme'] == 'exact_diffusion':
			agent.psi = dict()
			agent.phi = dict()
			with tf.name_scope('Initialization'):
				for w in params:
					agent.initialization.append(w.initializer)
				grads = opt.compute_gradients(agent.model.loss, params)
				
		else:
			pass
def train_diffusion(cluster, sess, train):
	pass