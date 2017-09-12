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
		params = agent.model.parameters
		data = agent.model.data
		if train['scheme'] == 'exact_diffusion':
			agent.psi = dict()
			agent.phi = dict()
			with tf.name_scope('Initialization'):
				opt = tf.train.GradientDescentOptimizer(learning_rate=train['step_size'])
				for w in params:
					agent.initialization.append(w.initializer)
				grads = opt.compute_gradients(agent.model.loss, )
		else:
			pass
def train_diffusion(cluster, sess, train):
	pass