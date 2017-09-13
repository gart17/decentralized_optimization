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

		if train['parameters_to_train'] == 'all':
			params = agent.model.parameters
		else:
			params = {name: agent.model.parameters[name] for name in train['parameters_to_train']}
		data = agent.model.data.values()
		step = train['step_size']

		if train['scheme'] == 'exact_diffusion':
			with tf.name_scope('Initialization'):
				# agent.psi, agent.psi_past and agent.phi for adaption and correction steps
				agent.model.psi = {name: tf.Variable(param.initialized_value()) for name, param in params}
				agent.model.psi_past = {name: tf.Variable(param.initialized_value()) for name, param in params}
				agent.model.phi = {name: tf.Variable(param.initialized_value()) for name, param in params}
				# initialize each parameter
				for name in params.keys():
					agent.initialization += [params[name].initializer, 
						agent.model.psi[name].initializer,
						agent.model.psi_past[name].initializer,
						agent.model.phi[name].initializer]
			with tf.name_scope('computation'):
				# assuming that variables on other agents have been initialized
				# compute gradients 
				grads = tf.gradients(agent.model.loss, params.values())
				# gradient adaption
				for name in params.keys():
					agent.computation += [tf.assign(agent.model.psi[name])]
				

				

		else:
			pass
def train_diffusion(cluster, sess, train):
	pass