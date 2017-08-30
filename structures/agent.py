import tensorflow as tf
from decentralized_optimization.structures.model import Model

class Agent:
	'''
	Holds decentralized operators,
	Execute initialization, gradient descent, 
	combination etc in the default session
	
	Algorithms in decentralized_optimization.algorithms take 
	an Agent as argument and add corresponding operators to it
	'''
	def __init__(self, 
				cluster, # parent cluster (to get parameters from other agents)
				agent_index, 
				dataset, 
				loss_model, 
				neighbor_weights, # list of weights corr to neighrbors
				parameters_to_train = 'all',
				options=None # additional options, eg 'svrg'
				): 
		self.cluster = cluster
		self.suffix = '.' + str(agent_index)
		with tf.name_scope('agent' + self.suffix):
			self.dataset = dataset
			self.model = Model(dataset=dataset, 
					agent_index=agent_index, 
					loss_model=loss_model, 
					parameters_to_train=parameters_to_train) 
		self.neighbor_weights = neighbor_weights  
		self.options = options
		
		self.gradients = dict() # store gradient values
		self.cache = None # store cache for certain functions, eg accelerations
	
	def initialize(self, sess):
		for _, parameter in self.model.parameters.items():
			sess.run(parameter.initializer)
	
	def compute_gradients(self, sess):
		for name, gradient_op in self.model.gradients.items():
			self.gradients[name] = sess.run(gradient_op)

	def train(self, sess):
		pass
		