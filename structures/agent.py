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
				parameters_to_train = 'all'): 
		self.cluster = cluster
		self.suffix = '.' + str(agent_index)
		with tf.name_scope('agent' + self.suffix) as scope:
			self.scope = scope
			self.dataset = dataset
			self.model = Model(dataset=dataset, loss_model=loss_model) 
		self.neighbor_weights = neighbor_weights
		
		# Every agent has the following lists of operators,
		# added by Cluster.set_training 
		# Cluster.train runs (in sequence or parallely) 
		# for example, init -> N times (comp -> comm) -> conc 
		self.initialization = []
		self.computation = []
		self.communication = []
		self.conclusion = []
		# self.gradients = dict() # store gradient values
		# self.cache = None # store cache for certain functions, eg accelerations
	
	# def initialize(self, sess):
	# 	for _, parameter in self.model.parameters.items():
	# 		sess.run(parameter.initializer)
	
	# def compute_gradients(self, sess):
	# 	for name, gradient_op in self.model.gradients.items():
	# 		self.gradients[name] = sess.run(gradient_op)
		