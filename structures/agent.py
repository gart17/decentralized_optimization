import tensorflow as tf
from decentralized_optimization.structures.model import Model

class Agent:
	'''
	Holds decentralized training operators,
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
		
		'''
		Every agent has the following lists of operators,
		added by Cluster.set_training 
		in general, Cluster.train runs (in sequence or parallely) 
		initialization -> N times (computation -> communication) -> conclusion 
		actual order will be determined inside Cluster.train() function
		'''
		self.initialization = []
		self.computation = []
		self.communication = []
		self.conclusion = []

