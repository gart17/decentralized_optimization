import tensorflow as tf

class Model:
	'''
	define a loss model
	it is a wrapper class for functions defined in ..models
	'''
	
	def __init__(self, dataset, loss_model):
		'''
		parameters_to_train: list of names, only compute and update these parameters
		'''
		self.data = dict()
		self.parameters = dict()
		self.loss_model = loss_model
		self.BATCH_SIZE = 1
		with tf.name_scope('model'):
			self.define_model(dataset, loss_model)
			# self.define_gradients(parameters_to_train)
			# self.define_descents(step_size)

	def define_model(self, dataset, loss_model):
		if self.loss_model == 'logistic':
			from model.logistic import define_logistic_model
			define_logistic_model(model, dataset)
		else:
			print('Loss model {0} not supported yet.'.format(loss_model))
			raise NotImplementedError
	


	# def define_gradients(self, parameters_to_train):
	# 	with tf.name_scope('gradients'):
	# 		if parameters_to_train == 'all':
	# 			parameters_to_train = self.parameters.keys()
	# 		gradient_ops = tf.gradients(
	# 			self.loss, [self.parameters[name] for name in parameters_to_train])
	# 		self.gradients = dict(zip(parameters_to_train, gradient_ops))	
	
	# def define_descents(self, step_size):
	# 	with tf.name_scope('descents'):
	# 		self.descents = dict()
	# 		for name, gradient in self.gradients.items():
	# 			self.descents[name] = self.parameters[name].assign_sub(step_size * gradient)
	
