import tensorflow as tf

class Model:
	'''
	Construct graphs for Agent to manipulate. 
	Provide API for loss, gradients, and descents
	For other ops, you need to define yourself by 
		self.my_op = blah

	Member variables:
	self.agent_index  # for naming purpose only
	self.data # dict of tf.placeholders
	self.parameters # dict of tf.Variables
	self.gradients # dict
	'''
	
	def __init__(self,
		dataset,
		agent_index,
		loss_model,
		parameters_to_train='all'):
		'''
		data_dict: a dict of the form {placeholder_name: tf.placeholder object}
		parameters_dict: a dict of the form {variable_name: tf.Variable object}
		parameters_to_train: list of names, only compute and update these parameters
		'''
		self.suffix = '.' + str(agent_index)
		self.data = dict()
		self.parameters = dict()
		self.loss_model = loss_model
		self.BATCH_SIZE = 1
		with tf.name_scope('model' + self.suffix):
			self.define_model(dataset)
			self.define_loss(loss_model)
			self.define_gradients(parameters_to_train)
			# self.define_descents(step_size)

	def define_model(self, dataset):
		if self.loss_model == 'logistic':
			data_size = dataset.output_shapes['data'].as_list()[0]
			n_labels = dataset.output_shapes['labels'].as_list()[0]
			data_type = dataset.output_types['data']
			label_type = dataset.output_types['labels'] 
			with tf.name_scope('data' + self.suffix):
				self.data['X'] = tf.placeholder(dtype=data_type, shape=[self.BATCH_SIZE, data_size], name='X')
				self.data['y'] = tf.placeholder(dtype=label_type, shape=[self.BATCH_SIZE, n_labels], name='y')
			with tf.name_scope('parameters' + self.suffix):
				self.parameters['W'] = tf.Variable(tf.zeros([data_size, n_labels], dtype=data_type), name='W')
				self.parameters['b'] = tf.Variable(tf.zeros([1, n_labels],dtype=data_type), name='b')
		else:
			print('Loss model {0} not supported yet.')
			raise NotImplementedError
	
	def define_loss(self, loss_model):
		with tf.name_scope('loss' + self.suffix):
			if loss_model == 'logistic':
				# logistic model using softmax
				# data_dict = {'X': predictor, 'y': predicted_value}
				# parameters_dict = {'W': coefficient, 'b': bias}
				# loss = cross_entropy(softmax(XW + b), y)
				logits = tf.matmul(self.data['X'], self.parameters['W']) + self.parameters['b']
				entropy = tf.nn.softmax_cross_entropy_with_logits(
					logits=logits, 
					labels=self.data['y'], 
					name='loss')
				self.loss = tf.reduce_mean(entropy)
				
	def define_gradients(self, parameters_to_train):
		with tf.name_scope('gradients' + self.suffix):
			if parameters_to_train == 'all':
				parameters_to_train = self.parameters.keys()
			gradient_ops = tf.gradients(
				self.loss, [self.parameters[name] for name in parameters_to_train])
			self.gradients = dict(zip(parameters_to_train, gradient_ops))	
	
	def define_descents(self, step_size):
		with tf.name_scope('descents' + self.suffix):
			self.descents = dict()
			for name, gradient in self.gradients.items():
				self.descents[name] = self.parameters[name].assign_sub(step_size * gradient)
	