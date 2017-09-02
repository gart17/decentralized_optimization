import tensorflow as tf

def define_logistic_model(model, dataset):
	'''
	takes a model object and add it a logistic model 
	'''
	# data = {'X': predictor, 'y': predicted_value}
	# parameters = {'W': coefficient, 'b': bias}
	# loss = cross_entropy(softmax(XW + b), y)
	data_size = dataset.output_shapes['data'].as_list()[0]
	n_labels = dataset.output_shapes['labels'].as_list()[0]
	data_type = dataset.output_types['data']
	label_type = dataset.output_types['labels'] 
	with tf.name_scope('data'):
		model.data['X'] = tf.placeholder(dtype=data_type, shape=[model.BATCH_SIZE, data_size], name='X')
		model.data['y'] = tf.placeholder(dtype=label_type, shape=[model.BATCH_SIZE, n_labels], name='y')
	with tf.name_scope('parameters'):
		model.parameters['W'] = tf.Variable(tf.zeros([data_size, n_labels], dtype=data_type), name='W')
		model.parameters['b'] = tf.Variable(tf.zeros([1, n_labels],dtype=data_type), name='b')
	with tf.name_scope('loss'):
		logits = tf.matmul(model.data['X'], model.parameters['W']) + model.parameters['b']
		entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=model.data['y'], name='loss')
		model.loss = tf.reduce_mean(entropy)