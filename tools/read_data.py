import tensorflow as tf

def mnist_data_distributor(n_agents, session):
	with tf.name_scope('data_distributor'):
		# read MNIST data
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets('/data/mnist', one_hot=True)
		mnist_dataset = tf.contrib.data.Dataset.from_tensor_slices(
			{'data': mnist.train.images, 'labels':mnist.train.labels}
			).batch(int(mnist.train.num_examples / n_agents)) # only use training dataset
		iterator = mnist_dataset.make_initializable_iterator()
		# Use batch to break dataset into N_AGENTS pieces
		next_batch = iterator.get_next()
		session.run(iterator.initializer)
		mnist_datasets = []
		for _ in range(n_agents):
			mnist_datasets.append(
				tf.contrib.data.Dataset.from_tensor_slices(
					session.run(next_batch)))
	return mnist_datasets