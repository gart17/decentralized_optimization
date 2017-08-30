import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import networkx as nx

from decentralized_optimization.structures.topology import Topology
from decentralized_optimization.structures.cluster import Cluster

####
tf.reset_default_graph()
sess = tf.Session()

N_AGENTS = 5

# distribute data across agents
with tf.name_scope('Data_Distributor'):
	# read MNIST data
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('/data/mnist', one_hot=True)
	mnist_dataset = tf.contrib.data.Dataset.from_tensor_slices(
		{'data': mnist.train.images, 'labels':mnist.train.labels}
		).batch(int(mnist.train.num_examples / N_AGENTS)) # only use training dataset
	iterator = mnist_dataset.make_initializable_iterator()
	# Use batch to break dataset into N_AGENTS pieces
	next_batch = iterator.get_next()
	sess.run(iterator.initializer)
	mnist_datasets = []
	for _ in range(N_AGENTS):
		mnist_datasets.append(
			tf.contrib.data.Dataset.from_tensor_slices(
				sess.run(next_batch)))

# create a cluster
c = Cluster(
	mnist_datasets, 
	loss_model='logistic',  
	decentralized_scheme='diffusion', 
	topology=Topology(N_AGENTS, 0.6, 'metropolis'))

# specify traning parameters
c.set_training_parameters({
	'step_size': 0.01

	})

# visualize through TensorBoard
writer = tf.summary.FileWriter('TensorBoard', sess.graph)

# Start training
c.train(sess)

writer.close()
sess.close()

if __name__ == '__main__':
	logistic_MNIST_demo()