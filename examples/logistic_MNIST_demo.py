import numpy as np
import tensorflow as tf

from decentralized_optimization.tools.read_data import mnist_data_distributor
from decentralized_optimization.structures.cluster import Cluster

tf.reset_default_graph()

# all execution are done through a default session
sess = tf.Session()

# read mnist data into a list of tf Dataset
mnist_datasets = mnist_data_distributor(n_agents=5, session=sess)

# the following has to be executed in sequence
# so if you reset topology, you have to reset datasets and training as well

# create a cluster
c = Cluster(mode='single_process_simulation')

# set topology of the cluster
c.set_topology(n_agents=5, density=0.6, policy='metropolis')

# assign datasets and loss_models to each agents in the cluster
c.set_model(datasets=mnist_datasets, loss_model='logistic')

# specify traning parameters
c.set_training(scheme='diffusion', exact=True, gradient='GD', step_size=0.01)

# visualize through TensorBoard
writer = tf.summary.FileWriter('TensorBoard', sess.graph)

# Start training
c.train(sess)

# clean up
writer.close()
sess.close()

if __name__ == '__main__':
	logistic_MNIST_demo()