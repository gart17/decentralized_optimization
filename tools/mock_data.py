def gaussian_data(n_agent, n_data, data_dim, mu=0, sigma=1, noise_level=0.3):
	'''
	Generate gaussian data along with true optimial values.

	Model: 
	y = X w0 + ep, 
	- X: n_data x n_agent, Gaussian random with mean mu and sd sigma
	- w0: data_dim x 1, uniformly random
	- ep: n_data x 1, Gaussian random with mean 0 and sd noise_level * sigma

	Input:
	- n_agent, n_data, data_dim: dimensions
	- mu, sigma
	- noise_level

	Output:
	Two dictionaries of the following:
	- data: 
		- X: explanatory data
		- y: dependent data
	- parameters:
		- w_opt: optimal parameters learnt from the whole dataset
		- cond: condition number of X, indicating difficulty of the problem
	'''
	X = np.random.randn(n_data, data_dim) * sigma
	w0 = np.random.rand(data_dim, 1)
	y = X.dot(w0) + noise_level * np.random.randn(n_data, 1)

	w_opt = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
	cond = np.linalg.cond(X)

	data = {'X': X, 'y': y}
	parameters = {'w_opt': w_opt, 'cond': cond}

	return data, parameters