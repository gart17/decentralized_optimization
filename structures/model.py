class Model:
    '''
    self.agent_index
    self.data
    self.parameters
    self.loss
    
    get_gradients
    gradient_descent
    '''
    
    def __init__(self, agent_index, data_dict, parameters_dict, loss_model='logistic'):
        '''
        data_dict: a dict of the form {placeholder_name: tf.placeholder object}
        parameters_dict: a dict of the form {variable_name: tf.Variable object}
        '''
        self.data = dict()
        self.parameters = dict()
        suffix = str(agent_index)
        with tf.name_scope('model' + suffix):
            with tf.name_scope('data' + suffix):
                for name, data_ph in data_dict.items():
                    self.data[name] = tf.identity(data_ph)

            with tf.name_scope('parameters' + suffix):
                for name, param in parameters_dict.items():
                    self.parameters[name] = tf.identity(param)

            with tf.name_scope('loss' + suffix):
                if loss_model == 'logistic':
                    # logistic model using softmax
                    # data_dict = {'X': predictor, 'y': predicted_value}
                    # parameters_dict = {'W': coefficient, 'b': bias}
                    # loss = cross_entropy(softmax(XW + b), y)
                    self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                        logits=tf.matmul(self.data['X'], self.parameters['W']) + self.parameters['b'], 
                        labels=self.data['y'], 
                        name='loss'))
                else: 
                    print('Loss model {0} not supported yet.'.format(loss_model))
                    raise NotImplementedError
    
    def get_gradients(self, parameter_names=None):
        '''
        returns gradients_dict {param_name: gradient_of_param_wrt_loss}
        '''
        if parameter_names is None:
            parameter_names = self.parameters.keys()
        gradients = tf.gradients(loss, [self.parameters[name] for name in parameter_names])
        return dict(zip(parameter_names, gradients))
    
    def descent_update(self, step_size, parameter_names=None, gradients_dict=None):
        '''
        returns descent_dict that updates parameters in parameter_names by -step_size*gradient
        '''
        if parameter_names is None:
            parameter_names = self.parameter.keys()
            gradients_dict = self.get_gradients()
        descent_ops = dict()
        for name in parameter_names:
            descent[name] = tf.assign_sub(self.parameters[name], step_size * gradients_dict[name])
        return descent_dict