class Model:
    '''
    Construct graphs for Agent to manipulate. 
    Provide API for loss, gradients, and descents
    For other ops, you need to define yourself by 
        self.my_op = blah
    
    Member variables:
    self.agent_index  # for naming purpose only
    self.data # dict
    self.parameters # dict
    self.loss # char
    self.gradients # dict
    self.descents # dict
    '''
    
    def __init__(self, 
                 agent_index, 
                 data_dict, 
                 parameters_dict, 
                 loss_model='logistic',
                 step_size=0.01,
                 parameters_to_train='all'):
        '''
        data_dict: a dict of the form {placeholder_name: tf.placeholder object}
        parameters_dict: a dict of the form {variable_name: tf.Variable object}
        parameters_to_train: list of names, only compute and update these parameters
        '''
        self.data = dict()
        self.parameters = dict()
        self.suffix = str(agent_index)
        with tf.name_scope('model' + self.suffix):
            self.define_data(data_dict)
            self.define_parameters(parameters_dict)
            self.define_loss(loss_model)
            self.define_gradients(parameters_to_train)
            self.define_descents(step_size)
            
    def define_data(self, data_dict):
        with tf.name_scope('data' + self.suffix):
            for name, data_ph in data_dict.items():
                self.data[name] = tf.identity(data_ph)

    def define_parameters(self, parameters_dict):
        with tf.name_scope('parameters' + self.suffix):
            for name, param in parameters_dict.items():
                self.parameters[name] = tf.identity(param)
    
    def define_loss(self, loss_model):
        with tf.name_scope('loss' + self.suffix):
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
    