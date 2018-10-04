class PredictChocolateRatings(object):
    def __init__(self, hparams):
        self.hparams = hparams 
        self.weights_dict = {'w': None, 'b': None}
        self.y_posterior = None
        
    def fit(self):
        N = x_train.shape[0]  # Number of rows in training data
        D = x_train.shape[1]  # Number of columns in training data

        x_ph = tf.placeholder(tf.float32, [None, D])  # Placeholder variable for x data
        y_ph = tf.placeholder(tf.float32, [None])  # Placeholder variable for y data 

        w = Normal(loc=tf.zeros(D), scale=tf.ones(D))  # Weights prior
        b = Normal(loc=tf.zeros(1.), scale=tf.ones(1.))  # Bias prior
        y = Normal(loc=ed.dot(x_ph, w) + b, scale=1.0)  # Likelihood function

        qw = Normal(loc=tf.get_variable('qw/loc', [D]),
                scale=tf.nn.softplus(tf.get_variable('qw/scale', [D])))  # Variational parameter
        qb = Normal(loc=tf.get_variable('qb/loc', [1]),
                    scale=tf.nn.softplus(tf.get_variable('qb/scale', [1])))  # Variational parameter

        data = preprocess_data.generator([x_train, y_train], hparams.batch_size)

        n_batch = int(N / hparams.batch_size)

        inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph}) # Reverse variational inference
        inference.initialize(
            n_iter=n_batch * hparams.num_epoch, n_samples=hparams.num_samples, scale={y: N / hparams.batch_size}, 
            logdir='log')
        tf.global_variables_initializer().run()

        for _ in range(inference.n_iter):
            X_batch, y_batch = next(data)
            info_dict = inference.update({x_ph:X_batch, y_ph: y_batch})
            inference.print_progress(info_dict)
        
        self.weights_dict['w'] = qw.sample(hparams.num_samples).eval()
        self.weights_dict['b'] = qw.sample(hparams.num_samples).eval()
           
    def evaluate(self, input_fn):
        print('to be completed')
    def visualize_weights(self, input_fn):
        print('to be completed')