from keras.layers import ConvLSTM2D,UpSampling2D,Reshape,Conv3D,Lambda,Flatten,Dense
from keras.models import Sequential,Model
from keras.engine.topology import Layer, InputSpec
#from keras.utils.vis_utils import plot_model
import keras.backend as K

def stacklayer(decoded):
    x = [decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,\
    decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,decoded,\
    decoded,decoded,decoded]
    return K.stack(x,axis=1)

def RAE(input_shape = (32,32,8192,1), filter_num = 10, encoding_len = 2):
	model = Sequential()
	model.add(ConvLSTM2D(filter_num, (32,32), strides=(32,32), padding='valid', activation='tanh', recurrent_activation='hard_sigmoid',activity_regularizer=None, return_sequences=False, dropout=0.0, recurrent_dropout=0.0, name='convlstm1',input_shape=input_shape))
	model.add(Flatten(name='flatten'))
	model.add(Dense(encoding_len,activation = 'linear',name='embedding'))

	model.add(Dense(2560,activation = 'linear'))
	model.add(Reshape((-1,256,filter_num)))
	model.add(UpSampling2D((32,32)))
	model.add(Lambda(stacklayer))
	model.add(ConvLSTM2D(filter_num, (32,32), strides=(1,1), padding='same', activation='tanh', recurrent_activation='hard_sigmoid',activity_regularizer=None, return_sequences=True, dropout=0.0, recurrent_dropout=0.0, name='deconvlstm1'))
	model.add(Conv3D(1,(32,32,1), strides=(1,1,1), padding='same',activation = 'sigmoid', name='deconv'))

	#model.summary()
	#plot_model(model,show_shapes=True)
	return model


class ClusteringLayer(Layer):

	def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)
		super(ClusteringLayer, self).__init__(**kwargs)

		self.n_clusters = n_clusters
		self.alpha = alpha
		self.initial_weights = weights
		self.input_spec = InputSpec(ndim=2)

	def build(self, input_shape):
		assert len(input_shape) == 2
		input_dim = input_shape[1]
		self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
		self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights
		self.built = True

	def call(self, inputs, **kwargs):
		q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
		q **= (self.alpha + 1.0) / 2.0
		q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
		return q

	def compute_output_shape(self, input_shape):
		assert input_shape and len(input_shape) == 2
		return input_shape[0], self.n_clusters

	def get_config(self):
		config = {'n_clusters': self.n_clusters}
		base_config = super(ClusteringLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class CREC(object):
    def __init__(self,
                 input_shape = (32,32,8192,1), 
                 filter_num = 10,
                 n_clusters=2,
                 alpha=1.0):

        super(CREC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.rae = RAE(input_shape = (32,32,8192,1), filter_num = 10, encoding_len = n_clusters)
        hidden = self.rae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.rae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.rae.input,
                           outputs=[clustering_layer, self.rae.output])


    def load_weights(self, weights_path):
    	self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    '''def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)'''







