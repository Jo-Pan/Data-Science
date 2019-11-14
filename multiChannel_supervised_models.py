"""
Modification based on supervised_model.py
To have multi-channel graphsage features and use them together for prediction
Not importing models.py

combined __init__() and build()
"""
import tensorflow as tf

import graphsage.multiChannel_models as models
from .aggregators import MaxPoolingAggregator
from .layers import Layer, Dense
from .inits import glorot, zeros #used only in maxPoolingAggregator
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS

class MultiChannelGraphsage():

    def __init__(self, data_dicts, num_classes, placeholders,
                 concat=True, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0,
            **kwargs):

        # Constant for all channels
        models.GeneralizedModel.__init__(self, **kwargs)

        self.aggregator_cls = MaxPoolingAggregator
        self.model_size = model_size
        self.concat = concat
        dim_mult = 2 if self.concat else 1
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.batch_size = placeholders["batch_size"]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.placeholders = placeholders  # it's a single channel placeholder
        self.outputs1_lst = []
        self.aggregators_lst = []

        # Different for each channel

        for channel in range(4):
            #----- Set up variables -----
            self.channel = channel
            data_dict = data_dicts[channel]
            self.features = data_dict['features']
            self.adj_info = data_dict['adj_info']
            self.degrees = data_dict['minibatch'].deg
            self.layer_infos = data_dict['layer_infos']
            embeds = None

            features = tf.Variable(tf.constant(self.features, dtype=tf.float32), trainable=False)
            self.dims = [(0 if features is None else self.features.shape[1]) + identity_dim]
            self.dims.extend([self.layer_infos[i].output_dim for i in range(len(self.layer_infos))])

            self.inputs1 = tf.placeholder(tf.int32, shape=(None), name='batch1-{}'.format(self.channel))
            num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

            # ----- Sample inputs ---
            samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
            outputs1, aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                    support_sizes1, concat=self.concat, modelinputs1_size=self.model_size)
            outputs1 = tf.nn.l2_normalize(outputs1, 1)
            # outputs1 shape = (?, 256)

            self.outputs1_lst.append(outputs1)
            self.aggregators_lst.append(aggregators)


        self.node_pred = Dense(dim_mult*self.dims[-1], self.num_classes,
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights-{}'.format(self.channel)] = glorot([hidden_dim, output_dim],
                                                                        name='neigh_weights-{}'.format(self.channel))

            self.vars['self_weights-{}'.format(self.channel)] = glorot([input_dim, output_dim],
                                                                       name='self_weights-{}'.format(self.channel))
            if self.bias:
                self.vars['bias-{}'.format(self.channel)] = zeros([self.output_dim],
                                                                  name='bias-{}'.format(self.channel))

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights-{}'.format(self.channel)])
        from_self = tf.matmul(self_vecs, self.vars["self_weights-{}".format(self.channel)])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias-{}'.format(self.channel)]

        return self.act(output)

