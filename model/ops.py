import tensorflow as tf
import numpy as np

weight_initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
weight_regularizer = tf.keras.regularizers.l2(0.0001)
weight_regularizer_fully = tf.keras.regularizers.l2(0.0001)


##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
class Conv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn :
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, use_bias=self.use_bias), name='sn_' + self.name)
        else :
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer,
                                               strides=self.stride, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        x = self.conv(x)

        return x

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(self.units,
                                                                  kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer_fully,
                                                                  use_bias=self.use_bias), name='sn_' + self.name)
        else :
            self.fc = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer_fully,
                                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = flatten()(x)
        x = self.fc(x)

        return x

##################################################################################
# Blocks
##################################################################################

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels = channels

        self.conv_0 = Conv(self.channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False,  name='conv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5, name='batch_norm_1')

    def call(self, x_init, training=None, mask=None):
        with tf.name_scope(self.name):
            with tf.name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = GLU()(x)

            with tf.name_scope('res2'):
                x = self.conv_1(x)
                x = self.batch_norm_1(x, training=training)

            return x + x_init


##################################################################################
# Normalization
##################################################################################

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5, name='BatchNorm'):
        super(BatchNorm, self).__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon

    def call(self, x, training=None, mask=None):
        x = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon,
                                               center=True, scale=True,
                                               name=self.name)(x, training=training)
        return x

##################################################################################
# Activation Function
##################################################################################

def Leaky_Relu(x=None, alpha=0.01, name='leaky_relu'):
    # pytorch alpha is 0.01
    if x is None:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)
    else:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)(x)

def Relu(x=None, name='relu'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)

    else:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)(x)

class GLU(tf.keras.layers.Layer):
    def __init__(self):
        super(GLU, self).__init__()

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, 'channels dont divide 2!'
        self.n_dim = len(input_shape)
        self.output_dim = input_shape[-1] // 2

    def call(self, x, training=None, mask=None):
        nc = self.output_dim
        if self.n_dim == 4:
            return x[:, :, :, :nc] * tf.sigmoid(x[:, :, :, nc:])
        if self.n_dim == 3:
            return x[:, :, :nc] * tf.sigmoid(x[:, :, nc:])
        if self.n_dim == 2:
            return x[:, :nc] * tf.sigmoid(x[:, nc:])

def Tanh(x=None, name='tanh'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name)
    else:
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name)(x)

##################################################################################
# Pooling & Resize
##################################################################################

def resize(x, target_size):
    return tf.image.resize(x, size=target_size, method=tf.image.ResizeMethod.BILINEAR)

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def flatten():
    return tf.keras.layers.Flatten()

##################################################################################
# KL-Divergence Loss Function
##################################################################################

def reparametrize(mean, logvar):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

@tf.function
def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss

##################################################################################
# Class function
##################################################################################

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        # self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel = self.w / sigma

    def restore_weights(self):

        self.layer.kernel = self.w

##################################################################################
# Natural Language Processing
##################################################################################

class VariousRNN(tf.keras.layers.Layer):
    def __init__(self, n_hidden=128, n_layer=1, dropout_rate=0.5, bidirectional=True, return_state=True, rnn_type='lstm', name='VariousRNN'):
        super(VariousRNN, self).__init__(name=name)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.return_state = return_state
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.cell_type = tf.keras.layers.LSTMCell
        elif self.rnn_type == 'gru':
            self.cell_type = tf.keras.layers.GRUCell
        else:
            raise NotImplementedError

        self.rnn = tf.keras.layers.RNN([self.cell_type(units=n_hidden, dropout=self.dropout_rate) for _ in range(self.n_layer)], return_sequences=True, return_state=self.return_state)
        if self.bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)
        """
        if also return_state=True, 
        whole_sequence, forward_hidden, forward_cell, backward_hidden, backward_cell (LSTM)
        whole_sequence, forward_hidden, forward_cell (GRU)
        sent_emb = tf.concat([forward_hidden, backward_hidden], axis=-1)
        """

    def call(self, x, training=None, mask=None):
        if self.return_state:
            if self.bidirectional:
                if self.rnn_type == 'gru':
                    output, forward_h, backward_h = self.rnn(x, training=training)
                else : # LSTM
                    output, forward_state, backward_state = self.rnn(x, training=training)
                    forward_h, backward_h = forward_state[0], backward_state[0]
                    forward_c, backward_c = forward_state[1], backward_state[1]

                sent_emb = tf.concat([forward_h, backward_h], axis=-1)
            else :
                if self.rnn_type =='gru':
                    output, forward_h = self.rnn(x, training=training)
                else :
                    output, forward_state = self.rnn(x, training=training)
                    forward_h, forward_c = forward_state

                sent_emb = forward_h

        else :
            output = self.rnn(x, training=training)
            sent_emb = output[:, -1, :]

        word_emb = output

        return word_emb, sent_emb

def EmbedSequence(n_words, embed_dim, trainable=True, name='embed_layer') :

    emeddings = tf.keras.layers.Embedding(input_dim=n_words, output_dim=embed_dim,
                                          trainable=trainable, name=name)
    return emeddings

class DropOut(tf.keras.layers.Layer):
    def __init__(self, drop_rate=0.5, name='DropOut'):
        super(DropOut, self).__init__(name=name)
        self.drop_rate = drop_rate

    def call(self, x, training=None, mask=None):
        x = tf.keras.layers.Dropout(self.drop_rate, name=self.name)(x, training=training)
        return x
