from model.ops import *
from tensorflow.keras import Sequential


##################################################################################
# Generator
##################################################################################
class RnnEncoder(tf.keras.Model):
    def __init__(self, n_words, embed_dim=256, drop_rate=0.5, n_hidden=128, n_layer=1, bidirectional=True, rnn_type='lstm', name='RnnEncoder'):
        super(RnnEncoder, self).__init__(name=name)
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.model = self.architecture()
        self.rnn = VariousRNN(self.n_hidden, self.n_layer, self.drop_rate, self.bidirectional, rnn_type=self.rnn_type, name=self.rnn_type + '_rnn')

    def architecture(self):
        model = []

        model += [EmbedSequence(self.n_words, self.embed_dim, name='embed_layer')] # [bs, seq_len, embed_dim]
        model += [DropOut(self.drop_rate, name='dropout')]

        model = Sequential(model)

        return model


    def call(self, caption, training=True, mask=None):
        # caption = [bs, seq_len]
        x = self.model(caption, training=training)
        word_emb, sent_emb = self.rnn(x, training=training)  # (bs, seq_len, n_hidden * 2) (bs, n_hidden * 2)
        mask = tf.equal(caption, 0)

        return word_emb, sent_emb, mask

class CA_NET(tf.keras.Model):
    def __init__(self, c_dim, name='CA_NET'):
        super(CA_NET, self).__init__(name=name)
        self.c_dim = c_dim # z_dim, condition dimension

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.c_dim * 2, name='mu_fc')]
        model += [Relu()]

        model = Sequential(model)

        return model

    def call(self, sent_emb, training=True, mask=None):
        x = self.model(sent_emb, training=training)

        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]

        c_code = reparametrize(mu, logvar)

        return c_code, mu, logvar

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, channels, name='SpatialAttention'):
        super(SpatialAttention, self).__init__(name=name)
        self.channels = channels # idf, x.shape[-1]

        self.word_conv = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='word_conv')
        self.sentence_fc = FullyConnected(units=self.channels, name='sent_fc')
        self.sentence_conv = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='sentence_conv')

    def build(self, input_shape):
        self.bs, self.h, self.w, _ = input_shape[0]
        self.hw = self.h * self.w # length of query
        self.seq_len = input_shape[2][1] # length of source
        self.gamma = self.add_weight(self.name + '_gammaw',
                                    shape=(),
                                    initializer=tf.initializers.Ones)

    def call(self, inputs, training=True):
        x, sentence, context, mask = inputs # context = word_emb
        x = tf.reshape(x, shape=[self.bs, self.hw, -1])

        context = tf.expand_dims(context, axis=1)
        context = self.word_conv(context)
        context = tf.squeeze(context, axis=1)

        attn = tf.matmul(x, context, transpose_b=True) # [bs, hw, seq_len]
        attn = tf.reshape(attn, shape=[self.bs * self.hw, self.seq_len])

        mask = tf.tile(mask, multiples=[self.hw, 1])
        attn = tf.where(tf.equal(mask, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=mask.shape), y=attn)
        attn = tf.nn.softmax(attn)
        attn = tf.reshape(attn, shape=[self.bs, self.hw, self.seq_len])
        attn = tf.multiply(attn, self.gamma)

        weighted_context = tf.matmul(context, attn, transpose_a=True, transpose_b=True)
        weighted_context = tf.reshape(tf.transpose(weighted_context, perm=[0, 2, 1]), shape=[self.bs, self.h, self.w, -1])
        word_attn = tf.reshape(attn, shape=[self.bs, self.h, self.w, -1])

        return weighted_context, word_attn

class FeatureAttention(tf.keras.layers.Layer):
    def __init__(self, channels, name='SpatialAttention'):
        super(FeatureAttention, self).__init__(name=name)
        self.channels = channels # idf, x.shape[-1]

        self.conv_f = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='conv_f')
        self.conv_g = Conv(self.channels, kernel=1, stride=1, use_bias=False, name='conv_g')
        

    def build(self, input_shape):
        self.bs, self.h, self.w, self.r = input_shape[0]
        self.hw = self.h * self.w # length of query
        self.gamma = self.add_weight(self.name + '_gamma',
                                    shape=(),
                                    initializer=tf.initializers.Ones)

        
    def call(self, inputs, training=True):
        x = inputs 
        x = tf.reshape(x, shape=[self.bs, self.hw, -1])
        x = tf.expand_dims(x, axis = -1)
        
        x_f = self.conv_f(x)
        x_g = self.conv_g(x)
        x_f = tf.reshape(x_f, shape = [self.bs, self.hw, -1])
        x_g = tf.reshape(x_g, shape = [self.bs, self.hw, -1])

        attn = tf.matmul(x_f, tf.transpose(x_g, perm = [0,2,1])) 
        attn = tf.nn.softmax(attn)
        attn = tf.multiply(attn, self.gamma)
        
        weighted_context = tf.matmul(tf.squeeze(x), attn, transpose_a=True, transpose_b=True)
        weighted_context = tf.reshape(tf.transpose(weighted_context, perm=[0, 2, 1]), shape=[self.bs, self.h, self.w, -1])
        
        return weighted_context

class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name='UpBlock'):
        super(UpBlock, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []

        model += [Conv(self.channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]

        model = Sequential(model)

        return model

    def call(self, x_init, training=True):
        x = nearest_up_sample(x_init, scale_factor=2)

        x = self.model(x, training=training)

        return x

class Generator_64(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_64'):
        super(Generator_64, self).__init__(name=name)
        self.channels = channels

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []

        model += [FullyConnected(units=self.channels * 4 * 4 * 2, use_bias=False, name='code_fc')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]
        model += [tf.keras.layers.Reshape(target_shape=[4, 4, self.channels])]

        for i in range(4):
            model += [UpBlock(self.channels // 2, name='up_block_' + str(i))]
            self.channels = self.channels // 2

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_64_logit')]
        generate_img_block += [Tanh()]

        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block


    def call(self, c_z_code, training=True, mask=None):
        h_code = self.model(c_z_code, training=training)
        x = self.generate_img_block(h_code, training=training)

        return h_code, x


class Generator_128(tf.keras.layers.Layer):
    def __init__(self, channels, name='Generator_128'):
        super(Generator_128, self).__init__(name=name)
        self.channels = channels # gf_dim

        self.spatial_attention = SpatialAttention(channels=self.channels)
        self.feature_attention = FeatureAttention(channels=self.channels)

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []

        for i in range(2):
            model += [ResBlock(self.channels * 3, name='resblock_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_128_logit')]
        generate_img_block += [Tanh()]

        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block

    def call(self, inputs, training=True):
        h_code, c_code, word_emb, mask = inputs
        c_code, _ = self.spatial_attention([h_code, c_code, word_emb, mask])
        f_code = self.feature_attention([h_code])

        h_c_f_code = tf.concat([h_code, c_code, f_code], axis=-1)

        h_code = self.model(h_c_f_code, training=training)
        x = self.generate_img_block(h_code)

        return x

class Generator(tf.keras.Model):
    def __init__(self, channels, name='Generator'):
        super(Generator, self).__init__(name=name)
        self.channels = channels

        self.g_64 = Generator_64(self.channels * 16, name='g_64')
        self.g_128 = Generator_128(self.channels, name='g_128')

    def call(self, inputs, training=True, mask=None):
        c_code, z_code, word_emb, mask = inputs
        c_z_code = tf.concat([c_code, z_code], axis=-1)

        h_code1, x_64 = self.g_64(c_z_code, training=training)
        x_128 = self.g_128([h_code1, c_code, word_emb, mask], training=training)

        x = [x_64, x_128]

        return x
