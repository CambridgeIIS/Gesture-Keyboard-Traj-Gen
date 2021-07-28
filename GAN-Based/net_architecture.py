import tensorflow as tf
from tensorflow.keras import layers
import os



class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # self.embedding = tf.keras.layers.Embedding(27, 4)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden_state):
        # x = self.embedding(x)
        # input = (batch size, max phrase length,1)
        # output = (batch size, max phrase length, encoder units), state = (batch size, encoder units)

        # hidden_state = tf.zeros((self.batch_sz, self.enc_units))

        output, state = self.gru(x, initial_state = hidden_state)
        return output, state

    def initialize_hidden_state(self):
      return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BahdanauAttention,self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query = (batch size, encoder units), values = (batch size, max phrase length, encoder units)
        # hidden with time axis = (batch, 1, encoder units)
        # score = (batch size, max phrase length, 1)
        # attention weights = (batch size, max phrase length, 1)
        # context vector = (batch size, encoder units)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights



class Decoder(tf.keras.layers.Layer):
  def __init__(self, param_dim,dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(param_dim)

    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden_state, enc_output):
    #   x = (batch size, 1, parameter dim), hidden = (batch size, encoder units), encoder output = (batch size, max phrase length, encoder units)
    #   context_vector = (batch size, encoder units)
    context_vector, attention_weights = self.attention(hidden_state, enc_output)

    # x = (batch size, 1, parameter dim + encoder units)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # output = (batch size, 1, decoder units)
    output, state = self.gru(x)

    # output = (batch size, decoder units)
    output = tf.reshape(output, (-1, output.shape[2]))

    # x = (batch size, parameter dim)
    x = self.fc(output)

    return x, state, attention_weights


class Convert_to_params(tf.keras.layers.Layer):
    def __init__(self, param_dim, dec_units, batch_sz):
        super(Convert_to_params, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(param_dim)

    def call(self, x, hidden_state):
        # x = (batch size, 1, input dimensions)
        # hidden = (batch size, decoder units)

        # output = (batch size, 1, decoder units)
        output, state = self.gru(x, initial_state = hidden_state)

        # output = (batch size, decoder units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # x = (batch size, parameter dim)
        x = self.fc(output)

        return x, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

# this is the decoder 2 with non teacher forcing
class loop_layer0(tf.keras.layers.Layer):
    def __init__(self, input_dim, param_dim, dec_units, batch_sz ):
        super(loop_layer0, self).__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.dec_units = dec_units
        self.batch_sz = batch_sz

        self.Convert = Convert_to_params(self.param_dim, self.dec_units, self.batch_sz)


    def call(self, input):
        dec_hidden = self.Convert.initialize_hidden_state()
        dec_input = tf.expand_dims([tf.zeros(self.input_dim)] * self.batch_sz, 1)
        x = tf.expand_dims([tf.zeros(self.param_dim)] * self.batch_sz, 1)

        for t in range(1, input.shape[1]):
            prediction, dec_hidden = self.Convert(dec_input, dec_hidden)
            prediction = tf.expand_dims(prediction, 1)
            dec_input = prediction
            x = tf.concat([x, prediction], 1)

        return x

# this is the decoder 2 with teacher forcing
class loop_layer1(tf.keras.layers.Layer):
    def __init__(self, input_dim, param_dim, dec_units, batch_sz ):
        super(loop_layer1, self).__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.dec_units = dec_units
        self.batch_sz = batch_sz

        self.Convert = Convert_to_params(self.param_dim, self.dec_units, self.batch_sz)


    def call(self, input):
        dec_hidden = self.Convert.initialize_hidden_state()
        dec_input = tf.expand_dims([tf.zeros(self.input_dim)] * self.batch_sz, 1)
        x = tf.expand_dims([tf.zeros(self.param_dim)] * self.batch_sz, 1)

        for t in range(1, input.shape[1]):
            prediction, dec_hidden = self.Convert(dec_input, dec_hidden)
            dec_input = tf.expand_dims(input[:, t], 1)
            prediction = tf.expand_dims(prediction, 1)
            x = tf.concat([x, prediction], 1)

        return x


# this is the decoder 1 with teacher forcing
class loop_layer2(tf.keras.layers.Layer):
    def __init__(self, enc_units, param_dim, dec_units, batch_sz):
        super(loop_layer2, self).__init__()
        self.enc_units = enc_units
        self.param_dim = param_dim
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.decoder = Decoder(self.param_dim,self.dec_units,self.batch_sz)

    def call(self, x, enc_output, enc_hidden):
        dec_input = tf.expand_dims([tf.zeros(self.param_dim)] * self.batch_sz, 1)
        outputs = tf.expand_dims([tf.zeros(self.param_dim)] * self.batch_sz, 1)
        dec_hidden = enc_hidden

        for t in range(1, x.shape[1]):
            prediction, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            prediction = tf.expand_dims(prediction, 1)

            dec_input = tf.expand_dims(x[:, t], 1)
            # dec_input  = prediction

            outputs = tf.concat([outputs, prediction],1)
        return outputs


# this is the decoder 1 without teacher focing
class loop_layer3(tf.keras.layers.Layer):
    def __init__(self, enc_units, param_dim, dec_units, batch_sz):
        super(loop_layer3, self).__init__()
        self.enc_units = enc_units
        self.param_dim = param_dim
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.decoder = Decoder(self.param_dim,self.dec_units,self.batch_sz)

    def call(self, x, enc_output, enc_hidden):
        dec_input = tf.expand_dims([tf.zeros(self.param_dim)] * self.batch_sz, 1)
        outputs = tf.expand_dims([tf.zeros(self.param_dim)] * self.batch_sz, 1)
        dec_hidden = enc_hidden

        for t in range(1, x.shape[1]):
            prediction, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            prediction = tf.expand_dims(prediction, 1)

            # dec_input = tf.expand_dims(x[:, t], 1)
            dec_input  = prediction

            outputs = tf.concat([outputs, prediction],1)
        return outputs

def make_generator_no_label(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length, max_c_length,input_dim, vis_model = True):


    # define input
    x = layers.Input(shape=(max_x_length,input_dim), batch_size=batch_sz)

    c = layers.Input(shape=(max_c_length, 1), batch_size=batch_sz, dtype=tf.float32)

    # convert to hidden latents
    loop1 = loop_layer1(input_dim, param_dim, dec_units, batch_sz)

    outputs = loop1(x)

    # define model
    model = tf.keras.Model([x,c],outputs)

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'Generator_inf.png')


    return model


def make_discriminator_no_label(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length, max_c_length, vis_model = True):
    # define input
    x = layers.Input(shape=(max_x_length, param_dim), batch_size=batch_sz)

    c = layers.Input(shape= (max_c_length,1), batch_size=batch_sz, dtype=tf.float32)

    input_dim = 2

    loop1 = loop_layer1(input_dim, param_dim, dec_units, batch_sz)

    outputs = loop1(x)

    # output to a tensor
    output = layers.Dense(1)(outputs)

    # how to output a tensor?

    model = tf.keras.Model([x, c],output)

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'discriminator.png')


    return model



def make_discriminator(enc_units, batch_sz, param_dim, dec_units, gen_path, max_x_length, max_c_length, vis_model = True):
    # define input
    x = layers.Input(shape=(max_x_length, param_dim), batch_size=batch_sz)

    c = layers.Input(shape= (max_c_length,1), batch_size=batch_sz, dtype=tf.float32)


    loop1 = loop_layer1(param_dim, param_dim, dec_units, batch_sz)

    outputs = loop1(x)

    # output to a tensor
    output = layers.Dense(1)(outputs)

    # how to output a tensor?

    model = tf.keras.Model([x, c],output)

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'discriminator.png')


    return model







