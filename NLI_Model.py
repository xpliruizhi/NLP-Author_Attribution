from keras.models import Model
from keras.layers import Input, Dense, LSTM, GlobalAvgPool1D, CuDNNLSTM,Permute
from keras.layers import GlobalMaxPool1D, Embedding, Bidirectional,Dot, Lambda, Softmax
from keras.layers import Concatenate, Subtract, Multiply, Dropout,Layer
import keras.backend as K
import tensorflow as tf
class ESIM:
    def __init__(self, n_classes, max_sequence_length, embedding_matrix, voc_size, learning_rate=0.0004, use_gpu=False):
        self._max_sequence_length = max_sequence_length
        self._learning_rate = learning_rate
        self._n_classes = n_classes
        self._embedding_matrix = embedding_matrix
        self._voc_size = voc_size
        self._use_gpu = use_gpu

        self._inputEncodingBlock = None
        self._localInferenceBlock = None
        self._compositionBlock = None

        self._premise = Input(name='premise', shape=(self._max_sequence_length,), dtype='int32')
        self._hypothesis = Input(name='hypothesis', shape=(self._max_sequence_length,), dtype='int32')
        self.model = self.build_ESIM_model()

    def _input_encoding_block(self):

        embedding_layer = Embedding(self._voc_size + 1,
                                    300,weights=[self._embedding_matrix],
                                    input_length=self._max_sequence_length,
                                    trainable=True, mask_zero=True)

        premise_embedded_sequences = embedding_layer(self._premise)
        hypothesis_embedded_sequences = embedding_layer(self._hypothesis)

        if self._use_gpu:
            encoding_layer = Bidirectional(CuDNNLSTM(300,return_sequences=True))
        else:
            encoding_layer = Bidirectional(LSTM(300, dropout=0.5, return_sequences=True))

        a_bar = encoding_layer(premise_embedded_sequences)
        b_bar = encoding_layer(hypothesis_embedded_sequences)

        return a_bar, b_bar

    def _local_inference_block(self, a_bar, b_bar):
        attention_weights = Dot(axes=-1)([a_bar, b_bar])

        weight_b = Softmax(axis=1)(attention_weights)
        ## Score转成(0-1), 
        weight_a = Permute((2,1))(Softmax(axis=2)(attention_weights))

        b_aligned = Dot(axes=1)([weight_b, a_bar])

        a_aligned = Dot(axes=1)([weight_a, b_bar])

        m_a = Concatenate()([a_bar, a_aligned, Subtract()([a_bar, a_aligned]), Multiply()([a_bar, a_aligned])])

        m_b = Concatenate()([b_bar, b_aligned, Subtract()([b_bar, b_aligned]), Multiply()([b_bar, b_aligned])])
        return m_a, m_b


    def _inference_composition_block(self, m_a, m_b):
        y_a = Bidirectional(LSTM(300, return_sequences=True))(m_a)
        y_b = Bidirectional(LSTM(300, return_sequences=True))(m_b)

        class GlobalAvgPool1DMasked(Layer):
            def __init__(self, **kwargs):
                self.supports_masking = True
                super(GlobalAvgPool1DMasked, self).__init__(**kwargs)

            def compute_mask(self, inputs, mask=None):
                return None

            def call(self, inputs, mask=None):
                if mask is not None:
                    mask = K.cast(mask, K.floatx())
                    mask = K.repeat(mask, inputs.shape[-1])
                    mask = tf.transpose(mask, [0, 2, 1])
                    inputs = inputs * mask
                    return K.sum(inputs, axis=1) / K.sum(mask, axis=1)
                else:
                    print('not mask average!')
                    return super().call(inputs)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[2])

        class GlobalMaxPool1DMasked(GlobalMaxPool1D):
            def __init__(self, **kwargs):
                self.supports_masking = True
                super(GlobalMaxPool1DMasked, self).__init__(**kwargs)

            def compute_mask(self, inputs, mask=None):
                return None

            def call(self, inputs, mask=None):
                return super(GlobalMaxPool1DMasked, self).call(inputs)

        max_pooling_a = GlobalMaxPool1D()(y_a)
        avg_pooling_a = GlobalAvgPool1D()(y_a)

        max_pooling_b = GlobalMaxPool1D()(y_b)
        avg_pooling_b = GlobalAvgPool1D()(y_b)

        y = Concatenate()([avg_pooling_a, max_pooling_a, avg_pooling_b, max_pooling_b])
        y = Dense(1024, activation='tanh')(y)
        ### 1024 神经元个数 
        y = Dropout(0.5)(y)
        y = Dense(1024, activation='tanh')(y)
        ### 1024 
        y = Dropout(0.5)(y)
        y = Dense(self._n_classes, activation='softmax')(y)
        return y

    def build_ESIM_model(self):
        a_bar, b_bar = self._input_encoding_block()
        m_a, m_b = self._local_inference_block(a_bar, b_bar)
        y = self._inference_composition_block(m_a, m_b)
        model = Model(inputs=[self._premise, self._hypothesis], outputs=[y])

        print(model.summary())
        return model
        
        
        





