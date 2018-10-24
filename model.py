import keras
import keras.backend as K
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from keras_contrib.layers import CRF


def attention_loss(factor=1e-6):
    def attention_regularizer(y, y_pred):
        input_len = K.shape(y_pred)[-1]
        return factor * K.square(K.batch_dot(y_pred, K.permute_dimensions(y_pred, (0, 2, 1)))
                                 - tf.eye(input_len))
    return attention_regularizer


def build_model(token_num,
                tag_num,
                embedding_dim=100,
                embedding_weights=None,
                rnn_units=100,
                return_attention=False,
                lr=1e-3):
    """Build the model for predicting tags.

    :param token_num: Number of tokens in the word dictionary.
    :param tag_num: Number of tags.
    :param embedding_dim: The output dimension of the embedding layer.
    :param embedding_weights: Initial weights for embedding layer.
    :param rnn_units: The number of RNN units in a single direction.
    :param return_attention: Whether to return the attention matrix.
    :param lr: Learning rate of optimizer.
    :return model: The built model.
    """
    if embedding_weights is not None and not isinstance(embedding_weights, list):
        embedding_weights = [embedding_weights]

    input_layer = keras.layers.Input(shape=(None,))
    embd_layer = keras.layers.Embedding(input_dim=token_num,
                                        output_dim=embedding_dim,
                                        mask_zero=True,
                                        weights=embedding_weights,
                                        trainable=embedding_weights is None,
                                        name='Embedding')(input_layer)
    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=rnn_units,
                                                              recurrent_dropout=0.4,
                                                              return_sequences=True),
                                            name='Bi-LSTM')(embd_layer)
    attention_layer = SeqSelfAttention(attention_activation='sigmoid',
                                       attention_width=9,
                                       return_attention=return_attention,
                                       name='Attention')(lstm_layer)
    if return_attention:
        attention_layer, attention = attention_layer
    crf = CRF(units=tag_num, sparse_target=True, name='CRF')

    outputs = [crf(attention_layer)]
    loss = {'CRF': crf.loss_function}
    if return_attention:
        outputs.append(attention)
        loss['Attention'] = attention_loss(1e-4) # Attention.loss(1e-4)

    model = keras.models.Model(inputs=input_layer, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=loss,
        metrics={'CRF': crf.accuracy},
    )
    return model, loss, metrics
