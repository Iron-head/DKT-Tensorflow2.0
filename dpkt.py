import numpy as np
import tensorflow as tf

import data_util


class DKTModel(tf.keras.Model):
    def __init__(self, nb_features, nb_skills, hidden_units=128, dropout_rate=0.2):
        inputs = tf.keras.Input(shape=(None, nb_features), name='inputs')

        # Masking layers: skip the MASK_VALUE
        x = tf.keras.layers.Masking(mask_value=data_util.MASK_VALUE)(inputs)

        x = tf.keras.layers.LSTM(units=nb_skills,
                                 return_sequences=True,
                                 dropout=dropout_rate)(x)

        dense = tf.keras.layers.Dense(nb_skills, activation='sigmoid')
        output = tf.keras.layers.TimeDistributed(dense, name='output')(x)

        super(DKTModel, self).__init__(inputs=inputs,
                                       outputs=output,
                                       name='DKTModel')

    def compile(self, optimizer, metrics=None):
        def custom_loss(y_true, y_pred):
            y_true, y_pred = data_util.get_target(y_true, y_pred)
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)

        super(DKTModel, self).compile(
            loss=custom_loss,
            optimizer=optimizer,
            metrics=metrics,
            experimental_run_tf_function=False
        )

    def fit(self,
            dataset,
            epochs=1,
            verbose=1,
            callback=None,
            validation_data=None,
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1):
        return super(DKTModel, self).fit(x=dataset,
                                         epochs=epochs,
                                         verbose=verbose,
                                         callbacks=callback,
                                         validation_data=validation_data,
                                         shuffle=shuffle,
                                         initial_epoch=initial_epoch,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_steps=validation_steps,
                                         validation_freq=validation_freq)

    def evaluate(self,
                 dataset,
                 verbose=1,
                 steps=None,
                 callbacks=None):
        return super(DKTModel, self).evaluate(dataset,
                                              verbose=verbose,
                                              steps=steps,
                                              callbacks=callbacks)

    def evaluate_generator(self, *args, **kwargs):
        raise SyntaxError("Not supported")

    def fit_generator(self, *args, **kwargs):
        raise SyntaxError("Not supported")

