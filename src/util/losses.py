import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
class MultiQuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, penalty_weight=0.1, name="multi_quantile_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles
        self.penalty_weight = penalty_weight
        self.q = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)

        error = y_true - y_pred

        loss = tf.maximum(self.q * error, (self.q - 1) * error)
        base_loss = tf.reduce_mean(loss)

        # crossing penalty
        diff = y_pred[:, :-1] - y_pred[:, 1:]
        crossing_penalty = tf.reduce_mean(tf.maximum(0.0, diff))

        return base_loss + self.penalty_weight * crossing_penalty

    def get_config(self):
        return {
            "quantiles": self.quantiles,
            "penalty_weight": self.penalty_weight,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SingleQuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantile, name="single_quantile_loss"):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(self.quantile * error, (self.quantile - 1) * error)
        )

    def get_config(self):
        return {
            "quantile": self.quantile,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)