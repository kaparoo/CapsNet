# -*- coding: utf-8 -*-

from capsnet.param import CapsNetParam

import tensorflow as tf
from tensorflow.keras import backend as K


def _squash(input_vector: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """Maps the norm of `input_vector` into [0, 1] using the non.
    
    Args:
        input_vector (tf.Tensor): A target vector (or list of vectors).
        eps (float; default=1e-7): A small constant.
    """
    _norm = tf.norm(input_vector, name="squash_norm")
    _norm_sqaure = tf.square(_norm, name="squash_norm_square")
    _coef = _norm_sqaure / (_norm_sqaure + 1)
    _unit = input_vector / (_norm + eps)
    return _coef * _unit


class FeatureMap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(FeatureMap, self).__init__(name="FeatureMap")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv = tf.keras.layers.Conv2D(
            name="feature_map_conv",
            input_shape=input_shape[1:],
            filters=self.param.conv1_filter,
            kernel_size=self.param.conv1_kernel,
            strides=self.param.conv1_stride,
            activation=tf.keras.activations.relu)
        self.built = True

    def call(self, input_images: tf.Tensor) -> tf.Tensor:
        return self.conv(input_images)

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return self.conv.compute_output_shape(input_shape)


class PrimaryCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(PrimaryCap, self).__init__(name="PrimaryCap")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv = tf.keras.layers.Conv2D(
            name="primary_cap_conv",
            input_shape=input_shape[1:],
            filters=self.param.conv2_filter,
            kernel_size=self.param.conv2_kernel,
            strides=self.param.conv2_stride,
            activation=tf.keras.activations.relu)
        self.reshape = tf.keras.layers.Reshape(
            name="primary_cap_reshape",
            target_shape=[-1, self.param.dim_primary])
        self.built = True

    def call(self, feature_maps: tf.Tensor) -> tf.Tensor:
        return _squash(self.reshape(self.conv(feature_maps)))

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        output_shape = self.conv.compute_output_shape(input_shape)
        return self.reshape.compute_output_shape(output_shape)


class DigitCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(DigitCap, self).__init__(name="DigitCap")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.num_primary = self.input_shape[1]
        self.dim_primary = self.input_shape[2]
        self.num_digit = self.param.num_digit
        self.dim_digit = self.param.dim_digit

        self.W = self.add_weight(name="digit_cap_weights",
                                 shape=[
                                     self.num_digit, self.num_primary,
                                     self.dim_digit, self.dim_primary
                                 ],
                                 dtype=tf.float32,
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.built = True

    def _dynamic_routing(self, u_hat: tf.Tensor) -> tf.Tensor:
        batch_size = u_hat.shape[0]
        b = tf.zeros_like(
            K.placeholder((batch_size, self.num_digit, 1, self.num_primary)))

        for r in range(self.param.num_routings):
            c = tf.nn.softmax(b, axis=1)
            v = tf.transpose(_squash(tf.matmul(c, u_hat)), perm=[0, 1, 3, 2])
            if r < self.param.num_routings - 1:
                b += tf.transpose(tf.matmul(u_hat, v), perm=[0, 1, 3, 2])

        return v

    def call(self, primary_caps: tf.Tensor) -> tf.Tensor:
        u = tf.expand_dims(tf.tile(tf.expand_dims(primary_caps, axis=1),
                                   [1, self.param.num_digit, 1, 1]),
                           axis=-1)

        u_hat = tf.squeeze(tf.map_fn(lambda u_i: tf.matmul(self.W, u_i), u),
                           name="digit_cap_u_hat")

        v = self._dynamic_routing(u_hat)
        return tf.squeeze(v, name="digit_caps")
