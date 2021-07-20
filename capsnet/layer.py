# -*- coding: utf-8 -*-

from capsnet.param import CapsNetParam

import tensorflow as tf
from tensorflow.keras import backend as K


class Squash(tf.keras.layers.Layer):
    """Non-linear squashing function (sabour et al., 2017, p. 2).
    
    Attributes:
        eps (float; default=7): A small constant for numerical stability.
    """
    def __init__(self, eps: float = 1e-7, name: str = "Squash") -> None:
        super(Squash, self).__init__(name=name)
        self.eps = eps

    def call(self, input_vector: tf.Tensor) -> tf.Tensor:
        """Maps the norm of `input_vector` into [0, 1].
        
        Args:
            input_vector (tf.Tensor): A target vector (or list of vectors).

        Returns:
            A tensor.
        """
        norm = tf.norm(input_vector,
                       axis=-1,
                       keepdims=True,
                       name="squash_norm")
        coef = norm**2 / (norm**2 + 1)
        unit = input_vector / (norm + self.eps)
        return coef * unit

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape


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
        self.squash = Squash(name="primary_cap_squash")
        self.built = True

    def call(self, feature_maps: tf.Tensor) -> tf.Tensor:
        return self.squash(self.reshape(self.conv(feature_maps)))

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        output_shape = self.conv.compute_output_shape(input_shape)
        output_shape = self.reshape.compute_output_shape(output_shape)
        return self.squash.compute_output_shape(output_shape)


class DigitCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(DigitCap, self).__init__(name="DigitCap")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        assert len(input_shape) == 3
        self.num_primary = input_shape[1]
        self.dim_primary = input_shape[2]
        self.W = self.add_weight(name="digit_cap_W",
                                 shape=[
                                     self.param.num_digit, self.num_primary,
                                     self.param.dim_digit, self.dim_primary
                                 ],
                                 dtype=tf.float32,
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.squash = Squash(name="digit_cap_squash")
        self.built = True

    def _dynamic_routing(self, u_hat: tf.Tensor) -> tf.Tensor:
        b_shape = K.placeholder(
            (u_hat.shape[0], self.param.num_digit, 1, self.num_primary))
        b = tf.zeros_like(b_shape, name="digit_cap_b")

        for r in range(self.param.num_routings):
            c = tf.nn.softmax(b, axis=1, name="digit_cap_c")
            v = tf.transpose(self.squash(tf.matmul(c, u_hat)),
                             perm=[0, 1, 3, 2],
                             name="digit_cap_v")
            if r < self.param.num_routings - 1:
                b += tf.transpose(tf.matmul(u_hat, v), perm=[0, 1, 3, 2])

        return tf.squeeze(v, name="digit_caps")

    def call(self, primary_caps: tf.Tensor) -> tf.Tensor:
        assert len(primary_caps.shape) == 3
        # u.shape: [batch_size, num_digit, num_primary, dim_primary, 1]
        u = tf.expand_dims(tf.tile(tf.expand_dims(primary_caps, axis=1),
                                   [1, self.param.num_digit, 1, 1]),
                           axis=-1,
                           name="digit_cap_u")
        # u_hat.shape: [batch_size, num_digit, num_primary, dim_digit]
        u_hat = tf.squeeze(tf.map_fn(lambda u_i: tf.matmul(self.W, u_i), u),
                           name="digit_cap_u_hat")
        return self._dynamic_routing(u_hat)

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape(
            shape=[input_shape[0], self.param.num_digit, self.param.dim_digit])
