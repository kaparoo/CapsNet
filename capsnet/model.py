# -*- coding: utf-8 -*-

from capsnet.layer import DigitCap
from capsnet.layer import FeatureMap
from capsnet.layer import PrimaryCap
from capsnet.loss import MarginLoss
from capsnet.param import CapsNetParam

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from typing import List


class CapsNet(tf.keras.models.Model):
    def __init__(self, param: CapsNetParam) -> None:
        super(CapsNet, self).__init__(name="CapsNet")
        self.param = param
        self.feature_map = FeatureMap(self.param)
        self.primary_cap = PrimaryCap(self.param)
        self.digit_cap = DigitCap(self.param)

    def call(self, input_images: tf.Tensor) -> tf.Tensor:
        feature_maps = self.feature_map(input_images)
        primary_caps = self.primary_cap(feature_maps)
        digit_caps = self.digit_cap(primary_caps)
        return tf.norm(digit_caps, axis=-1, name="digit_probs")


def make_model(param: CapsNetParam,
               optimizer: optimizers.Optimizer = optimizers.Adam(),
               loss: losses.Loss = MarginLoss(),
               metrics: List[metrics.Metric] = ["accuracy"]) -> CapsNet:
    model = CapsNet(param)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
