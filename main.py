# -*- coding: utf-8 -*-

from absl import app
from absl import flags

import capsnet

import os
import tensorflow as tf
from typing import Tuple

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir",
                    None,
                    "Directory to save results.",
                    required=True)
flags.DEFINE_integer("num_epochs", 10, "Number of epochs.", lower_bound=1)
flags.DEFINE_float("validation_split",
                   0.2,
                   "",
                   lower_bound=0.0,
                   upper_bound=1.0)


def _get_mnist_dataset(
    num_classes: int = 10
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = tf.cast(tf.expand_dims(X_train, axis=-1), dtype=tf.float32)
    y_train = tf.one_hot(y_train, depth=num_classes, dtype=tf.float32)
    X_test = tf.cast(tf.expand_dims(X_test, axis=-1), dtype=tf.float32)
    y_test = tf.one_hot(y_test, depth=num_classes, dtype=tf.float32)
    return (X_train, y_train), (X_test, y_test)


def main(_) -> None:
    param = capsnet.make_param()

    checkpoint_dir = FLAGS.checkpoint_dir
    initial_epoch = 0

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        pass

    (X_train, y_train), _ = _get_mnist_dataset(param.num_digit)
    model = capsnet.make_model(param)

    model.fit(x=X_train,
              y=y_train,
              validation_split=FLAGS.validation_split,
              initial_epoch=initial_epoch,
              epochs=FLAGS.num_epochs + initial_epoch)

    model.save(f"{checkpoint_dir}/model")


if __name__ == "__main__":
    app.run(main)
