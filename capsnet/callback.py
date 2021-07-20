# -*- coding: utf-8 -*-

import tensorflow as tf
from typing import List


def get_callbacks(checkpoint_dir: str) -> List[tf.keras.callbacks.Callback]:
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=f"{checkpoint_dir}/log.csv", append=True)
    model_save = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir +
                                                    "/{epoch:04d}.ckpt",
                                                    save_weights_only=True)
    return [csv_logger, model_save]
