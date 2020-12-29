import os
import numpy as np
import tensorflow as tf
from autoencoders_evaluate import eval_custom_CV


class CustomMetrics(tf.keras.callbacks.Callback):

    def __init__(self, dataset, path, ref_data, monitor=None, mode=None):
        super().__init__()
        self.dataset = dataset
        self.ref_data = ref_data

        self.val_writer = tf.summary.create_file_writer(
            os.path.join(path, 'validation'))
        # Used to write best metric on_train_end for AI platform
        self.monitor = monitor
        self.mode = mode
        if self.monitor and self.mode:
            if mode == 'min':
                self.monitor_op = np.less
            else:
                self.monitor_op = np.greater

            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        # Calculate custom metrics
        self.epoch = epoch
        average_auc, sd_auc = eval_custom_CV(self.ref_data, self.dataset,
                                             self.model)
        with self.val_writer.as_default():
            # Write to tensorboard
            tf.summary.scalar('average_auc', average_auc, step=epoch)
            tf.summary.scalar('sd_auc', sd_auc, step=epoch)

            # Append to logs for other callbacks
            logs['average_auc'] = average_auc
            logs['sd_auc'] = sd_auc

        if self.monitor and self.mode:
            if self.monitor_op(result[self.monitor], self.best):
                self.best = result[self.monitor]

    def on_train_end(self, logs={}):
        if self.monitor and self.mode:
            with self.val_writer.as_default():
                tf.summary.scalar(self.monitor, self.best, step=self.epoch + 1)
