# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# --------------------------------------------------------

import time
import logging
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                    # logging.info(s)
                    print(s)
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg['fcis_bbox_weight']
        bias = arg['fcis_bbox_bias']
        repeat = int(bias.shape[0] / means.shape[0])
        arg['fcis_bbox_weight_test'] = weight * mx.nd.repeat(mx.nd.array(stds),
                                                             repeats=repeat).reshape((bias.shape[0], 1, 1, 1))
        arg['fcis_bbox_bias_test'] = arg['fcis_bbox_bias'] * mx.nd.repeat(mx.nd.array(stds), repeats=repeat) \
                                     + mx.nd.repeat(mx.nd.array(means), repeats=repeat)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('fcis_bbox_weight_test')
        arg.pop('fcis_bbox_bias_test')
    return _callback
