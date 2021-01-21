#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
from ast import literal_eval
import codecs
import collections
import json
import loguru
import math
import os
import os.path as ops
import time
import yaml

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class Config(dict):
    """
    Config class
    """
    def __init__(self, *args, **kwargs):
        """
        init class
        :param args:
        :param kwargs:
        """
        if 'config_path' in kwargs:
            config_content = self._load_config_file(kwargs['config_path'])
            super(Config, self).__init__(config_content)
        else:
            super(Config, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        """

        :param key:
        :param value:
        :param create_if_not_exist:
        :return:
        """
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        """

        :param key:
        :param create_if_not_exist:
        :return:
        """
        if key in ["immutable"]:
            return self.__dict__[key]

        if key not in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = Config()
        if isinstance(self[key], dict):
            self[key] = Config(self[key])
        return self[key]

    def __setitem__(self, key, value):
        """

        :param key:
        :param value:
        :return:
        """
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(Config, self).__setitem__(key, value)

    @staticmethod
    def _load_config_file(config_file_path):
        """

        :param config_file_path
        :return:
        """
        if not os.access(config_file_path, os.R_OK):
            raise OSError('Config file: {:s}, can not be read'.format(config_file_path))
        with open(config_file_path, 'r') as f:
            config_content = yaml.safe_load(f)

        return config_content

    def update_from_config(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, dict):
            other = Config(other)
        assert isinstance(other, Config)
        diclist = [("", other)]
        while len(diclist):
            prefix, tdic = diclist[0]
            diclist = diclist[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    diclist.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def check_and_infer(self):
        """

        :return:
        """
        if self.DATASET.IMAGE_TYPE in ['rgb', 'gray']:
            self.DATASET.DATA_DIM = 3
        elif self.DATASET.IMAGE_TYPE in ['rgba']:
            self.DATASET.DATA_DIM = 4
        else:
            raise KeyError(
                'DATASET.IMAGE_TYPE config error, only support `rgb`, `gray` and `rgba`'
            )
        if self.MEAN is not None:
            self.DATASET.PADDING_VALUE = [x * 255.0 for x in self.MEAN]

        if not self.TRAIN_CROP_SIZE:
            raise ValueError(
                'TRAIN_CROP_SIZE is empty! Please set a pair of values in format (width, height)'
            )

        if not self.EVAL_CROP_SIZE:
            raise ValueError(
                'EVAL_CROP_SIZE is empty! Please set a pair of values in format (width, height)'
            )

        # Ensure file list is use UTF-8 encoding
        train_sets = codecs.open(self.DATASET.TRAIN_FILE_LIST, 'r', 'utf-8').readlines()
        val_sets = codecs.open(self.DATASET.VAL_FILE_LIST, 'r', 'utf-8').readlines()
        test_sets = codecs.open(self.DATASET.TEST_FILE_LIST, 'r', 'utf-8').readlines()
        self.DATASET.TRAIN_TOTAL_IMAGES = len(train_sets)
        self.DATASET.VAL_TOTAL_IMAGES = len(val_sets)
        self.DATASET.TEST_TOTAL_IMAGES = len(test_sets)

        if self.MODEL.MODEL_NAME == 'icnet' and \
                len(self.MODEL.MULTI_LOSS_WEIGHT) != 3:
            self.MODEL.MULTI_LOSS_WEIGHT = [1.0, 0.4, 0.16]

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file):
        """

        :param config_file:
        :return:
        """
        with codecs.open(config_file, 'r', 'utf-8') as f:
            dic = yaml.safe_load(f)
        self.update_from_config(dic)

    def set_immutable(self, immutable):
        """

        :param immutable:
        :return:
        """
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, Config):
                value.set_immutable(immutable)

    def is_immutable(self):
        """

        :return:
        """
        return self.immutable

    def dump_to_json_file(self, f_obj):
        """

        :param f_obj:
        :return:
        """
        origin_dict = dict()
        for key, val in self.items():
            if isinstance(val, Config):
                origin_dict.update({key: dict(val)})
            elif isinstance(val, dict):
                origin_dict.update({key: val})
            else:
                raise TypeError('Not supported type {}'.format(type(val)))
        return json.dump(origin_dict, f_obj)


lanenet_lane_detection_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(lanenet_lane_detection_directory, 'config', 'tusimple_lanenet.yaml')
lanenet_cfg = Config(config_path=config_path)

CFG = lanenet_cfg


def get_logger(log_file_name_prefix):
    """

    :param log_file_name_prefix: log文件名前缀
    :return:
    """
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file_name = '{:s}_{:s}.log'.format(log_file_name_prefix, start_time)
    log_file_path = ops.join(CFG.LOG.SAVE_DIR, log_file_name)

    logger = loguru.logger
    log_level = 'INFO'
    if CFG.LOG.LEVEL == "DEBUG":
        log_level = 'DEBUG'
    elif CFG.LOG.LEVEL == "WARNING":
        log_level = 'WARNING'
    elif CFG.LOG.LEVEL == "ERROR":
        log_level = 'ERROR'

    logger.add(
        log_file_path,
        level=log_level,
        format="{time} {level} {message}",
        retention="10 days",
        rotation="1 week"
    )

    return logger


LOG = get_logger(log_file_name_prefix='lanenet_test')


class CNNBaseModel(object):
    """
    Base model for other specific cnn ctpn_models
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def depthwise_conv(input_tensor, kernel_size, name, depth_multiplier=1,
                       padding='SAME', stride=1):
        """

        :param input_tensor:
        :param kernel_size:
        :param name:
        :param depth_multiplier:
        :param padding:
        :param stride:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            padding = padding.upper()

            depthwise_filter_shape = [kernel_size, kernel_size] + [in_channel, depth_multiplier]
            w_init = tf.contrib.layers.variance_scaling_initializer()

            depthwise_filter = tf.get_variable(
                name='depthwise_filter_w', shape=depthwise_filter_shape,
                initializer=w_init
            )

            result = tf.nn.depthwise_conv2d(
                input=input_tensor,
                filter=depthwise_filter,
                strides=[1, stride, stride, 1],
                padding=padding,
                name='depthwise_conv_output'
            )
        return result

    @staticmethod
    def relu(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param data_format:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param epsilon: epsilon to avoid divide-by-zero.
        :param use_bias: whether to use the extra affine transformation or not.
        :param use_scale: whether to use the extra affine transformation or not.
        :param data_format:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param name:
        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, noise_shape=None, name=None):
        """

        :param name:
        :param inputdata:
        :param keep_prob:
        :param noise_shape:
        :return:
        """
        return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        :param inputdata:  a tensor to be flattened except for the first dimension.
        :param out_dim: output dimension
        :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
        :param b_init: initializer for b. Defaults to zero
        :param use_bias: whether to use bias.
        :param name:
        :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: tf.identity(x, name='output'),
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init,
                              trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training, name, scale=True):
        """

        :param inputdata:
        :param is_training:
        :param name:
        :param scale:
        :return:
        """

        return tf.layers.batch_normalization(inputs=inputdata, training=is_training, name=name, scale=scale)

    @staticmethod
    def layergn(inputdata, name, group_size=32, esp=1e-5):
        """

        :param inputdata:
        :param name:
        :param group_size:
        :param esp:
        :return:
        """
        with tf.variable_scope(name):
            inputdata = tf.transpose(inputdata, [0, 3, 1, 2])
            n, c, h, w = inputdata.get_shape().as_list()
            group_size = min(group_size, c)
            inputdata = tf.reshape(inputdata, [-1, group_size, c // group_size, h, w])
            mean, var = tf.nn.moments(inputdata, [2, 3, 4], keep_dims=True)
            inputdata = (inputdata - mean) / tf.sqrt(var + esp)

            # 每个通道的gamma和beta
            gamma = tf.Variable(tf.constant(1.0, shape=[c]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[c]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, c, 1, 1])
            beta = tf.reshape(beta, [1, c, 1, 1])

            # 根据论文进行转换 [n, c, h, w, c] 到 [n, h, w, c]
            output = tf.reshape(inputdata, [-1, c, h, w])
            output = output * gamma + beta
            output = tf.transpose(output, [0, 2, 3, 1])

        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """

        :param inputdata:
        :param axis:
        :param name:
        :return:
        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param activation: whether to apply a activation func to deconv result
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)
        return ret

    @staticmethod
    def dilation_conv(input_tensor, k_size, out_dims, rate, padding='SAME',
                      w_init=None, b_init=None, use_bias=False, name=None):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param rate:
        :param padding:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_tensor, filters=w, rate=rate,
                                       padding=padding, name='dilation_conv')

            if use_bias:
                ret = tf.add(conv, b)
            else:
                ret = conv

        return ret

    @staticmethod
    def spatial_dropout(input_tensor, keep_prob, is_training, name, seed=1234):
        """
        空间dropout实现
        :param input_tensor:
        :param keep_prob:
        :param is_training:
        :param name:
        :param seed:
        :return:
        """

        def f1():
            input_shape = input_tensor.get_shape().as_list()
            noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
            return tf.nn.dropout(input_tensor, keep_prob, noise_shape, seed=seed, name="spatial_dropout")

        def f2():
            return input_tensor

        with tf.variable_scope(name_or_scope=name):

            output = tf.cond(is_training, f1, f2)

            return output

    @staticmethod
    def lrelu(inputdata, name, alpha=0.2):
        """

        :param inputdata:
        :param alpha:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            return tf.nn.relu(inputdata) - alpha * tf.nn.relu(-inputdata)


class _StemBlock(CNNBaseModel):
    """
    implementation of stem block module
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_StemBlock, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        with tf.variable_scope(name_or_scope=name_scope):
            input_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=output_channels,
                stride=2,
                name='conv_block_1',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            with tf.variable_scope(name_or_scope='downsample_branch_left'):
                branch_left_output = self._conv_block(
                    input_tensor=input_tensor,
                    k_size=1,
                    output_channels=int(output_channels / 2),
                    stride=1,
                    name='1x1_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=True
                )
                branch_left_output = self._conv_block(
                    input_tensor=branch_left_output,
                    k_size=3,
                    output_channels=output_channels,
                    stride=2,
                    name='3x3_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=True
                )
            with tf.variable_scope(name_or_scope='downsample_branch_right'):
                branch_right_output = self.maxpooling(
                    inputdata=input_tensor,
                    kernel_size=3,
                    stride=2,
                    padding=self._padding,
                    name='maxpooling_block'
                )
            result = tf.concat([branch_left_output, branch_right_output], axis=-1, name='concate_features')
            result = self._conv_block(
                input_tensor=result,
                k_size=3,
                output_channels=output_channels,
                stride=1,
                name='final_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
        return result


class _ContextEmbedding(CNNBaseModel):
    """
    implementation of context embedding module in bisenetv2
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_ContextEmbedding, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = input_tensor.get_shape().as_list()[-1]
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        with tf.variable_scope(name_or_scope=name_scope):
            result = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True, name='global_avg_pooling')
            result = self.layerbn(result, self._is_training, 'bn')
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='conv_block_1',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = tf.add(result, input_tensor, name='fused_features')
            result = self.conv2d(
                inputdata=result,
                out_channel=output_channels,
                kernel_size=3,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='final_conv_block'
            )
        return result


class _GatherExpansion(CNNBaseModel):
    """
    implementation of gather and expansion module in bisenetv2
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_GatherExpansion, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'
        self._stride = 1
        self._expansion_factor = 6

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def _apply_ge_when_stride_equal_one(self, input_tensor, e, name):
        """

        :param input_tensor:
        :param e:
        :param name
        :return:
        """
        input_tensor_channels = input_tensor.get_shape().as_list()[-1]
        with tf.variable_scope(name_or_scope=name):
            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=input_tensor_channels,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.depthwise_conv(
                input_tensor=result,
                kernel_size=3,
                depth_multiplier=e,
                padding=self._padding,
                stride=1,
                name='depthwise_conv_block'
            )
            result = self.layerbn(result, self._is_training, name='dw_bn')
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=input_tensor_channels,
                stride=1,
                name='1x1_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=False
            )
            result = tf.add(input_tensor, result, name='fused_features')
            result = self.relu(result, name='ge_output')
        return result

    def _apply_ge_when_stride_equal_two(self, input_tensor, output_channels, e, name):
        """

        :param input_tensor:
        :param output_channels:
        :param e:
        :param name
        :return:
        """
        input_tensor_channels = input_tensor.get_shape().as_list()[-1]
        with tf.variable_scope(name_or_scope=name):
            input_proj = self.depthwise_conv(
                input_tensor=input_tensor,
                kernel_size=3,
                name='input_project_dw_conv_block',
                depth_multiplier=1,
                padding=self._padding,
                stride=self._stride
            )
            input_proj = self.layerbn(input_proj, self._is_training, name='input_project_bn')
            input_proj = self._conv_block(
                input_tensor=input_proj,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='input_project_1x1_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=False
            )

            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=input_tensor_channels,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.depthwise_conv(
                input_tensor=result,
                kernel_size=3,
                depth_multiplier=e,
                padding=self._padding,
                stride=2,
                name='depthwise_conv_block_1'
            )
            result = self.layerbn(result, self._is_training, name='dw_bn_1')
            result = self.depthwise_conv(
                input_tensor=result,
                kernel_size=3,
                depth_multiplier=1,
                padding=self._padding,
                stride=1,
                name='depthwise_conv_block_2'
            )
            result = self.layerbn(result, self._is_training, name='dw_bn_2')
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='1x1_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=False
            )
            result = tf.add(input_proj, result, name='fused_features')
            result = self.relu(result, name='ge_output')
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = input_tensor.get_shape().as_list()[-1]
        if 'output_channels' in kwargs:
            output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        if 'stride' in kwargs:
            self._stride = kwargs['stride']
        if 'e' in kwargs:
            self._expansion_factor = kwargs['e']

        with tf.variable_scope(name_or_scope=name_scope):
            if self._stride == 1:
                result = self._apply_ge_when_stride_equal_one(
                    input_tensor=input_tensor,
                    e=self._expansion_factor,
                    name='stride_equal_one_module'
                )
            elif self._stride == 2:
                result = self._apply_ge_when_stride_equal_two(
                    input_tensor=input_tensor,
                    output_channels=output_channels,
                    e=self._expansion_factor,
                    name='stride_equal_two_module'
                )
            else:
                raise NotImplementedError('No function matched with stride of {}'.format(self._stride))
        return result


class _GuidedAggregation(CNNBaseModel):
    """
    implementation of guided aggregation module in bisenetv2
    """

    def __init__(self, phase):
        """

        :param phase:
        """
        super(_GuidedAggregation, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        detail_input_tensor = kwargs['detail_input_tensor']
        semantic_input_tensor = kwargs['semantic_input_tensor']
        name_scope = kwargs['name']
        output_channels = detail_input_tensor.get_shape().as_list()[-1]
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        with tf.variable_scope(name_or_scope=name_scope):
            with tf.variable_scope(name_or_scope='detail_branch'):
                detail_branch_remain = self.depthwise_conv(
                    input_tensor=detail_input_tensor,
                    kernel_size=3,
                    name='3x3_dw_conv_block',
                    depth_multiplier=1,
                    padding=self._padding,
                    stride=1
                )
                detail_branch_remain = self.layerbn(detail_branch_remain, self._is_training, name='bn_1')
                detail_branch_remain = self.conv2d(
                    inputdata=detail_branch_remain,
                    out_channel=output_channels,
                    kernel_size=1,
                    padding=self._padding,
                    stride=1,
                    use_bias=False,
                    name='1x1_conv_block'
                )

                detail_branch_downsample = self._conv_block(
                    input_tensor=detail_input_tensor,
                    k_size=3,
                    output_channels=output_channels,
                    stride=2,
                    name='3x3_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=False
                )
                detail_branch_downsample = self.avgpooling(
                    inputdata=detail_branch_downsample,
                    kernel_size=3,
                    stride=2,
                    padding=self._padding,
                    name='avg_pooling_block'
                )

            with tf.variable_scope(name_or_scope='semantic_branch'):
                semantic_branch_remain = self.depthwise_conv(
                    input_tensor=semantic_input_tensor,
                    kernel_size=3,
                    name='3x3_dw_conv_block',
                    depth_multiplier=1,
                    padding=self._padding,
                    stride=1
                )
                semantic_branch_remain = self.layerbn(semantic_branch_remain, self._is_training, name='bn_1')
                semantic_branch_remain = self.conv2d(
                    inputdata=semantic_branch_remain,
                    out_channel=output_channels,
                    kernel_size=1,
                    padding=self._padding,
                    stride=1,
                    use_bias=False,
                    name='1x1_conv_block'
                )
                semantic_branch_remain = self.sigmoid(semantic_branch_remain, name='semantic_remain_sigmoid')

                semantic_branch_upsample = self._conv_block(
                    input_tensor=semantic_input_tensor,
                    k_size=3,
                    output_channels=output_channels,
                    stride=1,
                    name='3x3_conv_block',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=False
                )
                semantic_branch_upsample = tf.image.resize_bilinear(
                    semantic_branch_upsample,
                    detail_input_tensor.shape[1:3],
                    name='semantic_upsample_features'
                )
                semantic_branch_upsample = self.sigmoid(semantic_branch_upsample, name='semantic_upsample_sigmoid')

            with tf.variable_scope(name_or_scope='aggregation_features'):
                guided_features_remain = tf.multiply(
                    detail_branch_remain,
                    semantic_branch_upsample,
                    name='guided_detail_features'
                )
                guided_features_downsample = tf.multiply(
                    detail_branch_downsample,
                    semantic_branch_remain,
                    name='guided_semantic_features'
                )
                guided_features_upsample = tf.image.resize_bilinear(
                    guided_features_downsample,
                    detail_input_tensor.shape[1:3],
                    name='guided_upsample_features'
                )
                guided_features = tf.add(guided_features_remain, guided_features_upsample, name='fused_features')
                guided_features = self._conv_block(
                    input_tensor=guided_features,
                    k_size=3,
                    output_channels=output_channels,
                    stride=1,
                    name='aggregation_feature_output',
                    padding=self._padding,
                    use_bias=False,
                    need_activate=True
                )
        return guided_features


class _SegmentationHead(CNNBaseModel):
    """
    implementation of segmentation head in bisenet v2
    """
    def __init__(self, phase):
        """

        """
        super(_SegmentationHead, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        ratio = kwargs['upsample_ratio']
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * ratio) for tmp in input_tensor_size]
        feature_dims = kwargs['feature_dims']
        classes_nums = kwargs['classes_nums']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        with tf.variable_scope(name_or_scope=name_scope):
            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=feature_dims,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.conv2d(
                inputdata=result,
                out_channel=classes_nums,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='1x1_conv_block'
            )
            result = tf.image.resize_bilinear(
                result,
                output_tensor_size,
                name='segmentation_head_logits'
            )
        return result


class BiseNetV2(CNNBaseModel):
    """
    implementation of bisenet v2
    """
    def __init__(self, phase, cfg):
        """

        """
        super(BiseNetV2, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        # set model hyper params
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._enable_ohem = self._cfg.SOLVER.OHEM.ENABLE
        if self._enable_ohem:
            self._ohem_score_thresh = self._cfg.SOLVER.OHEM.SCORE_THRESH
            self._ohem_min_sample_nums = self._cfg.SOLVER.OHEM.MIN_SAMPLE_NUMS
        self._ge_expand_ratio = self._cfg.MODEL.BISENETV2.GE_EXPAND_RATIO
        self._semantic_channel_ratio = self._cfg.MODEL.BISENETV2.SEMANTIC_CHANNEL_LAMBDA
        self._seg_head_ratio = self._cfg.MODEL.BISENETV2.SEGHEAD_CHANNEL_EXPAND_RATIO

        # set module used in bisenetv2
        self._se_block = _StemBlock(phase=phase)
        self._context_embedding_block = _ContextEmbedding(phase=phase)
        self._ge_block = _GatherExpansion(phase=phase)
        self._guided_aggregation_block = _GuidedAggregation(phase=phase)
        self._seg_head_block = _SegmentationHead(phase=phase)

        # set detail branch channels
        self._detail_branch_channels = self._build_detail_branch_hyper_params()
        # set semantic branch channels
        self._semantic_branch_channels = self._build_semantic_branch_hyper_params()

        # set op block params
        self._block_maps = {
            'conv_block': self._conv_block,
            'se': self._se_block,
            'ge': self._ge_block,
            'ce': self._context_embedding_block,
        }

        self._net_intermediate_results = collections.OrderedDict()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @classmethod
    def _build_detail_branch_hyper_params(cls):
        """

        :return:
        """
        params = [
            ('stage_1', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 1)]),
            ('stage_2', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 2)]),
            ('stage_3', [('conv_block', 3, 128, 2, 1), ('conv_block', 3, 128, 1, 2)]),
        ]
        return collections.OrderedDict(params)

    def _build_semantic_branch_hyper_params(self):
        """

        :return:
        """
        stage_1_channels = int(self._detail_branch_channels['stage_1'][0][2] * self._semantic_channel_ratio)
        stage_3_channels = int(self._detail_branch_channels['stage_3'][0][2] * self._semantic_channel_ratio)
        params = [
            ('stage_1', [('se', 3, stage_1_channels, 1, 4, 1)]),
            ('stage_3', [('ge', 3, stage_3_channels, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels, self._ge_expand_ratio, 1, 1)]),
            ('stage_4', [('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 1, 1)]),
            ('stage_5', [('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 3),
                         ('ce', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 1)])
        ]
        return collections.OrderedDict(params)

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def build_detail_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        result = input_tensor
        with tf.variable_scope(name_or_scope=name):
            for stage_name, stage_params in self._detail_branch_channels.items():
                with tf.variable_scope(stage_name):
                    for block_index, param in enumerate(stage_params):
                        block_op = self._block_maps[param[0]]
                        k_size = param[1]
                        output_channels = param[2]
                        stride = param[3]
                        repeat_times = param[4]
                        for repeat_index in range(repeat_times):
                            with tf.variable_scope(name_or_scope='conv_block_{:d}_repeat_{:d}'.format(
                                    block_index + 1, repeat_index + 1)):
                                if stage_name == 'stage_3' and block_index == 1 and repeat_index == 1:
                                    result = block_op(
                                        input_tensor=result,
                                        k_size=k_size,
                                        output_channels=output_channels,
                                        stride=stride,
                                        name='3x3_conv',
                                        padding='SAME',
                                        use_bias=False,
                                        need_activate=False
                                    )
                                else:
                                    result = block_op(
                                        input_tensor=result,
                                        k_size=k_size,
                                        output_channels=output_channels,
                                        stride=stride,
                                        name='3x3_conv',
                                        padding='SAME',
                                        use_bias=False,
                                        need_activate=True
                                    )
        return result

    def build_semantic_branch(self, input_tensor, name, prepare_data_for_booster=False):
        """

        :param input_tensor:
        :param name:
        :param prepare_data_for_booster:
        :return:
        """
        seg_head_inputs = collections.OrderedDict()
        result = input_tensor
        source_input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        with tf.variable_scope(name_or_scope=name):
            for stage_name, stage_params in self._semantic_branch_channels.items():
                seg_head_input = input_tensor
                with tf.variable_scope(stage_name):
                    for block_index, param in enumerate(stage_params):
                        block_op_name = param[0]
                        block_op = self._block_maps[block_op_name]
                        output_channels = param[2]
                        expand_ratio = param[3]
                        stride = param[4]
                        repeat_times = param[5]
                        for repeat_index in range(repeat_times):
                            with tf.variable_scope(name_or_scope='{:s}_block_{:d}_repeat_{:d}'.format(
                                    block_op_name, block_index + 1, repeat_index + 1)):
                                if block_op_name == 'ge':
                                    result = block_op(
                                        input_tensor=result,
                                        name='gather_expansion_block',
                                        stride=stride,
                                        e=expand_ratio,
                                        output_channels=output_channels
                                    )
                                    seg_head_input = result
                                elif block_op_name == 'ce':
                                    result = block_op(
                                        input_tensor=result,
                                        name='context_embedding_block'
                                    )
                                elif block_op_name == 'se':
                                    result = block_op(
                                        input_tensor=result,
                                        output_channels=output_channels,
                                        name='stem_block'
                                    )
                                    seg_head_input = result
                                else:
                                    raise NotImplementedError('Not support block type: {:s}'.format(block_op_name))
                    if prepare_data_for_booster:
                        result_tensor_size = result.get_shape().as_list()[1:3]
                        result_tensor_dims = result.get_shape().as_list()[-1]
                        upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
                        feature_dims = result_tensor_dims * self._seg_head_ratio
                        seg_head_inputs[stage_name] = self._seg_head_block(
                            input_tensor=seg_head_input,
                            name='block_{:d}_seg_head_block'.format(block_index + 1),
                            upsample_ratio=upsample_ratio,
                            feature_dims=feature_dims,
                            classes_nums=self._class_nums
                        )
        return result, seg_head_inputs

    def build_aggregation_branch(self, detail_output, semantic_output, name):
        """

        :param detail_output:
        :param semantic_output:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self._guided_aggregation_block(
                detail_input_tensor=detail_output,
                semantic_input_tensor=semantic_output,
                name='guided_aggregation_block'
            )
        return result

    def build_instance_segmentation_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3x3',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=128,
                stride=1,
                name='conv_1x1',
                use_bias=False,
                need_activate=False
            )
            output_tensor = tf.image.resize_bilinear(
                output_tensor,
                output_tensor_size,
                name='instance_logits'
            )
        return output_tensor

    def build_binary_segmentation_branch(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * 8) for tmp in input_tensor_size]

        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3x3',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=128,
                stride=1,
                name='conv_1x1',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=self._class_nums,
                stride=1,
                name='final_conv',
                use_bias=False,
                need_activate=False
            )
            output_tensor = tf.image.resize_bilinear(
                output_tensor,
                output_tensor_size,
                name='binary_logits'
            )
        return output_tensor

    def build_model(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # build detail branch
            detail_branch_output = self.build_detail_branch(
                input_tensor=input_tensor,
                name='detail_branch'
            )
            # build semantic branch
            semantic_branch_output, _ = self.build_semantic_branch(
                input_tensor=input_tensor,
                name='semantic_branch',
                prepare_data_for_booster=False
            )
            # build aggregation branch
            aggregation_branch_output = self.build_aggregation_branch(
                detail_output=detail_branch_output,
                semantic_output=semantic_branch_output,
                name='aggregation_branch'
            )
            # build binary and instance segmentation branch
            binary_seg_branch_output = self.build_binary_segmentation_branch(
                input_tensor=aggregation_branch_output,
                name='binary_segmentation_branch'
            )
            instance_seg_branch_output = self.build_instance_segmentation_branch(
                input_tensor=aggregation_branch_output,
                name='instance_segmentation_branch'
            )
            # gather frontend output result
            self._net_intermediate_results['binary_segment_logits'] = {
                'data': binary_seg_branch_output,
                'shape': binary_seg_branch_output.get_shape().as_list()
            }
            self._net_intermediate_results['instance_segment_logits'] = {
                'data': instance_seg_branch_output,
                'shape': instance_seg_branch_output.get_shape().as_list()
            }
        return self._net_intermediate_results


class VGG16FCN(CNNBaseModel):
    """
    VGG 16 based fcn net for semantic segmentation
    """
    def __init__(self, phase, cfg):
        """

        """
        super(VGG16FCN, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._net_intermediate_results = collections.OrderedDict()
        self._class_nums = self._cfg.DATASET.NUM_CLASSES

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _vgg16_conv_stage(self, input_tensor, k_size, out_dims, name,
                          stride=1, pad='SAME', need_layer_norm=True):
        """
        stack conv and activation in vgg16
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :param need_layer_norm:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(
                inputdata=input_tensor, out_channel=out_dims,
                kernel_size=k_size, stride=stride,
                use_bias=False, padding=pad, name='conv'
            )

            if need_layer_norm:
                bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

                relu = self.relu(inputdata=bn, name='relu')
            else:
                relu = self.relu(inputdata=conv, name='relu')

        return relu

    def _decode_block(self, input_tensor, previous_feats_tensor,
                      out_channels_nums, name, kernel_size=4,
                      stride=2, use_bias=False,
                      previous_kernel_size=4, need_activate=True):
        """

        :param input_tensor:
        :param previous_feats_tensor:
        :param out_channels_nums:
        :param kernel_size:
        :param previous_kernel_size:
        :param use_bias:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):

            deconv_weights_stddev = tf.sqrt(
                tf.divide(tf.constant(2.0, tf.float32),
                          tf.multiply(tf.cast(previous_kernel_size * previous_kernel_size, tf.float32),
                                      tf.cast(tf.shape(input_tensor)[3], tf.float32)))
            )
            deconv_weights_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=deconv_weights_stddev)

            deconv = self.deconv2d(
                inputdata=input_tensor, out_channel=out_channels_nums, kernel_size=kernel_size,
                stride=stride, use_bias=use_bias, w_init=deconv_weights_init,
                name='deconv'
            )

            deconv = self.layerbn(inputdata=deconv, is_training=self._is_training, name='deconv_bn')

            deconv = self.relu(inputdata=deconv, name='deconv_relu')

            fuse_feats = tf.add(
                previous_feats_tensor, deconv, name='fuse_feats'
            )

            if need_activate:

                fuse_feats = self.layerbn(
                    inputdata=fuse_feats, is_training=self._is_training, name='fuse_gn'
                )

                fuse_feats = self.relu(inputdata=fuse_feats, name='fuse_relu')

        return fuse_feats

    def _vgg16_fcn_encode(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # encode stage 1
            conv_1_1 = self._vgg16_conv_stage(
                input_tensor=input_tensor, k_size=3,
                out_dims=64, name='conv1_1',
                need_layer_norm=True
            )
            conv_1_2 = self._vgg16_conv_stage(
                input_tensor=conv_1_1, k_size=3,
                out_dims=64, name='conv1_2',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_1_share'] = {
                'data': conv_1_2,
                'shape': conv_1_2.get_shape().as_list()
            }

            # encode stage 2
            pool1 = self.maxpooling(
                inputdata=conv_1_2, kernel_size=2,
                stride=2, name='pool1'
            )
            conv_2_1 = self._vgg16_conv_stage(
                input_tensor=pool1, k_size=3,
                out_dims=128, name='conv2_1',
                need_layer_norm=True
            )
            conv_2_2 = self._vgg16_conv_stage(
                input_tensor=conv_2_1, k_size=3,
                out_dims=128, name='conv2_2',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_2_share'] = {
                'data': conv_2_2,
                'shape': conv_2_2.get_shape().as_list()
            }

            # encode stage 3
            pool2 = self.maxpooling(
                inputdata=conv_2_2, kernel_size=2,
                stride=2, name='pool2'
            )
            conv_3_1 = self._vgg16_conv_stage(
                input_tensor=pool2, k_size=3,
                out_dims=256, name='conv3_1',
                need_layer_norm=True
            )
            conv_3_2 = self._vgg16_conv_stage(
                input_tensor=conv_3_1, k_size=3,
                out_dims=256, name='conv3_2',
                need_layer_norm=True
            )
            conv_3_3 = self._vgg16_conv_stage(
                input_tensor=conv_3_2, k_size=3,
                out_dims=256, name='conv3_3',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_3_share'] = {
                'data': conv_3_3,
                'shape': conv_3_3.get_shape().as_list()
            }

            # encode stage 4
            pool3 = self.maxpooling(
                inputdata=conv_3_3, kernel_size=2,
                stride=2, name='pool3'
            )
            conv_4_1 = self._vgg16_conv_stage(
                input_tensor=pool3, k_size=3,
                out_dims=512, name='conv4_1',
                need_layer_norm=True
            )
            conv_4_2 = self._vgg16_conv_stage(
                input_tensor=conv_4_1, k_size=3,
                out_dims=512, name='conv4_2',
                need_layer_norm=True
            )
            conv_4_3 = self._vgg16_conv_stage(
                input_tensor=conv_4_2, k_size=3,
                out_dims=512, name='conv4_3',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_4_share'] = {
                'data': conv_4_3,
                'shape': conv_4_3.get_shape().as_list()
            }

            # encode stage 5 for binary segmentation
            pool4 = self.maxpooling(
                inputdata=conv_4_3, kernel_size=2,
                stride=2, name='pool4'
            )
            conv_5_1_binary = self._vgg16_conv_stage(
                input_tensor=pool4, k_size=3,
                out_dims=512, name='conv5_1_binary',
                need_layer_norm=True
            )
            conv_5_2_binary = self._vgg16_conv_stage(
                input_tensor=conv_5_1_binary, k_size=3,
                out_dims=512, name='conv5_2_binary',
                need_layer_norm=True
            )
            conv_5_3_binary = self._vgg16_conv_stage(
                input_tensor=conv_5_2_binary, k_size=3,
                out_dims=512, name='conv5_3_binary',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_5_binary'] = {
                'data': conv_5_3_binary,
                'shape': conv_5_3_binary.get_shape().as_list()
            }

            # encode stage 5 for instance segmentation
            conv_5_1_instance = self._vgg16_conv_stage(
                input_tensor=pool4, k_size=3,
                out_dims=512, name='conv5_1_instance',
                need_layer_norm=True
            )
            conv_5_2_instance = self._vgg16_conv_stage(
                input_tensor=conv_5_1_instance, k_size=3,
                out_dims=512, name='conv5_2_instance',
                need_layer_norm=True
            )
            conv_5_3_instance = self._vgg16_conv_stage(
                input_tensor=conv_5_2_instance, k_size=3,
                out_dims=512, name='conv5_3_instance',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_5_instance'] = {
                'data': conv_5_3_instance,
                'shape': conv_5_3_instance.get_shape().as_list()
            }

        return

    def _vgg16_fcn_decode(self, name):
        """

        :return:
        """
        with tf.variable_scope(name):

            # decode part for binary segmentation
            with tf.variable_scope(name_or_scope='binary_seg_decode'):

                decode_stage_5_binary = self._net_intermediate_results['encode_stage_5_binary']['data']

                decode_stage_4_fuse = self._decode_block(
                    input_tensor=decode_stage_5_binary,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_4_share']['data'],
                    name='decode_stage_4_fuse', out_channels_nums=512, previous_kernel_size=3
                )
                decode_stage_3_fuse = self._decode_block(
                    input_tensor=decode_stage_4_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_3_share']['data'],
                    name='decode_stage_3_fuse', out_channels_nums=256
                )
                decode_stage_2_fuse = self._decode_block(
                    input_tensor=decode_stage_3_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_2_share']['data'],
                    name='decode_stage_2_fuse', out_channels_nums=128
                )
                decode_stage_1_fuse = self._decode_block(
                    input_tensor=decode_stage_2_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_1_share']['data'],
                    name='decode_stage_1_fuse', out_channels_nums=64
                )
                binary_final_logits_conv_weights_stddev = tf.sqrt(
                    tf.divide(tf.constant(2.0, tf.float32),
                              tf.multiply(4.0 * 4.0,
                                          tf.cast(tf.shape(decode_stage_1_fuse)[3], tf.float32)))
                )
                binary_final_logits_conv_weights_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=binary_final_logits_conv_weights_stddev)

                binary_final_logits = self.conv2d(
                    inputdata=decode_stage_1_fuse,
                    out_channel=self._class_nums,
                    kernel_size=1, use_bias=False,
                    w_init=binary_final_logits_conv_weights_init,
                    name='binary_final_logits'
                )

                self._net_intermediate_results['binary_segment_logits'] = {
                    'data': binary_final_logits,
                    'shape': binary_final_logits.get_shape().as_list()
                }

            with tf.variable_scope(name_or_scope='instance_seg_decode'):

                decode_stage_5_instance = self._net_intermediate_results['encode_stage_5_instance']['data']

                decode_stage_4_fuse = self._decode_block(
                    input_tensor=decode_stage_5_instance,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_4_share']['data'],
                    name='decode_stage_4_fuse', out_channels_nums=512, previous_kernel_size=3)

                decode_stage_3_fuse = self._decode_block(
                    input_tensor=decode_stage_4_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_3_share']['data'],
                    name='decode_stage_3_fuse', out_channels_nums=256)

                decode_stage_2_fuse = self._decode_block(
                    input_tensor=decode_stage_3_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_2_share']['data'],
                    name='decode_stage_2_fuse', out_channels_nums=128)

                decode_stage_1_fuse = self._decode_block(
                    input_tensor=decode_stage_2_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_1_share']['data'],
                    name='decode_stage_1_fuse', out_channels_nums=64, need_activate=False)

                self._net_intermediate_results['instance_segment_logits'] = {
                    'data': decode_stage_1_fuse,
                    'shape': decode_stage_1_fuse.get_shape().as_list()
                }

    def build_model(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # vgg16 fcn encode part
            self._vgg16_fcn_encode(input_tensor=input_tensor, name='vgg16_encode_module')
            # vgg16 fcn decode part
            self._vgg16_fcn_decode(name='vgg16_decode_module')

        return self._net_intermediate_results


class LaneNetFrondEnd(CNNBaseModel):
    """
    LaneNet frontend which is used to extract image features for following process
    """
    def __init__(self, phase, net_flag, cfg):
        """

        """
        super(LaneNetFrondEnd, self).__init__()
        self._cfg = cfg

        self._frontend_net_map = {
            'vgg': VGG16FCN(phase=phase, cfg=self._cfg),
            'bisenetv2': BiseNetV2(phase=phase, cfg=self._cfg),
        }

        self._net = self._frontend_net_map[net_flag]

    def build_model(self, input_tensor, name, reuse):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """

        return self._net.build_model(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse
        )


def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):
    """
    discriminative loss
    :param prediction: inference of network
    :param correct_label: instance label
    :param feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cut off variance distance
    :param delta_d: cut off cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    """
    correct_label = tf.reshape(
        correct_label, [label_shape[1] * label_shape[0]]
    )
    reshaped_pred = tf.reshape(
        prediction, [label_shape[1] * label_shape[0], feature_dim]
    )

    # calculate instance nums
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    # calculate instance pixel embedding mean vec
    segmented_sum = tf.unsorted_segment_sum(
        reshaped_pred, unique_id, num_instances)
    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1, ord=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(
        mu_band_rep,
        (num_instances *
         num_instances,
         feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, axis=1, ord=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    l_reg = tf.reduce_mean(tf.norm(mu, axis=1, ord=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    """

    :return: discriminative loss and its three components
    """

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape, delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(
        cond, body, [
            correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg, 0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


class LaneNetBackEnd(CNNBaseModel):
    """
    LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
    """

    def __init__(self, phase, cfg):
        """
        init lanenet backend
        :param phase: train or test
        """
        super(LaneNetBackEnd, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._embedding_dims = self._cfg.MODEL.EMBEDDING_FEATS_DIMS
        self._binary_loss_type = self._cfg.SOLVER.LOSS_TYPE

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )

        return loss

    @classmethod
    def _multi_category_focal_loss(cls, onehot_labels, logits, classes_weights, gamma=2.0):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :param gamma:
        :return:
        """
        epsilon = 1.e-7
        alpha = tf.multiply(onehot_labels, classes_weights)
        alpha = tf.cast(alpha, tf.float32)
        gamma = float(gamma)
        y_true = tf.cast(onehot_labels, tf.float32)
        y_pred = tf.nn.softmax(logits, dim=-1)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)

        return loss

    def compute_loss(self, binary_seg_logits, binary_label,
                     instance_seg_logits, instance_label,
                     name, reuse):
        """
        compute lanenet loss
        :param binary_seg_logits:
        :param binary_label:
        :param instance_seg_logits:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_label_onehot = tf.one_hot(
                    tf.reshape(
                        tf.cast(binary_label, tf.int32),
                        shape=[binary_label.get_shape().as_list()[0],
                               binary_label.get_shape().as_list()[1],
                               binary_label.get_shape().as_list()[2]]),
                    depth=self._class_nums,
                    axis=-1
                )

                binary_label_plain = tf.reshape(
                    binary_label,
                    shape=[binary_label.get_shape().as_list()[0] *
                           binary_label.get_shape().as_list()[1] *
                           binary_label.get_shape().as_list()[2] *
                           binary_label.get_shape().as_list()[3]])
                unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
                counts = tf.cast(counts, tf.float32)
                inverse_weights = tf.divide(
                    1.0,
                    tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
                )
                if self._binary_loss_type == 'cross_entropy':
                    binary_segmenatation_loss = self._compute_class_weighted_cross_entropy_loss(
                        onehot_labels=binary_label_onehot,
                        logits=binary_seg_logits,
                        classes_weights=inverse_weights
                    )
                elif self._binary_loss_type == 'focal':
                    binary_segmenatation_loss = self._multi_category_focal_loss(
                        onehot_labels=binary_label_onehot,
                        logits=binary_seg_logits,
                        classes_weights=inverse_weights
                    )
                else:
                    raise NotImplementedError

            # calculate class weighted instance seg loss
            with tf.variable_scope(name_or_scope='instance_seg'):

                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                pix_embedding = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=self._embedding_dims,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )
                pix_image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
                instance_segmentation_loss, l_var, l_dist, l_reg = \
                    discriminative_loss(
                        pix_embedding, instance_label, self._embedding_dims,
                        pix_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
                    )

            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = binary_segmenatation_loss + instance_segmentation_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': binary_seg_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': instance_segmentation_loss
            }

        return ret

    def inference(self, binary_seg_logits, instance_seg_logits, name, reuse):
        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)

            with tf.variable_scope(name_or_scope='instance_seg'):
                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                instance_seg_prediction = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=self._embedding_dims,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )

        return binary_seg_prediction, instance_seg_prediction


class LaneNet(CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        """
        super(LaneNet, self).__init__()
        self._cfg = cfg
        self._net_flag = self._cfg.MODEL.FRONT_END

        self._frontend = LaneNetFrondEnd(
            phase=phase, net_flag=self._net_flag, cfg=self._cfg
        )
        self._backend = LaneNetBackEnd(
            phase=phase, cfg=self._cfg
        )

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

        return binary_seg_prediction, instance_seg_prediction

    def compute_loss(self, input_tensor, binary_label, instance_label, name, reuse=False):
        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

        return calculated_losses


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, cfg, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        # lane line fit
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple')
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        for index, single_lane_pts in enumerate(src_lane_pts):
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            else:
                raise ValueError('Wrong data source now only support tusimple')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                idx_low = np.argmax(fake_diff_smaller_than_zero)
                idx_high = np.argmin(fake_diff_bigger_than_zero)

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue

                lane_color = self._color_map[index].tolist()
                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
        }

        return ret


def test_lanenet(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        loop_times = 1
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )
        mask_image = postprocess_result['mask_image']

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])

        embedding_image = np.array(instance_seg_image[0], np.uint8)
        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

        cv2.imwrite("mask_image.png", mask_image)
        cv2.imwrite("src_image.png", image_vis)
        cv2.imwrite("instance_image.png", embedding_image)
        cv2.imwrite("binary_image.png", binary_seg_image[0] * 255)
    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    lanenet_dir = os.path.dirname(tools_dir)
    image_path = ops.join(lanenet_dir, "data", "custom_data", "image-001.jpeg")
    weights_path = ops.join(lanenet_dir, "model", "tusimple_lanenet", "tusimple_lanenet.ckpt")
    test_lanenet(image_path, weights_path)
