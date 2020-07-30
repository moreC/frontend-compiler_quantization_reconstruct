import logging
from typing import Tuple
from collections import namedtuple
from fast_pruning.data.pruning_dataset import PruningDataset

import tensorflow as tf
import numpy as np

logger = logging.getLogger()

VALID_PADDING_TYPE = ('valid', 'same')
VALID_TENSOR_FORMAT = ('nhwc', 'nchw')
OpRecord = namedtuple('OpRecord',
                      ['update_op', 'xtx_stat', 'yty_stat', 'xty_stat', 'normalize_term', 'kernel', 'bias', 'shape'])


class TFDataCollector:
    def __init__(self):
        """
        Tensorflow based data collector
        """
        self.op_record_dict = dict()

    @staticmethod
    def create_op_record(x: tf.Tensor,
                         y: tf.Tensor,
                         shape: Tuple,
                         has_bias=False,
                         kernel: tf.Variable = None,
                         bias: tf.Variable = None) -> OpRecord:
        if has_bias:
            ones = tf.fill([tf.shape(x)[0], 1], 1.0)
            x = tf.concat([x, ones], axis=1)
        logger.info(f'create op with shape x: {x.shape}, y: {y.shape}')
        xtx = tf.linalg.matmul(x, x, transpose_a=True)
        xty = tf.linalg.matmul(x, y, transpose_a=True)
        yty = tf.linalg.matmul(y, y, transpose_a=True)

        xtx_stat = tf.Variable(initial_value=np.zeros(xtx.shape), dtype=tf.float32)
        xty_stat = tf.Variable(initial_value=np.zeros(xty.shape), dtype=tf.float32)
        yty_stat = tf.Variable(initial_value=np.zeros(yty.shape), dtype=tf.float32)

        normalize_term = tf.Variable(initial_value=0, dtype=tf.int32)
        update_op = tf.group(xtx_stat.assign_add(xtx), xty_stat.assign_add(xty), yty_stat.assign_add(yty),
                             normalize_term.assign_add(tf.shape(x)[0]))

        return OpRecord(update_op=update_op,
                        xtx_stat=xtx_stat,
                        yty_stat=yty_stat,
                        xty_stat=xty_stat,
                        normalize_term=normalize_term,
                        kernel=kernel,
                        bias=bias,
                        shape=shape)

    def register_conv_op(self,
                         op_name: str,
                         input_tensor: tf.Tensor,
                         output_tensor: tf.Tensor,
                         kernel_size: Tuple[int, int] = (3, 3),
                         strides: int = 1,
                         has_bias: bool = False,
                         padding: str = 'same',
                         tensor_format: str = 'nhwc',
                         kernel: tf.Variable = None,
                         bias: tf.Variable = None):
        """
        function for register conv op

        Parameters
        ----------
        op_name: str
            Unique name of this op
        input_tensor: tf.Tensor
            Input tensor of this op
        output_tensor: tf.Tensor
            Output Tensor of this op
        kernel_size: Tuple[int, int]
            2d tuple of kernel size
        strides: int
            strides of convolution
        padding: str
            padding method of convolution
        tensor_format: str
            format of input and output tensor
        has_bias: bool
            this op include bias or not
        kernel: tf.Variable
            Original kernel weigh matrix
        bias: tf.Variable
            Original bias weight matrix
        """
        logger.info(f'register conv_op: {op_name}')
        if op_name in self.op_record_dict:
            raise KeyError('op_name: {op_name} is already registered'.format(op_name=op_name))

        padding = padding.lower()
        if padding not in VALID_PADDING_TYPE:
            raise ValueError('invalid padding type: {}, it should be one of {}'.format(padding, VALID_PADDING_TYPE))

        tensor_format = tensor_format.lower()
        if tensor_format not in VALID_TENSOR_FORMAT:
            raise ValueError('invalid tensor_format: {}, it should be one of {}'.format(
                tensor_format, VALID_TENSOR_FORMAT))

        if tensor_format != 'nhwc':
            raise NotImplementedError('only support nhwc tensorflow format currently')

        x = tf.image.extract_patches(images=input_tensor,
                                     sizes=(1, kernel_size[0], kernel_size[1], 1),
                                     strides=(1, strides, strides, 1),
                                     rates=(1, 1, 1, 1),
                                     padding=padding.upper(),
                                     name=None)
        x = tf.reshape(x, (-1, kernel_size[0] * kernel_size[1] * int(input_tensor.shape[3])))
        y = tf.reshape(output_tensor, (-1, output_tensor.shape[3]))

        self.op_record_dict[op_name] = self.create_op_record(x=x,
                                                             y=y,
                                                             shape=(*kernel_size, input_tensor.shape[-1],
                                                                    output_tensor.shape[-1]),
                                                             has_bias=has_bias,
                                                             kernel=kernel,
                                                             bias=bias)

    def register_fc_op(self,
                       op_name: str,
                       input_tensor: tf.Tensor,
                       output_tensor: tf.Tensor,
                       has_bias: bool = False,
                       kernel: tf.Variable = None,
                       bias: tf.Variable = None):
        """
        function for register fc op

        Parameters
        ----------
        op_name: str
            Unique name of this op
        input_tensor: tf.Tensor
            Input tensor of this op
        output_tensor: tf.Tensor
            Output Tensor of this op
        has_bias: bool
            this op include bias or not
        kernel: tf.Variable
            Original kernel weigh matrix
        bias: tf.Variable
            Original bias weight matrix
        """
        logger.info(f'register fc_op: {op_name}')
        logger.info(f'input tensor shape: {input_tensor.shape}, output tensor shape : {output_tensor.shape}')
        if op_name in self.op_record_dict:
            raise KeyError('op_name: {op_name} is already registered'.format(op_name=op_name))
        x = input_tensor
        y = output_tensor
        self.op_record_dict[op_name] = self.create_op_record(x=x,
                                                             y=y,
                                                             shape=(input_tensor.shape[-1], output_tensor.shape[-1]),
                                                             has_bias=has_bias,
                                                             kernel=kernel,
                                                             bias=bias)

    def get_update_op(self):
        return tf.group(*[record.update_op for record in self.op_record_dict.values()])

    def get_op_record(self, op_name) -> OpRecord:
        return self.op_record_dict[op_name]

    def create_pruning_dataset(self, sess: tf.Session):
        dataset = PruningDataset()

        for op_name, record in self.op_record_dict.items():
            xtx, xty, yty = sess.run([
                record.xtx_stat / tf.cast(record.normalize_term, tf.float32),
                record.xty_stat / tf.cast(record.normalize_term, tf.float32),
                record.yty_stat / tf.cast(record.normalize_term, tf.float32)
            ])
            if record.kernel is not None:
                kernel = sess.run(record.kernel)
            else:
                kernel = None

            if record.bias is not None:
                bias = sess.run(record.bias)
            else:
                bias = None
            dataset.insert_record(op_name=op_name,
                                  xtx=xtx,
                                  xty=xty,
                                  yty=yty,
                                  shape=list(map(int, record.shape)),
                                  kernel=kernel,
                                  bias=bias)

        return dataset
