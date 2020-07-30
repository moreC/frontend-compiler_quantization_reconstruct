import logging
import os
import pickle

import tensorflow as tf
import sys
sys.path.append('/hdd1/lizhilong/project/frontendcompiler/')
import src
import json
import numpy as np

# from examples.imagenet.imagenet_input import ImagenetInputBuilder
from fast_pruning.core.tf_data_collector import TFDataCollector
from fast_pruning.data.pruning_dataset import PruningDataset
from fast_pruning.core.model_pruner import ModelPruner

from tqdm import tqdm


logging.basicConfig(format='%(asctime)s[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model():

    with open('resnet50_v1b/IR_fused_for_CModel_graph.json', 'r') as f:
        jgraph = json.load(f)
    params = np.load('resnet50_v1b/IR_fused_for_CModel_params.npz', allow_pickle=True)['arr_0'][()]
    params = src.transform_weight_from_mxnet_to_tensorflow(params)
    trc = src.TFReconstructor(jgraph, params)
    trc._execute()

    x = tf.get_default_graph().get_tensor_by_name("0:0")
    y = tf.placeholder(tf.int64, (None,))
    logits = tf.layers.flatten(tf.get_default_graph().get_tensor_by_name("506:0"))

    op_names = []
    in_tensors = []
    out_tensors = []
    kernel_list = []
    bias_list = []

    for operation in tf.get_default_graph().get_operations():
        if operation.type == 'Conv2D':
            in_tensors.append(operation.inputs._inputs[0])
            out_tensors.append(operation.outputs[0])
            kernel_list.append(operation.inputs._inputs[1])
            bias_list.append(None)
            op_names.append(operation.inputs._inputs[1].name)

    # replace all the kernel by variable
    kernel_name_val_mapping = {kernel.name: tf.make_ndarray(kernel.op.get_attr('value')) for kernel in kernel_list}
    tf.io.write_graph(tf.get_default_graph(), './', 'temp.pb', as_text=False)
    tf.reset_default_graph()
    graphdef = tf.GraphDef()
    with open('./temp.pb', 'rb') as f:
        graphdef.ParseFromString(f.read())

    kernel_var_mapping = {name: tf.Variable(val) for name, val in kernel_name_val_mapping.items()}
    tf.graph_util.import_graph_def(graphdef, input_map=kernel_var_mapping)
    x = tf.get_default_graph().get_tensor_by_name("import/0:0")
    y = tf.placeholder(tf.int64, (None, ))
    logits = tf.layers.flatten(tf.get_default_graph().get_tensor_by_name("import/506:0"))
    kernel_list = [kernel_var_mapping[kernel.name] for kernel in kernel_list]

    op_names = []
    in_tensors = []
    out_tensors = []
    # kernel_list = []
    bias_list = []

    for operation in tf.get_default_graph().get_operations():
        if operation.type == 'Conv2D':
            in_tensors.append(operation.inputs._inputs[0])
            out_tensors.append(operation.outputs[0])
            # kernel_list.append(operation.inputs._inputs[1])
            bias_list.append(None)
            op_names.append(operation.inputs._inputs[1].name)
    return x, y, logits, op_names, kernel_list, bias_list, in_tensors, out_tensors

def get_label(image_list):
    def _label(image_path):
        return image_path.split('/')[-2]
    return np.array([_label(p) for p in image_list])

def load_data(image_list_file):
    # image_input_builder = ImagenetInputBuilder('/home/ubuntu/weberwu/fast_pruning/data/',
    #                                            training_epochs=1,  # not use
    #                                            batch_size=32,  # not used
    #                                            eval_batch_size=32,
    #                                            num_training_images=1,  # not use
    #                                            num_classes=1000)
    # # we use validation for training
    # train_data = image_input_builder.eval_input_fn(split='validation')
    # valid_input_data = image_input_builder.eval_input_fn(split='test')
    label_dict = src.readlines('imagenet_lsvrc_2015_synsets.txt')
    image_list = src.readlines(image_list_file)
    batch_size = 25
    image_batches = [image_list[i:i+batch_size] for i in range(
        0, len(image_list), batch_size)]

    x = []
    y = []
    for image_files_batch in image_batches:
        image_batch = src.process_image_batch(image_files_batch)
        label_batch = get_label(image_files_batch)
        label_batch = [label_dict.index(l) for l in label_batch]
        x.append(image_batch)
        y.append(label_batch)
    return x, y


def classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), label)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return loss, accuracy


def evaluate(x, y, data, accuracy_op):
    num_examples = 0
    total_accuracy = 0
    sess = tf.get_default_session()
    allx, ally = data
    for batch_x, batch_y in zip(allx, ally):
        accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        num_examples += len(batch_x)
    return total_accuracy / num_examples


def restore_evaluate(pruned_weights):
    tf.reset_default_graph()
    x, y, logits, op_names, kernel_list, bias_list, in_tensors, out_tensors = create_model()
    allx, ally = load_data('imagenet_5000test.list')
    loss, accuracy_op = classification_loss(logit=logits, label=y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        accuracy_orig = evaluate(x, y, [allx, ally], accuracy_op)

        logger.info('pruned_weights is not None, reset weights from it and re-evaluate')
        for op_name, kernel, bias in zip(op_names, kernel_list, bias_list):
            if op_name not in pruned_weights:
                logger.warning(f'op_name: {op_name} not in pruned_weights. Skip.')
            else:
                logger.info(f'restore {op_name}')
                # import pdb; pdb.set_trace()
                sess.run(
                    kernel.assign(
                        tf.reshape(tf.constant(pruned_weights[op_name], dtype=tf.float32),
                                   shape=kernel.shape)))
        accuracy = evaluate(x, y, [allx, ally], accuracy_op)
        logger.info(f'train accuracy: {accuracy_orig}')
        logger.info(f'pruned train accuracy: {accuracy}')


def build_pruning_dataset(output_path):
    tf.reset_default_graph()
    x, y, logits, op_names, kernel_list, bias_list, in_tensors, out_tensors = create_model()
    loss, accuracy_op = classification_loss(logit=logits, label=y)
    tf_data_collector = TFDataCollector()

    for op_name, kernel, bias, in_tensor, out_tensor in zip(op_names, kernel_list, bias_list, in_tensors, out_tensors):
        if in_tensor.shape[1] <= out_tensor.shape[1] + 2:
            strides = 1
        else:
            strides = 2
        padding = 'valid'
        tf_data_collector.register_conv_op(op_name=op_name,
                                           kernel_size=(int(kernel.shape[0]), int(kernel.shape[1])),
                                           padding=padding,
                                           strides=strides,
                                           input_tensor=in_tensor,
                                           output_tensor=out_tensor,
                                           kernel=kernel,
                                           bias=bias,
                                           has_bias=bias is not None)

    update_op = tf_data_collector.get_update_op()
    allx, ally = load_data('imagenet_100test.list')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger.info('Update statistics')
        total_accuracy = 0.0
        num_examples = 0
        for batch_x, batch_y in zip(allx, ally):
            accuracy, _ = sess.run([accuracy_op, update_op], feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
            num_examples += len(batch_x)

        logger.info(f"finish collecting data with accuracy {total_accuracy / num_examples}...")
        dataset = tf_data_collector.create_pruning_dataset(sess)
        dataset.dump(output_path)


def run_pruning_algorithm(pruning_data_path, output_path):
    dataset = PruningDataset()
    dataset.load(pruning_data_path)
    logger.info('pruning on dataset with variables:')
    for k, r in dataset.get_record_dict().items():
        logger.info(f'{k} {r.shape}')

    model_pruner = ModelPruner(num_workers=16)
    pruned_weights_dict = model_pruner.pruning(dataset=dataset,
                                               target_sparsity_dict={
                                                   'params/resnetv1b_conv0_weight:0': 0.0,
                                                   'params/resnetv1b_layers1_conv0_weight:0': 0.0,
                                                   'params/resnetv1b_layers1_conv1_weight:0': 0.0,
                                                   'params/resnetv1b_down1_conv0_weight:0': 0.0,
                                                   'params/resnetv1b_down2_conv0_weight:0': 0.0,
                                                   'params/resnetv1b_down3_conv0_weight:0': 0.0,
                                                   'params/resnetv1b_down4_conv0_weight:0': 0.0,
                                               },
                                               default_sparsity=0.667,
                                               cross_channel=False)
    with open(output_path, 'wb') as f:
        pickle.dump(pruned_weights_dict, f, protocol=-1)

def main():
    exp_base_dir = 'experiments/resnet_v1_pb_exp_1'
    pruning_data_path = os.path.join(exp_base_dir, 'pruning_dataset/resnet_50_v1')
    build_pruning_dataset(output_path=pruning_data_path)
    pruned_weights_path = os.path.join(exp_base_dir, 'pruned_weights.pkl')
    run_pruning_algorithm(pruning_data_path=pruning_data_path, output_path=pruned_weights_path)
    pruned_weights = pickle.load(open(pruned_weights_path, 'rb'))
    restore_evaluate(pruned_weights=pruned_weights)

if __name__ == '__main__':
    main()
