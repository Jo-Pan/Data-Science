import os,sys
import time
from sklearn import metrics
import numpy as np
import tensorflow as tf

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data

import sklearn

#------------------------ Flags ----------------------------#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


# --- Things need to be define --- #
flags.DEFINE_string('train_prefix', '', 'output dir name')
flags.DEFINE_string('input_dir0', './data/0Fvecs_15NN/input', 'directory identifying training data')
flags.DEFINE_string('input_dir1', './data/1Fvecs_15NN/input', 'directory identifying training data')
flags.DEFINE_string('input_dir2', './data/2Fvecs_15NN/input', 'directory identifying training data')
flags.DEFINE_string('input_dir3', './data/3Fvecs_15NN/input', 'directory identifying training data')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')


# --- Things can be changed --- #
flags.DEFINE_integer('gpu', 1, "which gpu to use.")

flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')

flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")


# --- Things are left as old --- #
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)
GPU_MEM_FRACTION = 0.8


#------------------------ Evaluation Utils ----------------------------#

def final_activation(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return y_true, y_pred

def calc_f1(y_true, y_pred):
    y_true, y_pred= final_activation(y_true, y_pred)
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_accuracy(y_true, y_pred):
    y_true, y_pred = final_activation(y_true, y_pred)
    return np.mean(y_true == y_pred)

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)

def log_dir():
    log_dir = FLAGS.base_log_dir + "/graphsage_output/" + FLAGS.train_prefix
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    output_features = []
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss, model.outputs1, model.node_preds],
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1

        output_features.append(np.concatenate([batch_labels, node_outs_val[0], node_outs_val[2], node_outs_val[3]], axis=1))

    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    output_features = np.vstack(output_features)

    f1_scores = calc_f1(labels, val_preds)
    accuracy = calc_accuracy(labels, val_preds)

    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test), accuracy, output_features


#------------------------ Train ----------------------------#
def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

# def construct_placeholders(num_classes):
#     # Define placeholders
#     placeholders = {
#         'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
#
#         'batch-0' : tf.placeholder(tf.int32, shape=(None), name='batch-01'),
#         'batch-1': tf.placeholder(tf.int32, shape=(None), name='batch-11'),
#         'batch-2': tf.placeholder(tf.int32, shape=(None), name='batch-21'),
#         'batch-3': tf.placeholder(tf.int32, shape=(None), name='batch-31'),
#
#         'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
#         'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
#     }
#     return placeholders


def train(train_datas, test_data=None):
    # same for all data
    id_map = train_datas[0][2]
    class_map = train_datas[0][4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    placeholders = construct_placeholders(num_classes)

    data_dicts = []
    for data_id, train_data in enumerate(train_datas):
        G = train_data[0]
        features = train_data[1]
        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        context_pairs = train_data[3] if FLAGS.random_context else None
        minibatch = NodeMinibatchIterator(G,
                                          id_map,
                                          placeholders,
                                          class_map,
                                          num_classes,
                                          batch_size=FLAGS.batch_size,
                                          max_degree=FLAGS.max_degree,
                                          context_pairs=context_pairs,
                                          data_post='-{}'.format(data_id))

        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info-{}".format(data_id))
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        data_dict = {'G': G,
                     'features': features,
                     'context_pairs':context_pairs,
                     'minibatch': minibatch,
                     'layer_infos': layer_infos,
                     'adj_info': adj_info,
                     'adj_info_ph': adj_info_ph
        }
        data_dicts.append(data_dict)

    # ========================= old ============================#
    model = SupervisedGraphsage(num_classes,
                                placeholders,
                                features,
                                adj_info,
                                minibatch.deg,
                                layer_infos=layer_infos,
                                aggregator_type="maxpool",
                                model_size=FLAGS.model_size,
                                sigmoid_loss=FLAGS.sigmoid,
                                identity_dim=FLAGS.identity_dim,
                                logging=True)



    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    del G, id_map, class_map, features

    # Train model
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch + 1), FLAGS.train_prefix)
        epoch_val_costs.append(0)

        if epoch == FLAGS.epochs - 1:
            output_features = []

        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()

            if epoch == FLAGS.epochs - 1:
                outs = sess.run([merged, model.opt_op, model.loss, model.preds, model.outputs1, model.node_preds],
                                feed_dict=feed_dict)
                output_features.append(np.concatenate([labels, outs[3], outs[4], outs[5]], axis=1))

            else:
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)

            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    print("Performing only testing, but not validation")
                    val_cost, val_f1_mic, val_f1_mac, duration, _, _ = incremental_evaluate(sess, model, minibatch,
                                                                                            FLAGS.batch_size, test=True)
                else:
                    val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch,
                                                                          FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[3])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    if epoch == FLAGS.epochs - 1:
        output_features = np.vstack(output_features)
        print('Train output_features shape ', output_features.shape)
        np.savetxt('{}/train_output_features.txt'.format(log_dir()), output_features)

    print("Optimization Finished!")

    sess.run(val_adj_info.op)
    if FLAGS.validate_batch_size != -1:
        print("Performing final validation ....")
        val_cost, val_f1_mic, val_f1_mac, duration, val_accuracy, output_features = incremental_evaluate(sess, model,
                                                                                                         minibatch,
                                                                                                         FLAGS.batch_size)
        print('Validation output_features shape ', output_features.shape)
        np.savetxt('{}/valid_output_features.txt'.format(log_dir()), output_features)

        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "f1_micro=", "{:.5f}".format(val_f1_mic),
              "f1_macro=", "{:.5f}".format(val_f1_mac),
              "accuracy=", "{:.5f}".format(val_accuracy),
              "time=", "{:.5f}".format(duration))
        with open(log_dir() + "valid_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} accuracy={:.5f} time={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac, val_accuracy, duration))

    print("Performing final testing ....")
    val_cost, val_f1_mic, val_f1_mac, duration, val_accuracy, output_features = incremental_evaluate(sess, model,
                                                                                                     minibatch,
                                                                                                     FLAGS.batch_size,
                                                                                                     test=True)
    print('Testing output_features shape ', output_features.shape)
    np.savetxt('./{}/test_output_features.txt'.format(log_dir()), output_features)

    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} accuracy={:.5f}".
                 format(val_cost, val_f1_mic, val_f1_mac, val_accuracy))


def main(argv=None):
    print("Using Parameters: batch_size = {}; epoch = {}; train_prefix ={} ".format(FLAGS.batch_size, FLAGS.epochs,
                                                                                    FLAGS.train_prefix))
    print("Loading training data..")
    train_datas = []
    input_dirs = [FLAGS.input_dir0, FLAGS.input_dir1, FLAGS.input_dir2, FLAGS.input_dir3]
    for input_dir in input_dirs:
        train_datas.append(load_data(input_dir))
    print("Done loading {} set of training data..".format(len(input_dirs)))
    train(train_datas)


if __name__ == '__main__':
    tf.app.run()
