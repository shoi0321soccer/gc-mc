""" Experiment runner for the model with knowledge graph attached to interaction data """

from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

from gcmc.preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features
from gcmc.model import RecommenderGAE, RecommenderSideInfoGAE
from gcmc.utils import construct_feed_dict
from data_utils import data_iterator
from process_music import process_mpd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from sklearn.feature_extraction import DictVectorizer

def main_process(user_embeddings, item_embeddings, playlists_tracks, test_playlists, train_playlists_count):
    output_file = 'output_lightFM.csv'    
    fuse_perc = 0.7
    dv = DictVectorizer()
    dv.fit_transform(playlists_tracks)
    with open(output_file, 'w') as fout:
        print('team_info,shoiTK,creative,shoi0321soccer@gmail.com', file=fout)
        for i, playlist in enumerate(test_playlists):
            playlist_pos = train_playlists_count + i
            y_pred = user_embeddings[playlist_pos].dot(item_embeddings[playlist_pos//100].T) #+ item_biases
            topn = np.argsort(-y_pred)[:len(playlists_tracks[playlist_pos])+1000]
            rets = [(dv.feature_names_[t], float(y_pred[t])) for t in topn]
            songids = [s for s, _ in rets if s not in playlists_tracks[playlist_pos]]
            songids = sorted(songids,  key=lambda x:x[1], reverse=True)
            print(' , '.join([playlist] + [x for x in songids[:500]]), file=fout)

# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ml_1m", choices=['ml_100k', 'ml_1m', 'ml_10m'],
                help="Dataset string.")


ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=20,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 10],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ac", "--accumulation", type=str, default="stack", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.3,
                help="Dropout fraction")

ap.add_argument("-edo", "--edge_dropout", type=float, default=0.,
                help="Edge dropout rate (1 - keep probability).")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324)")


ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Dataset string ('ml_100k', 'ml_1m')")

ap.add_argument("-bs", "--batch_size", type=int, default=10000,
                help="Batch size used for batching loss function contributions.")

# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)

args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
FEATHIDDEN = args['feat_hidden']
TESTING = args['testing']
BATCHSIZE = args['batch_size']
SYM = args['norm_symmetric']
ACCUM = args['accumulation']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True

if DATASET == 'ml_1m' or DATASET == 'ml_100k':
    NUMCLASSES = 5
elif DATASET == 'ml_10m':
    NUMCLASSES = 10
else:
    raise ValueError('Invalid choice of dataset: %s' % DATASET)

# Splitting dataset in training, validation and test set

if DATASET == 'ml_1m' or DATASET == 'ml_10m':
    datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
elif DATASET == 'ml_100k':
    if FEATURES:
        datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
    else:
        datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'

u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
val_labels, val_u_indices, val_v_indices, test_labels, \
test_u_indices, test_v_indices, class_values = create_trainvaltest_split(DATASET, DATASEED, TESTING,
                                                                         datasplit_path, SPLITFROMFILE, VERBOSE)

adj_train, u_features, v_features, test_playlists, \
train_playlists_count, playlists_tracks = process_mpd(1, 100)

train_labels = []
train_u_indices = []
train_v_indices = []
last_train_labels = adj_train.shape[0]-1

test_labels_set = []
test_u_indices_set = []
test_v_indices_set = []
num_max = 0
num_mini_batch = 0
NUMCLASSES = int(adj_train.max())
class_values = np.arange(1, NUMCLASSES+1)
print("matrix size: ", sys.getsizeof(adj_train), adj_train.shape)
for i, adj in enumerate(adj_train):
  if (i % 100 == 99) or i == last_train_labels:
    adj2 = sp.vstack([adj2, adj])
    adj2 = coo_matrix(adj2.toarray())

    train_labels.append(adj2.data-1)
    train_u_indices.append(adj2.row + 100*(i//100))
    train_v_indices.append(adj2.col)
    
    #print(coo_matrix((data, (row, col))))
    # a = coo_matrix(np.ones((adj2.shape[0], adj2.shape[1])))
    # adj2 = coo_matrix(adj2 + a)
    # test_labels_set.append(adj2.data)
    # test_u_indices_set.append(adj2.row + 100*(i//100))
    # test_v_indices_set.append(adj2.col)
    num_mini_batch += 1
  elif i % 100 == 0:
    adj2 = adj
  else:
    adj2 = sp.vstack([adj2, adj])

train_u = list(set(coo_matrix(adj_train.toarray()).row))
train_v = list(set(coo_matrix(adj_train.toarray()).col))

test_labels = train_labels
test_u_indices = train_u_indices
test_v_indices = train_v_indices

test_playlists_index = list()
for i, _ in enumerate(test_playlists):
  test_playlists_index.append(train_playlists_count + i)

num_users, num_items = adj_train.shape

print("NUMCLASSES:", NUMCLASSES)
print ('num mini batch = ', num_mini_batch)
print("num_users: ", num_users, "num_items: ", num_items)
num_side_features = 0

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features

    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)
    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]
    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

else:
    raise ValueError('Features are not supported in this implementation.')

# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)
for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)
    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)

support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

if SELFCONNECTIONS:
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))

num_support = len(support)
support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div

# Collect all user and item nodes for validation set
val_u = list(set(val_u_indices))
val_v = list(set(val_v_indices))
val_u_dict = {n: i for i, n in enumerate(val_u)}
val_v_dict = {n: i for i, n in enumerate(val_v)}

val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

val_support = support[np.array(val_u)]
val_support_t = support_t[np.array(val_v)]

# features as side info
if FEATURES:
    test_u_features_side = None#u_features_side[np.array(test_u)]
    test_v_features_side = None #v_features_side[np.array(test_v)]

    val_u_features_side = None #u_features_side[np.array(val_u)]
    val_v_features_side = None #v_features_side[np.array(val_v)]

    train_u_features_side = u_features_side[np.array(train_u)]
    train_v_features_side = v_features_side[np.array(train_v)]

else:
    test_u_features_side = None
    test_v_features_side = None

    val_u_features_side = None
    val_v_features_side = None

    train_u_features_side = None
    train_v_features_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'dropout': tf.placeholder_with_default(0., shape=()),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}
print(num_support)
# create model
if FEATURES:
    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   logging=True)
else:
    model = RecommenderGAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_classes=NUMCLASSES,
                           num_support=num_support,
                           self_connections=SELFCONNECTIONS,
                           num_basis_functions=BASES,
                           hidden=HIDDEN,
                           num_users=num_users,
                           num_items=num_items,
                           accum=ACCUM,
                           learning_rate=LR,
                           logging=True)

val_support = sparse_to_tuple(val_support)
val_support_t = sparse_to_tuple(val_support_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
# No dropout for validation and test runs
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side)

# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score = np.inf
best_val_loss = np.inf
best_epoch = 0
wait = 0

print('Training...')
for epoch in range(NB_EPOCH):

  batch_iter = 0
  #data_iter = data_iterator([train_u_indices, train_v_indices, train_labels], batch_size=BATCHSIZE)

  for i in range(num_mini_batch):
    t = time.time()

    train_u_indices_batch = train_u_indices[i]
    train_v_indices_batch = train_v_indices[i]
    train_labels_batch = train_labels[i]
    print(train_v_indices_batch.max())
    print(len(set(train_u_indices_batch)), len(set(train_v_indices_batch)), len(set(train_labels_batch)))
    adj2 = coo_matrix((train_labels_batch, (train_u_indices_batch-i*100, train_v_indices_batch)))
    a = coo_matrix(np.ones((adj2.shape[0], adj2.shape[1])))
    adj2 = coo_matrix(adj2 + a)
    train_labels_batch = adj2.data
    train_u_indices_batch = adj2.row + 100*i
    train_v_indices_batch = adj2.col

    # Collect all user and item nodes for train set
    train_u = list(set(train_u_indices_batch))
    train_v = list(set(train_v_indices_batch))
    train_u_dict = {n: i for i, n in enumerate(train_u)}
    train_v_dict = {n: i for i, n in enumerate(train_v)}

    train_u_indices_batch = np.array([train_u_dict[o] for o in train_u_indices_batch])
    train_v_indices_batch = np.array([train_v_dict[o] for o in train_v_indices_batch])

    train_support_batch = sparse_to_tuple(support[np.array(train_u)])
    train_support_t_batch = sparse_to_tuple(support_t[np.array(train_v)])

    train_feed_dict_batch = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                                v_features_nonzero,
                                                train_support_batch,
                                                train_support_t_batch,
                                                train_labels_batch, train_u_indices_batch,
                                                train_v_indices_batch, class_values, DO)

    # with exponential moving averages
    outs = sess.run([model.embeddings, model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict_batch)
    train_avg_loss = outs[2]
    train_rmse = outs[3]
    print("embeddings:", outs[0][0].shape, outs[0][1].shape)
    #print(len(train_u), len(train_v), len(train_u_indices_batch), len(train_v_indices_batch))

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

    if VERBOSE and batch_iter == num_mini_batch-1:
        print('[*] Iteration: %04d' % (epoch*num_mini_batch + batch_iter),  " Epoch:", '%04d' % epoch,
              "minibatch iter:", '%04d' % batch_iter,
              "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_rmse=", "{:.5f}".format(val_rmse),
              "\t\ttime=", "{:.5f}".format(time.time() - t))

    if val_rmse < best_val_score:
        
        best_val_score = val_rmse
        best_epoch = epoch*num_mini_batch + batch_iter

    if epoch*num_mini_batch+batch_iter % 100 == 0 and not TESTING and False:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

        # load polyak averages
        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, save_path)

        val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

        print('polyak val loss = ', val_avg_loss)
        print('polyak val rmse = ', val_rmse)

        # load back normal variables
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
    batch_iter += 1

# store model including exponential moving averages
saver = tf.train.Saver()
save_path = saver.save(sess, "tmp/%s.ckpt" % model.name, global_step=model.global_step)

if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration', best_epoch)

user_embeddings_vstack = np.ndarray([])
item_embeddings_vstack = []
if TESTING:
    for j in range(len(test_labels_set)):
      #print(j, test_labels_set[j][:5], test_u_indices_set[j][:5], test_v_indices_set[j][:5])
      test_u = list(set(test_u_indices_set[j]))
      test_v = list(set(test_v_indices_set[j]))
      #print(len(test_u), len(test_v))
      test_u_dict = {n: i for i, n in enumerate(test_u)}
      test_v_dict = {n: i for i, n in enumerate(test_v)}
      test_u_indices_batch = np.array([test_u_dict[o] for o in test_u_indices_set[j]])
      test_v_indices_batch = np.array([test_v_dict[o] for o in test_v_indices_set[j]])

      test_support_batch = sparse_to_tuple(support[np.array(test_u)])
      test_support_t_batch = sparse_to_tuple(support_t[np.array(test_v)])

      test_feed_dict_batch = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                                  v_features_nonzero,
                                                  test_support_batch,
                                                  test_support_t_batch,
                                                  test_labels_set[j], test_u_indices_batch,
                                                  test_v_indices_batch, class_values, DO)

      outs = sess.run([model.embeddings, model.loss, model.rmse], feed_dict=test_feed_dict_batch)
      embeddings = outs[0]
      user_embeddings = embeddings[0]
      item_embeddings = embeddings[1]
      if j == 0:
        user_embeddings_vstack = user_embeddings
        item_embeddings_vstack.append(item_embeddings)
      else:
        user_embeddings_vstack = np.vstack((user_embeddings_vstack, user_embeddings))
        item_embeddings_vstack.append(item_embeddings)

      print("Iteration", j, 'test loss:', outs[1], 'test rmse:', outs[2])
else:
    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)
    print('polyak val loss = ', val_avg_loss)
    print('polyak val rmse = ', val_rmse)

print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).iteritems()):
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
results.update({'best_val_score': float(best_val_score), 'best_epoch': best_epoch})
print(json.dumps(results))

main_process(user_embeddings_vstack, item_embeddings_vstack, playlists_tracks, test_playlists, train_playlists_count)

sess.close()
