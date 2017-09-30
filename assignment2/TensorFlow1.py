import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                # print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                #      .format(int(iter_cnt),float(loss),float(np.sum(corr))/actual_batch_size))
                ss = float(np.sum(corr) / actual_batch_size)
                print("Iteration ", iter_cnt, ": with minibatch training loss = ", loss, " and accuracy of", ss)
                # print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                #      .format(iter_cnt,loss, ss))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct


def my_model(X,y,is_training, params, scope = None):
    with tf.variable_scope(scope):
        cparams, hparams = params
        #f_size, f_num, f_stride, pool_size = cparams
        ip_size = 32
        ch_size = 3
        pool_size = 2
        n_size = None

        for i, cp in enumerate(cparams):
            #print (X.shape)
            regularizer = tf.contrib.layers.l2_regularizer(scale = 1.0)
            f_size, f_num, f_stride = cp
            Wconv = tf.get_variable("Wconv"+str(i), shape=[f_size, f_size, ch_size, f_num],regularizer=regularizer)
            bconv = tf.get_variable("bconv"+str(i), shape=[f_num], regularizer=regularizer)
            a = tf.nn.conv2d(X, Wconv, strides = [1, f_stride, f_stride, 1], padding='SAME') + bconv
            h = tf.nn.relu(a)
            ip_size = int(np.ceil(float(ip_size)/float(f_stride)))
            X = tf.reshape(h, [-1, f_num])
            X = tf.layers.batch_normalization(X, training = is_training)
    #        X = tf.reshape(X, [-1, f_num, ip_size, ip_size])
            X = tf.reshape(X, [-1, ip_size, ip_size, f_num])

            ch_size = f_num
            if (i == len(cparams)-1):
                X = tf.nn.max_pool(X, ksize = [1, pool_size, pool_size, 1], strides = [1, 2, 2, 1], padding = 'SAME')
                ip_size = int(np.ceil(float(ip_size)/2.0))
                n_size = f_num*ip_size**2
                X = tf.reshape(X, [-1, n_size])


        for j,h in enumerate(hparams):
            W = tf.get_variable("W"+ str(j), shape=[n_size, h], regularizer = regularizer)
            b = tf.get_variable("b"+str(j), shape=[h], regularizer = regularizer)
            X = tf.matmul(X, W) + b
            X = tf.layers.batch_normalization(X, training = is_training)
            X = tf.nn.relu(X)
            n_size = h

        W = tf.get_variable("WLast", shape=[n_size, 10])
        b = tf.get_variable("bLast"+str(j), shape=[10])
        y_out = tf.matmul(X, W) + b
        return y_out
        pass



tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


params_ls = [([(5, 128, 2), (3, 64, 2), (3, 32, 2)], [100, 100]),
          ([(3, 32, 1), (3, 32, 1), (3, 32, 1), (3, 32, 1)], [100, 100, 100]),
          ([(7, 256, 2), (5, 128, 2), (5, 64, 2), (3, 32, 1), (1, 32, 1)], [30])
         ] #(fs, fn, S) Pl  # HIDDEN DIM

ts_list = []
loss_list = []
y_list = []
lr_list = []
reg_list = []
param_idx = []
n_trials = 20
for n in range(n_trials):

    import random
    lr = 10**(random.uniform(-1.0, -5.0))
    reg = 10**(random.uniform(-1.0, -4.0))
    midx = np.random.randint(low = 0, high = 3)
    midx = 1
    print (n, midx, lr, reg)
    params = params_ls[midx]
    param_idx.append(midx)
    name = "model" + str(n)
    optimizer = tf.train.AdamOptimizer(lr)
    y_out = my_model(X,y,is_training,params,name)
    mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = y_out))
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = mean_loss + reg*reg_loss
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(total_loss )
        ts_list.append(train_step)
        loss_list.append(total_loss)
        y_list.append(y_out)
pass

sess = tf.Session()

sess.run(tf.global_variables_initializer())

max_acc = -1.0
best_n = 0
for n in range(n_trials):
    print (n)
    print('Training')
    run_model(sess,y_list[n],loss_list[n],X_train,y_train,1,64,100,ts_list[n])
    print('Validation')
    total_loss, val_acc = run_model(sess,y_list[n],loss_list[n],X_val,y_val,1,64)
    print ("val_acc", val_acc)
    if (max_acc < val_acc):
        max_acc = val_acc
        best_n = n

best_ts = ts_list[best_n]
best_y = y_list[best_n]
best_ls = loss_list[best_n]


print (lr_list, reg_list,best_n)
print('Training')
run_model(sess,best_y,best_ls,X_train,y_train,1,64)
print('Validation')
run_model(sess,best_y,best_ls,X_val,y_val,1,64)

print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
